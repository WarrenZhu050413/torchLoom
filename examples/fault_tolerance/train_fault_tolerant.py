"""
Fault Tolerant Training Example for torchLoom

This example demonstrates worker preemption based on external monitoring.
The training process integrates with torchLoom's control plane to handle:
- Worker health monitoring through threadlets
- Automatic worker preemption when issues are detected
- Dynamic recovery and training continuation
- Proactive fault tolerance based on external signals
"""

import logging
import os
import platform
import sys
import threading
import time
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms

# Add torchLoom imports
sys.path.insert(0, '/srv/apps/torchLoom')  # HACK: Should be configured properly
from torchLoom.threadlet.threadlet import Threadlet
from torchLoom.common.utils import maybe_get_device_uuid, get_device_status

# Import utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from utils import get_cifar10_dataset

# Import fault tolerance components
from fault_injector import FaultInjector
from monitoring_simulator import MonitoringSimulator

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pynvml not available ({e}). Device status will be disabled.")
    PYNVML_AVAILABLE = False
    pynvml = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResNet18(nn.Module):
    """Simple ResNet18-like model for demonstration."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class FaultTolerantTrainer:
    """Main trainer class with fault tolerance capabilities."""
    
    def __init__(self, config):
        self.config = config
        self.process_id = f"fault-tolerant-{uuid.uuid4().hex[:8]}"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_preempted = False
        self.training_start_time = None
        
        # Model and training components
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = None
        self.val_loader = None
        
        # Fault tolerance components
        self.threadlet = None
        self.fault_injector = None
        self.monitoring_simulator = None
        self.checkpoint_manager = CheckpointManager(self.process_id)
        
        # Metrics tracking
        self.training_metrics = {
            'epoch': 0,
            'batch': 0,
            'loss': 0.0,
            'accuracy': 0.0,
            'preemptions': 0,
            'recoveries': 0
        }
        
        self._setup_components()
    
    def _setup_components(self):
        """Initialize all training and fault tolerance components."""
        self._setup_model()
        self._setup_data()
        self._setup_threadlet()
        self._setup_fault_tolerance()
        self._setup_monitoring()
    
    def _setup_model(self):
        """Initialize model and optimizer."""
        self.model = ResNet18(num_classes=10).to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.01),
            momentum=0.9,
            weight_decay=1e-4
        )
        logger.info(f"Model initialized on device: {self.device}")
    
    def _setup_data(self):
        """Setup data loaders."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # Use synthetic data for quick testing
        train_dataset = get_cifar10_dataset(
            train=True, 
            transform=transform, 
            quick_run=self.config.get('quick_run', True)
        )
        val_dataset = get_cifar10_dataset(
            train=False, 
            transform=transform, 
            quick_run=self.config.get('quick_run', True)
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=2
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=2
        )
        
        logger.info(f"Data loaders initialized - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    def _setup_threadlet(self):
        """Setup threadlet for fault tolerance coordination."""
        try:
            device_uuid = maybe_get_device_uuid()
            self.threadlet = Threadlet(
                process_id=self.process_id,
                device_uuid=f"fault-tolerant-{device_uuid}"
            )
            
            # Register handlers for fault tolerance commands
            self.threadlet.register_handler("preempt_worker", self._handle_preemption)
            self.threadlet.register_handler("resume_worker", self._handle_resume)
            self.threadlet.register_handler("checkpoint_now", self._handle_checkpoint)
            self.threadlet.register_handler("health_check", self._handle_health_check)
            
            self.threadlet.start()
            logger.info(f"Threadlet initialized for process: {self.process_id}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize threadlet: {e}")
            self.threadlet = None
    
    def _setup_fault_tolerance(self):
        """Setup fault injection for testing."""
        self.fault_injector = FaultInjector(
            process_id=self.process_id,
            failure_probability=self.config.get('failure_probability', 0.1)
        )
    
    def _setup_monitoring(self):
        """Setup external monitoring simulation."""
        self.monitoring_simulator = MonitoringSimulator(
            process_id=self.process_id,
            threadlet=self.threadlet
        )
        self.monitoring_simulator.start()
    
    def _handle_preemption(self, reason="external_signal"):
        """Handle worker preemption."""
        logger.warning(f"Worker preemption triggered: {reason}")
        self.is_preempted = True
        self.training_metrics['preemptions'] += 1
        
        # Save checkpoint before preemption
        self._create_checkpoint()
        
        # Notify weaver of preemption
        if self.threadlet:
            self.threadlet.publish_status({
                'status': 'preempted',
                'reason': reason,
                'timestamp': time.time(),
                'metrics': self.training_metrics
            })
    
    def _handle_resume(self, checkpoint_path=None):
        """Handle worker resume after preemption."""
        logger.info("Resuming worker after preemption")
        self.is_preempted = False
        self.training_metrics['recoveries'] += 1
        
        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        else:
            # Load latest checkpoint
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            if latest_checkpoint:
                self._load_checkpoint(latest_checkpoint)
        
        # Notify weaver of recovery
        if self.threadlet:
            self.threadlet.publish_status({
                'status': 'resumed',
                'timestamp': time.time(),
                'metrics': self.training_metrics
            })
    
    def _handle_checkpoint(self, immediate=True):
        """Handle checkpoint creation request."""
        if immediate:
            self._create_checkpoint()
    
    def _handle_health_check(self):
        """Handle health check request."""
        health_status = {
            'process_id': self.process_id,
            'status': 'preempted' if self.is_preempted else 'healthy',
            'device': str(self.device),
            'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'training_progress': {
                'epoch': self.training_metrics['epoch'],
                'batch': self.training_metrics['batch']
            },
            'timestamp': time.time()
        }
        
        if self.threadlet:
            self.threadlet.publish_status(health_status)
        
        return health_status
    
    def _create_checkpoint(self):
        """Create training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.training_metrics.copy(),
            'config': self.config,
            'timestamp': time.time()
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(checkpoint)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def _load_checkpoint(self, checkpoint_path):
        """Load training checkpoint."""
        try:
            checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_metrics.update(checkpoint['metrics'])
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
    
    def train_epoch(self, epoch):
        """Train for one epoch with fault tolerance."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Check for preemption
            if self.is_preempted:
                logger.info("Training paused due to preemption")
                while self.is_preempted:
                    time.sleep(1)  # Wait for resume signal
                logger.info("Training resumed")
            
            # Inject faults for testing
            if self.fault_injector.should_inject_fault():
                self._handle_preemption("injected_fault")
                continue
            
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            self.training_metrics.update({
                'epoch': epoch,
                'batch': batch_idx,
                'loss': total_loss / (batch_idx + 1),
                'accuracy': 100. * correct / total
            })
            
            # Periodic status reporting
            if batch_idx % 50 == 0:
                self._publish_training_status()
                
            # Periodic checkpointing
            if batch_idx % 200 == 0:
                self._create_checkpoint()
        
        logger.info(f"Epoch {epoch} completed - Loss: {total_loss/len(self.train_loader):.4f}, "
                   f"Accuracy: {100.*correct/total:.2f}%")
    
    def _publish_training_status(self):
        """Publish current training status."""
        if not self.threadlet:
            return
        
        status = {
            'process_id': self.process_id,
            'training_metrics': self.training_metrics.copy(),
            'device_info': str(self.device),
            'timestamp': time.time()
        }
        
        self.threadlet.publish_status(status)
    
    def train(self):
        """Main training loop with fault tolerance."""
        logger.info("Starting fault-tolerant training")
        self.training_start_time = time.time()
        
        try:
            for epoch in range(self.config.get('num_epochs', 10)):
                self.train_epoch(epoch)
                
                # Validation
                if epoch % 2 == 0:
                    self.validate()
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._handle_preemption(f"training_error: {str(e)}")
        finally:
            self._cleanup()
    
    def validate(self):
        """Validation with fault tolerance."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                if self.is_preempted:
                    break
                
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        if total > 0:
            val_accuracy = 100. * correct / total
            val_loss /= len(self.val_loader)
            
            logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
            
            self.training_metrics.update({
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })
            
            self._publish_training_status()
    
    def _cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up fault-tolerant trainer")
        
        # Final checkpoint
        self._create_checkpoint()
        
        # Stop monitoring
        if self.monitoring_simulator:
            self.monitoring_simulator.stop()
        
        # Stop threadlet
        if self.threadlet:
            self.threadlet.stop()
        
        logger.info(f"Training completed - Preemptions: {self.training_metrics['preemptions']}, "
                   f"Recoveries: {self.training_metrics['recoveries']}")


class CheckpointManager:
    """Manages training checkpoints for fault tolerance."""
    
    def __init__(self, process_id):
        self.process_id = process_id
        self.checkpoint_dir = f"./checkpoints/{process_id}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, checkpoint_data):
        """Save checkpoint to disk."""
        timestamp = int(time.time())
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_{timestamp}.pt"
        )
        
        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint from disk."""
        return torch.load(checkpoint_path, map_location='cpu')
    
    def get_latest_checkpoint(self):
        """Get path to latest checkpoint."""
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith('checkpoint_') and f.endswith('.pt')
        ]
        
        if not checkpoints:
            return None
        
        # Sort by timestamp (embedded in filename)
        checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        return os.path.join(self.checkpoint_dir, checkpoints[-1])


def main():
    """Main entry point for fault tolerant training."""
    # Configuration
    config = {
        'learning_rate': 0.01,
        'batch_size': 32,
        'num_epochs': 5,
        'quick_run': True,  # Use synthetic data for demo
        'failure_probability': 0.05  # 5% chance of failure per batch
    }
    
    # Initialize and start training
    trainer = FaultTolerantTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 