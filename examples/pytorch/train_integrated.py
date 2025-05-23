import argparse
import time
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# torchLoom imports
from torchLoom.weavelet import Weavelet, weavelet_handler

# Simple random dataset for testing
class RandomDataset(Dataset):
    def __init__(self, num_samples, input_shape=(1, 28, 28), num_classes=10):
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.data = torch.randn(num_samples, *input_shape)
        self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class IntegratedTrainer:
    """Training class with full torchLoom integration."""
    
    def __init__(self, args, replica_id=None):
        self.args = args
        self.replica_id = replica_id or f"train_integrated_{uuid.uuid4().hex[:8]}"
        
        # Training parameters (configurable via weavelet)
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.training_enabled = True
        self.verbose = False
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize model and optimizer
        self.model = Net().to(self.device)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.learning_rate)
        
        # Setup datasets
        train_dataset = RandomDataset(args.train_samples)
        test_dataset = RandomDataset(args.test_samples)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)
        
        # Initialize weavelet for torchLoom integration
        self.weavelet = Weavelet(replica_id=self.replica_id)
        self._register_handlers()
        self.weavelet.start()
        
        print(f"🧵 Integrated trainer initialized with replica_id: {self.replica_id}")
        print(f"📊 Will publish training status to torchLoom UI")
    
    def _setup_device(self):
        """Setup compute device."""
        use_accel = not self.args.no_accel and (
            torch.cuda.is_available() or 
            (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
        )
        
        if use_accel:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            print(f"Using accelerator: {device}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        
        return device
    
    def _register_handlers(self):
        """Register weavelet handlers for dynamic configuration."""
        
        @weavelet_handler("learning_rate", float)
        def update_learning_rate(new_lr: float):
            print(f"📈 Learning rate updated: {self.learning_rate} → {new_lr}")
            self.learning_rate = new_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        
        @weavelet_handler("batch_size", int)
        def update_batch_size(new_batch_size: int):
            print(f"📦 Batch size updated: {self.batch_size} → {new_batch_size}")
            self.batch_size = new_batch_size
            # Note: Would need to recreate DataLoader for this to take effect
        
        @weavelet_handler("training_enabled", bool)
        def toggle_training(enabled: bool):
            print(f"⏯️ Training {'enabled' if enabled else 'paused'}")
            self.training_enabled = enabled
        
        @weavelet_handler("verbose", bool)
        def toggle_verbose(enabled: bool):
            print(f"🗣️ Verbose mode {'enabled' if enabled else 'disabled'}")
            self.verbose = enabled
        
        # Register all handlers with the weavelet
        for local_var in locals().values():
            if callable(local_var) and hasattr(local_var, '_weavelet_config_key'):
                config_key = local_var._weavelet_config_key
                expected_type = getattr(local_var, '_weavelet_expected_type', None)
                self.weavelet.register_handler(config_key, local_var, expected_type)
                print(f"✅ Registered handler: {config_key}")
    
    def train_epoch(self, epoch):
        """Train for one epoch with status reporting."""
        self.model.train()
        total_samples = len(self.train_loader.dataset)
        
        # Publish epoch start status
        self._publish_status({
            "type": "epoch_start",
            "epoch": epoch,
            "status": "training",
            "total_batches": len(self.train_loader)
        })
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Check for configuration updates
            self.weavelet.check_and_apply_updates()
            
            # Skip training if disabled
            if not self.training_enabled:
                print("⏸️ Training paused - skipping batch")
                time.sleep(0.1)
                continue
            
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            
            # Publish batch status
            if batch_idx % self.args.log_interval == 0:
                progress_percent = 100. * batch_idx / len(self.train_loader)
                
                if self.verbose:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), total_samples,
                        progress_percent, loss.item()))
                
                # Publish detailed training status
                self._publish_status({
                    "type": "batch_update",
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "step": epoch * len(self.train_loader) + batch_idx,
                    "step_progress": progress_percent,
                    "loss": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "status": "training"
                })
                
                # Publish GPU status (simulated)
                self._publish_gpu_status(batch_idx)
                
                if self.args.dry_run:
                    break
        
        # Publish epoch end status
        self._publish_status({
            "type": "epoch_complete",
            "epoch": epoch,
            "status": "training"
        })
    
    def test_model(self):
        """Test the model and publish results."""
        self.model.eval()
        test_loss = 0
        correct = 0
        total_samples = len(self.test_loader.dataset)
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= total_samples
        accuracy = 100. * correct / total_samples

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, total_samples, accuracy))
        
        # Publish test results
        self._publish_status({
            "type": "test_complete",
            "test_loss": test_loss,
            "accuracy": accuracy,
            "status": "training"
        })
    
    def _publish_status(self, status_data):
        """Publish training status to weaver."""
        try:
            # Add common fields
            status_data.update({
                "replica_id": self.replica_id,
                "device_id": f"device_{self.replica_id}",
                "timestamp": time.time()
            })
            
            # Add configuration info
            status_data["config"] = {
                "batch_size": str(self.batch_size),
                "learning_rate": str(self.learning_rate),
                "optimizer_type": "Adadelta"
            }
            
            self.weavelet.publish_training_status(status_data)
            
        except Exception as e:
            print(f"Warning: Failed to publish status: {e}")
    
    def _publish_gpu_status(self, batch_idx):
        """Publish simulated GPU status."""
        try:
            # Simulate GPU metrics (in real scenario, would use nvidia-ml-py or similar)
            gpu_utilization = 60 + (batch_idx % 30)  # 60-90%
            gpu_temperature = 50 + (batch_idx % 25)   # 50-75°C
            memory_used = 2.0 + (batch_idx % 6) * 0.5  # 2-5 GB
            
            gpu_status = {
                "type": "gpu_update",
                "system": {
                    "gpu_utilization": gpu_utilization,
                    "gpu_temperature": gpu_temperature,
                    "gpu_memory_used": memory_used,
                    "gpu_memory_total": 8.0
                },
                "replica_id": self.replica_id,
                "server_id": "local_server"
            }
            
            self.weavelet.publish_training_status(gpu_status)
            
        except Exception as e:
            print(f"Warning: Failed to publish GPU status: {e}")
    
    def run_training(self):
        """Main training loop."""
        try:
            print(f"🚀 Starting training for {self.args.epochs} epochs...")
            
            # Publish training start status
            self._publish_status({
                "type": "training_start",
                "epochs": self.args.epochs,
                "status": "starting"
            })
            
            for epoch in range(1, self.args.epochs + 1):
                self.train_epoch(epoch)
                self.test_model()
                time.sleep(0.01)  # Brief pause between epochs
            
            # Publish training completion
            self._publish_status({
                "type": "training_complete", 
                "status": "completed"
            })
            
            if self.args.save_model:
                torch.save(self.model.state_dict(), "random_cnn_integrated.pt")
                print("💾 Model saved")
            
        except KeyboardInterrupt:
            print("🛑 Training interrupted by user")
            self._publish_status({
                "type": "training_interrupted",
                "status": "stopped"
            })
        except Exception as e:
            print(f"❌ Training failed: {e}")
            self._publish_status({
                "type": "training_failed",
                "error": str(e),
                "status": "failed"
            })
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'weavelet'):
                self.weavelet.stop()
                print("🧵 Weavelet stopped")
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

def main():
    parser = argparse.ArgumentParser(description='PyTorch Random Data Example with torchLoom Integration')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true',
                        help='For Saving the current Model')
    parser.add_argument('--train-samples', type=int, default=60000,
                        help='Number of random training samples (default: 60000)')
    parser.add_argument('--test-samples', type=int, default=10000,
                        help='Number of random test samples (default: 10000)')
    parser.add_argument('--replica-id', type=str, default=None,
                        help='Replica ID for torchLoom (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    # Create and run integrated trainer
    trainer = IntegratedTrainer(args, replica_id=args.replica_id)
    trainer.run_training()

if __name__ == '__main__':
    main() 