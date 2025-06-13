"""
Topology-Aware Training Example for torchLoom

This example demonstrates allreduce topology optimization via real-time network profiling.
The training process adapts its communication topology based on:
- Real-time network bandwidth measurements
- Latency profiling between nodes
- Dynamic topology reconfiguration
- Performance optimization based on network conditions
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
from typing import Dict, List, Any, Optional

# Add torchLoom imports
sys.path.insert(0, '/srv/apps/torchLoom')  # HACK: Should be configured properly
from torchLoom.threadlet.threadlet import Threadlet
from torchLoom.common.utils import maybe_get_device_uuid

# Import utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from utils import get_cifar10_dataset

# Import topology optimization components
from network_profiler import NetworkProfiler
from topology_optimizer import TopologyOptimizer, TopologyType
from bandwidth_simulator import BandwidthSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TopologyAwareModel(nn.Module):
    """Model with topology-aware communication patterns."""
    
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
        
        # Track gradient communication metrics
        self.communication_stats = {
            'allreduce_time': [],
            'bandwidth_utilization': [],
            'topology_switches': 0
        }
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class TopologyAwareTrainer:
    """Main trainer with topology optimization capabilities."""
    
    def __init__(self, config):
        self.config = config
        self.process_id = f"topology-aware-{uuid.uuid4().hex[:8]}"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rank = 0  # Simulated rank for demo
        self.world_size = config.get('world_size', 4)
        
        # Model and training components
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = None
        self.val_loader = None
        
        # Topology optimization components
        self.threadlet = None
        self.network_profiler = None
        self.topology_optimizer = None
        self.bandwidth_simulator = None
        self.current_topology = TopologyType.RING
        
        # Training metrics
        self.training_metrics = {
            'epoch': 0,
            'batch': 0,
            'loss': 0.0,
            'accuracy': 0.0,
            'communication_efficiency': 0.0,
            'topology_switches': 0,
            'avg_allreduce_time': 0.0
        }
        
        self._setup_components()
    
    def _setup_components(self):
        """Initialize all training and topology optimization components."""
        self._setup_model()
        self._setup_data()
        self._setup_threadlet()
        self._setup_network_profiling()
        self._setup_topology_optimization()
    
    def _setup_model(self):
        """Initialize model and optimizer."""
        self.model = TopologyAwareModel(num_classes=10).to(self.device)
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
        """Setup threadlet for topology coordination."""
        try:
            device_uuid = maybe_get_device_uuid()
            self.threadlet = Threadlet(
                process_id=self.process_id,
                device_uuid=f"topology-aware-{device_uuid}"
            )
            
            # Register handlers for topology optimization commands
            self.threadlet.register_handler("switch_topology", self._handle_topology_switch)
            self.threadlet.register_handler("update_bandwidth", self._handle_bandwidth_update)
            self.threadlet.register_handler("optimize_topology", self._handle_topology_optimization)
            self.threadlet.register_handler("profile_network", self._handle_network_profiling)
            
            self.threadlet.start()
            logger.info(f"Threadlet initialized for process: {self.process_id}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize threadlet: {e}")
            self.threadlet = None
    
    def _setup_network_profiling(self):
        """Setup network profiling for topology optimization."""
        self.network_profiler = NetworkProfiler(
            process_id=self.process_id,
            rank=self.rank,
            world_size=self.world_size,
            threadlet=self.threadlet
        )
        
        # Setup bandwidth simulation
        self.bandwidth_simulator = BandwidthSimulator(
            process_id=self.process_id,
            num_nodes=self.world_size
        )
        
        self.network_profiler.start()
        self.bandwidth_simulator.start()
        
        logger.info("Network profiling and bandwidth simulation initialized")
    
    def _setup_topology_optimization(self):
        """Setup topology optimization system."""
        self.topology_optimizer = TopologyOptimizer(
            process_id=self.process_id,
            world_size=self.world_size,
            network_profiler=self.network_profiler,
            threadlet=self.threadlet
        )
        
        # Start topology optimization
        self.topology_optimizer.start()
        
        logger.info("Topology optimization system initialized")
    
    def _handle_topology_switch(self, new_topology_str):
        """Handle topology switch command from weaver."""
        try:
            new_topology = TopologyType(new_topology_str)
            old_topology = self.current_topology
            
            self.current_topology = new_topology
            self.training_metrics['topology_switches'] += 1
            
            logger.info(f"Topology switched: {old_topology.value} -> {new_topology.value}")
            
            # Update model communication strategy
            self._configure_model_topology(new_topology)
            
            # Publish topology change event
            if self.threadlet:
                self.threadlet.publish_event('topology_changed', {
                    'old_topology': old_topology.value,
                    'new_topology': new_topology.value,
                    'timestamp': time.time()
                })
            
        except ValueError as e:
            logger.error(f"Invalid topology type: {new_topology_str}: {e}")
    
    def _handle_bandwidth_update(self, bandwidth_data):
        """Handle bandwidth update from network profiling."""
        if self.bandwidth_simulator:
            self.bandwidth_simulator.update_bandwidth_matrix(bandwidth_data)
        
        # Trigger topology optimization based on new bandwidth data
        if self.topology_optimizer:
            self.topology_optimizer.optimize_based_on_bandwidth(bandwidth_data)
    
    def _handle_topology_optimization(self, optimization_params):
        """Handle topology optimization request."""
        if self.topology_optimizer:
            optimal_topology = self.topology_optimizer.find_optimal_topology(
                **optimization_params
            )
            
            if optimal_topology != self.current_topology:
                self._handle_topology_switch(optimal_topology.value)
    
    def _handle_network_profiling(self, profile_config):
        """Handle network profiling request."""
        if self.network_profiler:
            profile_results = self.network_profiler.run_comprehensive_profile(
                **profile_config
            )
            
            # Publish profiling results
            if self.threadlet:
                self.threadlet.publish_status({
                    'network_profile': profile_results,
                    'timestamp': time.time()
                })
    
    def _configure_model_topology(self, topology: TopologyType):
        """Configure model's communication topology."""
        if topology == TopologyType.RING:
            self._configure_ring_topology()
        elif topology == TopologyType.TREE:
            self._configure_tree_topology()
        elif topology == TopologyType.BUTTERFLY:
            self._configure_butterfly_topology()
        elif topology == TopologyType.ALL_TO_ALL:
            self._configure_all_to_all_topology()
        elif topology == TopologyType.HIERARCHICAL:
            self._configure_hierarchical_topology()
    
    def _configure_ring_topology(self):
        """Configure ring allreduce topology."""
        logger.info("Configuring ring topology for allreduce")
        # In a real implementation, this would modify PyTorch's distributed backend
        # to use ring-based communication patterns
    
    def _configure_tree_topology(self):
        """Configure tree-based allreduce topology."""
        logger.info("Configuring tree topology for allreduce")
        # Tree topology for better scalability with large number of nodes
    
    def _configure_butterfly_topology(self):
        """Configure butterfly topology for allreduce."""
        logger.info("Configuring butterfly topology for allreduce")
        # Butterfly topology for optimal bandwidth utilization
    
    def _configure_all_to_all_topology(self):
        """Configure all-to-all topology."""
        logger.info("Configuring all-to-all topology for allreduce")
        # Direct all-to-all communication
    
    def _configure_hierarchical_topology(self):
        """Configure hierarchical topology."""
        logger.info("Configuring hierarchical topology for allreduce")
        # Hierarchical approach for multi-level network architectures
    
    def _simulate_allreduce(self, tensor_size_mb: float) -> float:
        """Simulate allreduce operation and measure performance."""
        start_time = time.time()
        
        # Get current network conditions
        network_conditions = self.network_profiler.get_current_conditions()
        bandwidth_matrix = self.bandwidth_simulator.get_current_bandwidth_matrix()
        
        # Calculate allreduce time based on topology and network conditions
        allreduce_time = self.topology_optimizer.estimate_allreduce_time(
            topology=self.current_topology,
            tensor_size_mb=tensor_size_mb,
            bandwidth_matrix=bandwidth_matrix,
            latency_matrix=network_conditions.get('latency_matrix')
        )
        
        # Add some realistic variance
        allreduce_time *= (0.8 + 0.4 * torch.rand(1).item())
        
        # Simulate the communication delay
        time.sleep(min(allreduce_time / 1000.0, 0.1))  # Convert ms to seconds, cap at 100ms
        
        actual_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Update communication stats
        self.model.communication_stats['allreduce_time'].append(actual_time)
        
        # Calculate bandwidth utilization
        bandwidth_utilization = self._calculate_bandwidth_utilization(
            tensor_size_mb, actual_time, bandwidth_matrix
        )
        self.model.communication_stats['bandwidth_utilization'].append(bandwidth_utilization)
        
        return actual_time
    
    def _calculate_bandwidth_utilization(self, tensor_size_mb: float, 
                                       allreduce_time_ms: float, 
                                       bandwidth_matrix: Dict) -> float:
        """Calculate bandwidth utilization efficiency."""
        if allreduce_time_ms == 0:
            return 0.0
        
        # Theoretical minimum time based on available bandwidth
        max_bandwidth = max(max(row) for row in bandwidth_matrix['matrix'])
        theoretical_min_time = (tensor_size_mb * 8) / max_bandwidth * 1000  # ms
        
        # Utilization is theoretical minimum / actual time
        utilization = min(1.0, theoretical_min_time / allreduce_time_ms)
        return utilization
    
    def train_epoch(self, epoch):
        """Train for one epoch with topology optimization."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Simulate gradient allreduce with topology optimization
            gradient_size_mb = self._estimate_gradient_size()
            allreduce_time = self._simulate_allreduce(gradient_size_mb)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Calculate communication efficiency
            avg_allreduce_time = sum(self.model.communication_stats['allreduce_time'][-10:]) / min(10, len(self.model.communication_stats['allreduce_time']))
            avg_bandwidth_util = sum(self.model.communication_stats['bandwidth_utilization'][-10:]) / min(10, len(self.model.communication_stats['bandwidth_utilization']))
            
            self.training_metrics.update({
                'epoch': epoch,
                'batch': batch_idx,
                'loss': total_loss / (batch_idx + 1),
                'accuracy': 100. * correct / total,
                'communication_efficiency': avg_bandwidth_util * 100,
                'avg_allreduce_time': avg_allreduce_time
            })
            
            # Periodic status reporting and optimization
            if batch_idx % 20 == 0:
                self._publish_training_status()
                self._trigger_topology_optimization()
                
            # Adaptive topology switching based on performance
            if batch_idx % 50 == 0:
                self._adaptive_topology_decision()
        
        logger.info(f"Epoch {epoch} completed - Loss: {total_loss/len(self.train_loader):.4f}, "
                   f"Accuracy: {100.*correct/total:.2f}%, "
                   f"Comm Efficiency: {self.training_metrics['communication_efficiency']:.1f}%")
    
    def _estimate_gradient_size(self) -> float:
        """Estimate gradient size in MB."""
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # Assume float32 (4 bytes per parameter)
        size_mb = (total_params * 4) / (1024 * 1024)
        return size_mb
    
    def _trigger_topology_optimization(self):
        """Trigger topology optimization based on current conditions."""
        if self.topology_optimizer:
            # Get recent performance metrics
            recent_allreduce_times = self.model.communication_stats['allreduce_time'][-10:]
            recent_bandwidth_util = self.model.communication_stats['bandwidth_utilization'][-10:]
            
            if recent_allreduce_times and recent_bandwidth_util:
                avg_time = sum(recent_allreduce_times) / len(recent_allreduce_times)
                avg_util = sum(recent_bandwidth_util) / len(recent_bandwidth_util)
                
                # Trigger optimization if performance is poor
                if avg_util < 0.6 or avg_time > 100:  # Less than 60% utilization or >100ms
                    optimal_topology = self.topology_optimizer.optimize_for_current_conditions()
                    
                    if optimal_topology != self.current_topology:
                        logger.info(f"Performance-based topology optimization: "
                                  f"switching to {optimal_topology.value}")
                        self._handle_topology_switch(optimal_topology.value)
    
    def _adaptive_topology_decision(self):
        """Make adaptive topology decisions based on training phase."""
        # Different topologies might be optimal for different training phases
        current_epoch = self.training_metrics['epoch']
        
        if current_epoch < 2:
            # Early training: prioritize fast iteration
            preferred_topology = TopologyType.RING
        elif current_epoch < 5:
            # Mid training: balance speed and accuracy
            preferred_topology = TopologyType.TREE
        else:
            # Late training: prioritize convergence quality
            preferred_topology = TopologyType.BUTTERFLY
        
        if preferred_topology != self.current_topology:
            confidence_threshold = 0.8
            if self.topology_optimizer.get_topology_confidence(preferred_topology) > confidence_threshold:
                logger.info(f"Adaptive topology decision: switching to {preferred_topology.value} "
                          f"for training phase (epoch {current_epoch})")
                self._handle_topology_switch(preferred_topology.value)
    
    def _publish_training_status(self):
        """Publish current training status including topology metrics."""
        if not self.threadlet:
            return
        
        # Calculate topology performance metrics
        topology_metrics = self._calculate_topology_metrics()
        
        status = {
            'process_id': self.process_id,
            'training_metrics': self.training_metrics.copy(),
            'topology_metrics': topology_metrics,
            'current_topology': self.current_topology.value,
            'device_info': str(self.device),
            'timestamp': time.time()
        }
        
        self.threadlet.publish_status(status)
    
    def _calculate_topology_metrics(self) -> Dict[str, Any]:
        """Calculate topology-specific performance metrics."""
        comm_stats = self.model.communication_stats
        
        if not comm_stats['allreduce_time']:
            return {}
        
        return {
            'avg_allreduce_time_ms': sum(comm_stats['allreduce_time']) / len(comm_stats['allreduce_time']),
            'avg_bandwidth_utilization': sum(comm_stats['bandwidth_utilization']) / len(comm_stats['bandwidth_utilization']),
            'topology_switches': comm_stats.get('topology_switches', self.training_metrics['topology_switches']),
            'communication_overhead_pct': self._calculate_communication_overhead(),
            'topology_efficiency_score': self._calculate_topology_efficiency()
        }
    
    def _calculate_communication_overhead(self) -> float:
        """Calculate communication overhead as percentage of total time."""
        if not self.model.communication_stats['allreduce_time']:
            return 0.0
        
        total_comm_time = sum(self.model.communication_stats['allreduce_time'])
        # Estimate total training time (simplified)
        estimated_total_time = len(self.model.communication_stats['allreduce_time']) * 100  # ms per batch
        
        return min(100.0, (total_comm_time / estimated_total_time) * 100)
    
    def _calculate_topology_efficiency(self) -> float:
        """Calculate overall topology efficiency score."""
        if not self.model.communication_stats['bandwidth_utilization']:
            return 0.0
        
        bandwidth_score = sum(self.model.communication_stats['bandwidth_utilization']) / len(self.model.communication_stats['bandwidth_utilization'])
        
        # Penalty for excessive topology switches
        switch_penalty = max(0, (self.training_metrics['topology_switches'] - 5) * 0.1)
        
        efficiency_score = max(0.0, bandwidth_score - switch_penalty)
        return min(1.0, efficiency_score)
    
    def train(self):
        """Main training loop with topology optimization."""
        logger.info("Starting topology-aware training")
        
        try:
            for epoch in range(self.config.get('num_epochs', 5)):
                self.train_epoch(epoch)
                
                # Validation
                if epoch % 2 == 0:
                    self.validate()
                
                # Epoch-level topology optimization
                self._epoch_topology_optimization(epoch)
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
        finally:
            self._cleanup()
    
    def _epoch_topology_optimization(self, epoch):
        """Perform epoch-level topology optimization."""
        logger.info(f"Performing epoch-level topology optimization for epoch {epoch}")
        
        # Analyze epoch performance
        topology_metrics = self._calculate_topology_metrics()
        
        # Log topology performance
        logger.info(f"Topology Performance - "
                   f"Efficiency: {topology_metrics.get('topology_efficiency_score', 0):.3f}, "
                   f"Bandwidth Util: {topology_metrics.get('avg_bandwidth_utilization', 0):.3f}, "
                   f"Switches: {topology_metrics.get('topology_switches', 0)}")
    
    def validate(self):
        """Validation with topology-aware metrics."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
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
        logger.info("Cleaning up topology-aware trainer")
        
        # Stop optimization components
        if self.topology_optimizer:
            self.topology_optimizer.stop()
        
        if self.network_profiler:
            self.network_profiler.stop()
        
        if self.bandwidth_simulator:
            self.bandwidth_simulator.stop()
        
        # Stop threadlet
        if self.threadlet:
            self.threadlet.stop()
        
        # Final metrics report
        final_metrics = self._calculate_topology_metrics()
        logger.info(f"Final topology metrics: {final_metrics}")


def main():
    """Main entry point for topology-aware training."""
    # Configuration
    config = {
        'learning_rate': 0.01,
        'batch_size': 32,
        'num_epochs': 5,
        'world_size': 4,  # Simulate 4-node distributed training
        'quick_run': True,  # Use synthetic data for demo
    }
    
    # Initialize and start training
    trainer = TopologyAwareTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 