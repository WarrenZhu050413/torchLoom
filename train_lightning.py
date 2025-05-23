"""
Lightning-based demo trainer with weavelet integration for torchLoom.

This module provides a lightweight PyTorch Lightning trainer that integrates
with weavelet for dynamic configuration management and publishes training
statistics to the torchft-ui.
"""

import json
import time
import psutil
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import lightning as L
from lightning.pytorch.callbacks import Callback

from torchLoom.lightning_integration import WeaveletLightningModule
from torchLoom.weavelet import weavelet_handler
from torchLoom.constants import torchLoomConstants
from torchLoom.log.logger import setup_logger

logger = setup_logger(name="demo_lightning_trainer", log_file="./torchLoom/log/demo_lightning_train.log")


@dataclass
class LightningTrainingConfig:
    """Configuration for the Lightning training process."""
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 2
    num_samples: int = 2000
    input_dim: int = 50
    hidden_dim: int = 128
    num_classes: int = 5
    device_id: str = "demo_device_1"
    replica_id: str = "demo_replica_1"
    dropout_rate: float = 0.1


class SimpleClassificationDataset(Dataset):
    """Enhanced dataset with more complex patterns."""
    
    def __init__(self, num_samples: int, input_dim: int, num_classes: int, seed: int = 42):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        torch.manual_seed(seed)
        
        # Create structured data for better learning
        self.data = torch.randn(num_samples, input_dim)
        
        # Create meaningful labels with some structure
        weights = torch.randn(input_dim, num_classes) * 0.5
        bias = torch.randn(num_classes) * 0.2
        logits = torch.matmul(self.data, weights) + bias
        
        # Add non-linear patterns
        logits += 0.1 * torch.sin(torch.sum(self.data[:, :5], dim=1)).unsqueeze(1)
        
        self.labels = torch.argmax(logits, dim=1)
        
        # Add controlled noise
        noise_ratio = 0.05
        noise_indices = torch.randperm(num_samples)[:int(num_samples * noise_ratio)]
        self.labels[noise_indices] = torch.randint(0, num_classes, (len(noise_indices),))
        
        logger.info(f"Created dataset: {num_samples} samples, {input_dim} features, {num_classes} classes")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


class SimpleClassificationModel(nn.Module):
    """Simple neural network for classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)


class TorchLoomLightningModule(WeaveletLightningModule):
    """Lightning module with torchLoom weavelet integration."""
    
    def __init__(self, config: LightningTrainingConfig):
        super().__init__(
            replica_id=config.replica_id,
            torchLoom_addr=torchLoomConstants.DEFAULT_ADDR
        )
        
        self.config = config
        self.save_hyperparameters()
        
        # Initialize model
        self.model = SimpleClassificationModel(
            config.input_dim, 
            config.hidden_dim, 
            config.num_classes,
            config.dropout_rate
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state for status publishing
        self.training_start_time = time.time()
        self.best_accuracy = 0.0
        
        logger.info(f"Lightning module initialized with replica_id: {config.replica_id}")
    
    @weavelet_handler("learning_rate", float)
    def update_learning_rate(self, new_lr: float):
        """Handle learning rate updates from weavelet."""
        self.config.learning_rate = new_lr
        # Update optimizer learning rate if trainer and optimizers exist
        try:
            if hasattr(self, '_trainer') and self._trainer is not None:
                optimizers = self._trainer.optimizers
                if optimizers:
                    for optimizer in optimizers:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
        except (RuntimeError, AttributeError):
            # Trainer not attached yet, will use new LR when training starts
            pass
        logger.info(f"Updated learning rate to {new_lr}")
    
    @weavelet_handler("dropout_rate", float)
    def update_dropout_rate(self, new_dropout: float):
        """Handle dropout rate updates from weavelet."""
        self.config.dropout_rate = new_dropout
        # Update model dropout layers
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = new_dropout
        logger.info(f"Updated dropout rate to {new_dropout}")
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }
    
    def _user_training_step(self, batch, batch_idx):
        """Implement the actual training logic."""
        data, target = batch
        output = self.model(data)
        loss = self.criterion(output, target)
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == target).sum().item()
        accuracy = correct / target.size(0)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'predicted': predicted,
            'target': target
        }
    
    def _collect_additional_status(self) -> Optional[Dict[str, Any]]:
        """Collect additional status information for UI publishing."""
        try:
            # Get system metrics
            system_metrics = {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
            }
            
            if torch.cuda.is_available():
                try:
                    system_metrics['gpu_memory_used'] = torch.cuda.memory_allocated() / 1024**3  # GB
                    system_metrics['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    system_metrics['gpu_utilization'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 50.0  # Fallback value
                except:
                    system_metrics['gpu_utilization'] = 50.0  # Default fallback
            
            # Get current learning rate
            current_lr = self.config.learning_rate
            try:
                trainer_ref = getattr(self, '_trainer', None) or getattr(self, 'trainer', None)
                if trainer_ref and hasattr(trainer_ref, 'optimizers') and trainer_ref.optimizers:
                    current_lr = trainer_ref.optimizers[0].param_groups[0]['lr']
            except (RuntimeError, AttributeError):
                # Trainer not attached yet, use config value
                pass
            
            # Get current training metrics if available
            current_loss = None
            current_accuracy = None
            try:
                trainer_ref = getattr(self, '_trainer', None) or getattr(self, 'trainer', None)
                if trainer_ref and hasattr(trainer_ref, 'logged_metrics'):
                    metrics = trainer_ref.logged_metrics
                    current_loss = metrics.get('train_loss_step', metrics.get('train_loss', None))
                    current_accuracy = metrics.get('train_accuracy_step', metrics.get('train_accuracy', None))
                    
                    # Convert tensors to float
                    if current_loss is not None and hasattr(current_loss, 'item'):
                        current_loss = float(current_loss.item())
                    if current_accuracy is not None and hasattr(current_accuracy, 'item'):
                        current_accuracy = float(current_accuracy.item())
            except (RuntimeError, AttributeError):
                # Trainer not attached or no metrics available
                pass
            
            # Build status with enhanced information
            status = {
                'device_id': self.config.device_id,
                'server_id': 'local_server',  # Add server identification
                'config': {
                    'learning_rate': current_lr,
                    'batch_size': self.config.batch_size,
                    'dropout_rate': self.config.dropout_rate,
                    'num_epochs': self.config.num_epochs,
                    'input_dim': self.config.input_dim,
                    'hidden_dim': self.config.hidden_dim,
                    'num_classes': self.config.num_classes,
                    'optimizer_type': 'Adam'
                },
                'system': system_metrics,
                'learning_rate': current_lr,
                'training_time': time.time() - self.training_start_time,
                'best_accuracy': self.best_accuracy,
                'model_parameters': sum(p.numel() for p in self.model.parameters()),
                'step_progress': 0.0,  # Will be updated during training
                'type': 'training_metrics'  # For UI message routing
            }
            
            # Add current metrics if available
            if current_loss is not None:
                status['loss'] = current_loss
            if current_accuracy is not None:
                status['accuracy'] = current_accuracy
            
            # Add epoch information if available
            try:
                trainer_ref = getattr(self, '_trainer', None) or getattr(self, 'trainer', None)
                if trainer_ref:
                    status['epoch'] = getattr(trainer_ref, 'current_epoch', 0)
                    status['step'] = getattr(trainer_ref, 'global_step', 0)
                    
                    # Calculate step progress within epoch
                    if hasattr(trainer_ref, 'num_training_batches') and trainer_ref.num_training_batches > 0:
                        batch_idx = getattr(trainer_ref, 'batch_idx', 0)
                        status['step_progress'] = (batch_idx / trainer_ref.num_training_batches) * 100.0
            except (RuntimeError, AttributeError):
                # Trainer not attached, use default values
                status['epoch'] = 0
                status['step'] = 0
            
            return status
            
        except Exception as e:
            logger.warning(f"Error collecting additional status: {e}")
            return {
                'device_id': self.config.device_id,
                'server_id': 'local_server',
                'type': 'training_metrics',
                'error': str(e)
            }
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        super().on_train_epoch_end()
        
        # Update best accuracy tracking
        try:
            trainer_ref = getattr(self, '_trainer', None) or getattr(self, 'trainer', None)
            if trainer_ref and hasattr(trainer_ref, 'logged_metrics'):
                metrics = trainer_ref.logged_metrics
                accuracy_key = None
                for key in ['train_accuracy_epoch', 'train_accuracy', 'accuracy']:
                    if key in metrics:
                        accuracy_key = key
                        break
                
                if accuracy_key:
                    current_accuracy = metrics[accuracy_key]
                    if hasattr(current_accuracy, 'item'):
                        current_accuracy = float(current_accuracy.item())
                    else:
                        current_accuracy = float(current_accuracy)
                    
                    if current_accuracy > self.best_accuracy:
                        self.best_accuracy = current_accuracy
                        logger.info(f"New best accuracy: {self.best_accuracy:.4f}")
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"Could not update best accuracy: {e}")
            pass


class UIStatusCallback(Callback):
    """Lightning callback for publishing training status to torchft-ui."""
    
    def __init__(self, config: LightningTrainingConfig):
        super().__init__()
        self.config = config
        self.start_time = time.time()
        self.epoch_start_time = time.time()
    
    def _create_base_status(self, status_type: str, trainer: L.Trainer, pl_module: 'TorchLoomLightningModule') -> Dict[str, Any]:
        """Create a base status dictionary with common fields."""
        return {
            'type': status_type,
            'device_id': self.config.device_id,
            'replica_id': self.config.replica_id,
            'server_id': 'local_server',
            'timestamp': time.time(),
            'epoch': getattr(trainer, 'current_epoch', 0),
            'step': getattr(trainer, 'global_step', 0),
            'total_epochs': self.config.num_epochs,
            'training_time': time.time() - self.start_time,
            'best_accuracy': getattr(pl_module, 'best_accuracy', 0.0),
        }
    
    def on_train_start(self, trainer: L.Trainer, pl_module: 'TorchLoomLightningModule'):
        """Called when training starts."""
        logger.info("Training started - UI status callback active")
        
        # Publish training start status
        if hasattr(pl_module, 'weavelet'):
            status = self._create_base_status('training_start', trainer, pl_module)
            status.update({
                'status': 'starting',
                'progress': 0.0,
                'message': 'Training initialization complete'
            })
            
            # Add configuration details
            status['config'] = {
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'dropout_rate': self.config.dropout_rate,
                'num_epochs': self.config.num_epochs,
                'input_dim': self.config.input_dim,
                'hidden_dim': self.config.hidden_dim,
                'num_classes': self.config.num_classes,
                'optimizer_type': 'Adam'
            }
            
            pl_module.weavelet.publish_training_status(status)
            logger.info(f"Published training start status: {status['type']}")
    
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: 'TorchLoomLightningModule'):
        """Called at the start of each training epoch."""
        self.epoch_start_time = time.time()
        
        # Publish epoch start status via weavelet
        if hasattr(pl_module, 'weavelet'):
            status = self._create_base_status('epoch_start', trainer, pl_module)
            status.update({
                'status': 'training',
                'step_progress': 0.0,
                'progress': (trainer.current_epoch / self.config.num_epochs) * 100.0,
                'message': f'Starting epoch {trainer.current_epoch + 1}/{self.config.num_epochs}'
            })
            
            pl_module.weavelet.publish_training_status(status)
            logger.debug(f"Published epoch start status for epoch {trainer.current_epoch}")
    
    def on_train_batch_end(self, trainer: L.Trainer, pl_module: 'TorchLoomLightningModule', outputs, batch, batch_idx):
        """Called at the end of each training batch."""
        # Publish batch-level updates every few batches to avoid overwhelming the UI
        if batch_idx % 5 == 0 and hasattr(pl_module, 'weavelet'):
            try:
                # Get current metrics
                current_loss = None
                current_accuracy = None
                
                if outputs and isinstance(outputs, dict):
                    if 'loss' in outputs:
                        current_loss = float(outputs['loss'].item()) if hasattr(outputs['loss'], 'item') else float(outputs['loss'])
                    if 'accuracy' in outputs:
                        current_accuracy = float(outputs['accuracy'])
                
                # Calculate batch progress within epoch
                step_progress = 0.0
                if trainer.num_training_batches > 0:
                    step_progress = (batch_idx / trainer.num_training_batches) * 100.0
                
                # Calculate overall progress
                epoch_progress = (trainer.current_epoch / self.config.num_epochs) * 100.0
                batch_progress_in_epoch = step_progress / 100.0 / self.config.num_epochs * 100.0
                overall_progress = epoch_progress + batch_progress_in_epoch
                
                status = self._create_base_status('batch_update', trainer, pl_module)
                status.update({
                    'status': 'training',
                    'batch_idx': batch_idx,
                    'step_progress': step_progress,
                    'progress': overall_progress,
                    'message': f'Epoch {trainer.current_epoch + 1}/{self.config.num_epochs}, Batch {batch_idx + 1}/{trainer.num_training_batches}'
                })
                
                if current_loss is not None:
                    status['loss'] = current_loss
                if current_accuracy is not None:
                    status['accuracy'] = current_accuracy
                
                # Add system metrics if available
                try:
                    additional_status = pl_module._collect_additional_status()
                    if additional_status and 'system' in additional_status:
                        status['system'] = additional_status['system']
                    if additional_status and 'learning_rate' in additional_status:
                        status['learning_rate'] = additional_status['learning_rate']
                except Exception as e:
                    logger.debug(f"Could not collect additional status: {e}")
                
                pl_module.weavelet.publish_training_status(status)
                
            except Exception as e:
                logger.debug(f"Error publishing batch update: {e}")
    
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: 'TorchLoomLightningModule'):
        """Called at the end of each training epoch."""
        # Publish epoch completion status
        if hasattr(pl_module, 'weavelet'):
            try:
                metrics = trainer.logged_metrics
                
                # Extract epoch-level metrics
                epoch_loss = None
                epoch_accuracy = None
                
                # Try different metric name patterns
                for loss_key in ['train_loss_epoch', 'train_loss', 'loss']:
                    if loss_key in metrics:
                        epoch_loss = float(metrics[loss_key].item()) if hasattr(metrics[loss_key], 'item') else float(metrics[loss_key])
                        break
                
                for acc_key in ['train_accuracy_epoch', 'train_accuracy', 'accuracy']:
                    if acc_key in metrics:
                        epoch_accuracy = float(metrics[acc_key].item()) if hasattr(metrics[acc_key], 'item') else float(metrics[acc_key])
                        break
                
                epoch_time = time.time() - self.epoch_start_time
                progress = ((trainer.current_epoch + 1) / self.config.num_epochs) * 100.0
                
                status = self._create_base_status('epoch_complete', trainer, pl_module)
                status.update({
                    'status': 'training',
                    'step_progress': 100.0,  # Epoch is complete
                    'progress': progress,
                    'epoch_time': epoch_time,
                    'message': f'Completed epoch {trainer.current_epoch + 1}/{self.config.num_epochs}'
                })
                
                if epoch_loss is not None:
                    status['loss'] = epoch_loss
                if epoch_accuracy is not None:
                    status['accuracy'] = epoch_accuracy
                
                # Add comprehensive metrics
                status['metrics'] = {
                    'epoch_loss': epoch_loss,
                    'epoch_accuracy': epoch_accuracy,
                    'best_accuracy': pl_module.best_accuracy,
                    'epoch_time': epoch_time,
                    'learning_rate': getattr(pl_module.config, 'learning_rate', 0.001)
                }
                
                # Add all available metrics for debugging
                status['all_metrics'] = {k: float(v.item()) if hasattr(v, 'item') else float(v) 
                                       for k, v in metrics.items() 
                                       if isinstance(v, (int, float, torch.Tensor))}
                
                # Add system information
                try:
                    additional_status = pl_module._collect_additional_status()
                    if additional_status and 'system' in additional_status:
                        status['system'] = additional_status['system']
                except Exception as e:
                    logger.debug(f"Could not collect system status: {e}")
                
                pl_module.weavelet.publish_training_status(status)
                
                logger.info(f"Epoch {trainer.current_epoch + 1} completed: "
                           f"Loss={epoch_loss:.4f if epoch_loss else 'N/A'}, "
                           f"Acc={epoch_accuracy:.4f if epoch_accuracy else 'N/A'}, "
                           f"Best={pl_module.best_accuracy:.4f}, "
                           f"Time={epoch_time:.2f}s, Progress={progress:.1f}%")
                
            except Exception as e:
                logger.warning(f"Error publishing epoch completion status: {e}")
    
    def on_train_end(self, trainer: L.Trainer, pl_module: 'TorchLoomLightningModule'):
        """Called when training ends."""
        if hasattr(pl_module, 'weavelet'):
            try:
                final_metrics = trainer.logged_metrics
                total_time = time.time() - self.start_time
                
                # Extract final metrics
                final_loss = None
                final_accuracy = None
                
                for loss_key in ['train_loss_epoch', 'train_loss', 'loss']:
                    if loss_key in final_metrics:
                        final_loss = float(final_metrics[loss_key].item()) if hasattr(final_metrics[loss_key], 'item') else float(final_metrics[loss_key])
                        break
                
                for acc_key in ['train_accuracy_epoch', 'train_accuracy', 'accuracy']:
                    if acc_key in final_metrics:
                        final_accuracy = float(final_metrics[acc_key].item()) if hasattr(final_metrics[acc_key], 'item') else float(final_metrics[acc_key])
                        break
                
                status = self._create_base_status('training_complete', trainer, pl_module)
                status.update({
                    'status': 'completed',
                    'progress': 100.0,
                    'step_progress': 100.0,
                    'final_epoch': trainer.current_epoch,
                    'final_step': trainer.global_step,
                    'message': f'Training completed successfully in {total_time:.2f}s'
                })
                
                # Add comprehensive final metrics
                status['final_metrics'] = {
                    'final_loss': final_loss,
                    'final_accuracy': final_accuracy,
                    'best_accuracy': pl_module.best_accuracy,
                    'total_time': total_time,
                    'epochs_completed': trainer.current_epoch + 1,
                    'steps_completed': trainer.global_step
                }
                
                # Add all logged metrics
                if final_metrics:
                    status['all_final_metrics'] = {k: float(v.item()) if hasattr(v, 'item') else float(v) 
                                                 for k, v in final_metrics.items() 
                                                 if isinstance(v, (int, float, torch.Tensor))}
                
                # Add final system status
                try:
                    additional_status = pl_module._collect_additional_status()
                    if additional_status:
                        if 'system' in additional_status:
                            status['system'] = additional_status['system']
                        if 'config' in additional_status:
                            status['config'] = additional_status['config']
                except Exception as e:
                    logger.debug(f"Could not collect final additional status: {e}")
                
                # Publish the completion status multiple times to ensure UI receives it
                for i in range(3):
                    pl_module.weavelet.publish_training_status(status)
                    if i < 2:  # Don't sleep after the last publish
                        time.sleep(0.1)
                
                logger.info(f"Training completed - Total time: {total_time:.2f}s, "
                           f"Best accuracy: {pl_module.best_accuracy:.4f}, "
                           f"Final loss: {final_loss:.4f if final_loss else 'N/A'}, "
                           f"Final accuracy: {final_accuracy:.4f if final_accuracy else 'N/A'}")
                
                # Also publish a final status update after a short delay
                time.sleep(0.5)
                status['message'] = 'Training session ended - ready for new training'
                pl_module.weavelet.publish_training_status(status)
                
            except Exception as e:
                logger.warning(f"Error publishing training completion status: {e}")
                
                # Publish a basic completion status even if there's an error
                try:
                    basic_status = self._create_base_status('training_complete', trainer, pl_module)
                    basic_status.update({
                        'status': 'completed',
                        'progress': 100.0,
                        'message': 'Training completed with errors in status collection',
                        'error': str(e)
                    })
                    pl_module.weavelet.publish_training_status(basic_status)
                except Exception as inner_e:
                    logger.error(f"Failed to publish basic completion status: {inner_e}")


class TorchLoomDataModule(L.LightningDataModule):
    """Lightning data module for the demo dataset."""
    
    def __init__(self, config: LightningTrainingConfig):
        super().__init__()
        self.config = config
        self.dataset: Optional[SimpleClassificationDataset] = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training."""
        if stage == "fit" or stage is None:
            self.dataset = SimpleClassificationDataset(
                self.config.num_samples,
                self.config.input_dim,
                self.config.num_classes
            )
    
    def train_dataloader(self):
        """Return training dataloader."""
        if self.dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Keep simple for demo
        )


def main():
    """Main function to run the Lightning demo training."""
    # Configuration
    config = LightningTrainingConfig()
    
    # Create Lightning components
    data_module = TorchLoomDataModule(config)
    model = TorchLoomLightningModule(config)
    
    # Create callbacks
    ui_callback = UIStatusCallback(config)
    
    # Configure trainer
    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        callbacks=[ui_callback],
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=5,  # Log frequently for UI updates
        check_val_every_n_epoch=1,
        default_root_dir="./lightning_logs"
    )
    
    try:
        logger.info("Starting Lightning training with weavelet integration...")
        trainer.fit(model, data_module)
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Ensure weavelet is properly stopped
        logger.info("Entering finally block to stop weavelet...")
        if hasattr(model, 'weavelet'):
            logger.info("Weavelet attribute found on model, attempting to stop...")
            model.weavelet.stop()
            logger.info("Weavelet stop method called.")
        else:
            logger.warning("Weavelet attribute not found on model. Could not stop weavelet.")

        logger.info("Exiting finally block.")

if __name__ == "__main__":
    import os
    
    # Create necessary directories
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./lightning_logs", exist_ok=True)
    
    main() 