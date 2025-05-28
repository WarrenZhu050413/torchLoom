import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import time
import uuid
import logging
import threading

# torchLoom imports for threadlet integration
from torchLoom.threadlet.threadlet import Threadlet

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Define the LightningModule
class LocalSGDLightning(pl.LightningModule):
    def __init__(self, sync_every=5, learning_rate=1e-3, process_id=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = Net()
        self.criterion = nn.CrossEntropyLoss()
        self.sync_every = sync_every
        self.learning_rate = learning_rate
        
        # Training tracking for threadlet
        self.training_start_time = None
        self.max_epochs = 10  # Will be updated from trainer
        self.max_steps = 1000  # Estimated max steps
        
        # Threadlet integration
        self.process_id = process_id or f"localsgd-lightning-{uuid.uuid4().hex[:8]}"
        self.threadlet = None
        self._setup_threadlet()

    def _setup_threadlet(self):
        """Setup threadlet for configuration management and status reporting."""
        try:
            # Create threadlet instance
            self.threadlet = Threadlet(
                process_id=self.process_id,
                device_uuid=f"localsgd-device-{uuid.uuid4().hex[:8]}",
            )
            
            # Register handler for sync_every parameter changes
            self.threadlet.register_handler(
                "sync_every", 
                self._update_sync_every
            )
            
            # Start threadlet
            self.threadlet.start()
            
            print(f"Threadlet initialized for process: {self.process_id}")
            
        except Exception as e:
            print(f"Warning: Failed to initialize threadlet: {e}")
            self.threadlet = None

    def _update_sync_every(self, new_sync_every):
        """Handler for sync_every parameter updates from weaver."""
        try:
            old_sync_every = self.sync_every
            new_sync_every_value = int(new_sync_every)
            self.sync_every = new_sync_every_value
            print(f"🔄 sync_every updated: {old_sync_every} -> {self.sync_every}")
            
            # Update hyperparameters for logging
            self.hparams.sync_every = self.sync_every
            
        except (ValueError, TypeError) as e:
            print(f"❌ Failed to update sync_every: invalid value '{new_sync_every}': {e}")

    def _publish_training_status(self, batch_idx=None):
        """Publish training status to threadlet/weaver."""
        if not self.threadlet:
            return
            
        try:
            # Calculate training time
            current_time = time.time()
            training_time = current_time - self.training_start_time if self.training_start_time else 0.0
            
            # Get current metrics from trainer logs
            metrics = {}
            if hasattr(self.trainer, 'logged_metrics'):
                for key, value in self.trainer.logged_metrics.items():
                    if isinstance(value, torch.Tensor):
                        metrics[key] = str(value.item())
                    else:
                        metrics[key] = str(value)
            
            # Add current configuration
            config = {
                "sync_every": str(self.sync_every),
                "learning_rate": str(self.learning_rate),
                "batch_size": str(self.trainer.datamodule.batch_size) if hasattr(self.trainer, 'datamodule') else "unknown",
                "max_epochs": str(self.trainer.max_epochs),
                "strategy": str(self.trainer.strategy.__class__.__name__) if self.trainer.strategy else "unknown",
            }
            
            # Prepare status data according to TrainingStatus proto
            status_data = {
                "current_step": self.global_step,
                "epoch": self.current_epoch,
                "metrics": metrics,
                "training_time": training_time,
                "max_step": self.max_steps,
                "max_epoch": self.trainer.max_epochs if self.trainer else self.max_epochs,
                "config": config,
            }
            
            # Publish to threadlet
            self.threadlet.publish_training_status(status_data=status_data)
            
        except Exception as e:
            print(f"Warning: Failed to publish training status: {e}")

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        """Called when training starts."""
        self.training_start_time = time.time()
        
        # Update max_steps estimate based on dataloader
        if hasattr(self.trainer, 'num_training_batches'):
            self.max_steps = self.trainer.num_training_batches * self.trainer.max_epochs
            
        # Publish initial training status
        self._publish_training_status()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss)
        
        # Publish training status every 10 steps
        if (self.global_step + 1) % 10 == 0:
            self._publish_training_status(batch_idx)
        
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (self.global_step + 1) % self.sync_every == 0:
            if self.trainer.world_size > 1:
                # Synchronize parameters using all_reduce
                if self.trainer.is_global_zero:
                    print(f"Attempting to sync parameters via all_reduce at global_step {self.global_step + 1} (world_size: {self.trainer.world_size}, sync_every: {self.sync_every})")
                for param in self.model.parameters():
                    if param.requires_grad:
                        torch.distributed.all_reduce(param.data, op=torch.distributed.ReduceOp.AVG)
                if self.trainer.is_global_zero:
                    print(f"Synced parameters via all_reduce at global_step {self.global_step + 1}")
            elif self.trainer.is_global_zero: # Single GPU/CPU case
                print(f"LocalSGD: Sync step {self.global_step + 1} (single process, no all_reduce needed, sync_every: {self.sync_every}).")

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Publish training status after validation
        self._publish_training_status()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels).item() / len(labels)
        self.log("val_acc", acc)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("test_loss", loss)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels).item() / len(labels)
        self.log("test_acc", acc)
        return loss

    def on_train_end(self):
        """Called when training ends."""
        # Publish final training status
        self._publish_training_status()
        
        # Stop threadlet
        if self.threadlet:
            try:
                self.threadlet.stop()
                print(f"Threadlet stopped for process: {self.process_id}")
            except Exception as e:
                print(f"Warning: Error stopping threadlet: {e}")

# Define the DataModule
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./cifar", batch_size: int = 64, num_workers: int = 2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def prepare_data(self):
        # download
        torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar_full = torchvision.datasets.CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])
        if stage == "test" or stage is None:
            self.cifar_test = torchvision.datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        sampler = None
        shuffle = True
        
        # Use DistributedSampler if we're in a distributed setting
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(
                self.cifar_train, 
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=True
            )
            shuffle = False  # DistributedSampler handles shuffling
            
        return DataLoader(
            self.cifar_train, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=shuffle,
            sampler=sampler,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        sampler = None
        
        # Use DistributedSampler for validation as well (without shuffling)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(
                self.cifar_val, 
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=False
            )
            
        return DataLoader(
            self.cifar_val, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            sampler=sampler,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        sampler = None
        
        # Use DistributedSampler for test as well (without shuffling)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(
                self.cifar_test, 
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=False
            )
            
        return DataLoader(
            self.cifar_test, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            sampler=sampler,
            persistent_workers=True if self.num_workers > 0 else False
        )

if __name__ == "__main__":
    QUICK_RUN = bool(os.environ.get("QUICK_RUN", False))
    
    # Create a unique process ID for this training run
    process_id = f"localsgd-{uuid.uuid4().hex[:8]}"
    print(f"\n=== LocalSGD Lightning Training with Threadlet Integration ===")
    print(f"Process ID: {process_id}")
    print(f"Quick Run Mode: {QUICK_RUN}")
    print("\nThreadlet Integration Features:")
    print("• Handles 'sync_every' parameter updates from weaver commands")
    print("• Publishes training status including metrics, config, and progress")
    print("• Compatible with distributed training strategies")
    print("\nTo test threadlet integration:")
    print("1. Start the weaver: python -m torchLoom.weaver.core")
    print("2. Run this script")
    print("3. Send config updates via weaver UI or commands\n")

    dm = CIFAR10DataModule(batch_size=1024)
    model = LocalSGDLightning(
        sync_every=5 if QUICK_RUN else 100, 
        learning_rate=1e-3,
        process_id=process_id
    )

    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath="localsgd_lightning_checkpoints", # Save checkpoints in a dedicated directory
        filename="localsgd-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=1 if QUICK_RUN else 10,
        accelerator="auto",
        devices="auto", # Uses all available GPUs or CPU
        strategy="ddp_find_unused_parameters_true" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "auto",
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger(".", name="localsgd_lightning"), # Save logs in the current directory
        fast_dev_run=True if QUICK_RUN else False
    )

    try:
        print("Starting training...")
        trainer.fit(model, dm)
        print("Training finished.")

        if not QUICK_RUN:
            print("Starting testing...")
            trainer.test(model, dm)
            print("Testing finished.")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        # Ensure threadlet is properly stopped
        if hasattr(model, 'threadlet') and model.threadlet:
            try:
                model.threadlet.stop()
                print("Threadlet stopped successfully")
            except Exception as e:
                print(f"Warning: Error stopping threadlet: {e}")