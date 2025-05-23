import multiprocessing as mp
import os
import time
from typing import Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.demos import Transformer
from torch.utils.data import DataLoader, Dataset

from torchLoom.lightning_integration import (
    EnhancedWeaveletLightningModule,
    weavelet_handler,
)
from torchLoom.weavelet import Weavelet


class RandomTextDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=32, vocab_size=1000):
        self.inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.targets = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class LightningTransformer(L.LightningModule):
    """Original Lightning implementation with enhanced decorator-based handlers."""

    def __init__(self, vocab_size, replica_id: Optional[str] = None):
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size)
        self.optimizer_type = "SGD"
        self.lr = 0.1
        self.optimizer = None

        # Initialize process-based weavelet for weaver communication
        self.weavelet = Weavelet(replica_id=replica_id)

        # Register handlers using decorators (new enhanced API)
        @self.weavelet.handler("optimizer_type")
        def update_optimizer(new_type: str):
            self.update_optimizer(new_type)

        @self.weavelet.handler("learning_rate")
        def update_learning_rate(new_lr: float):
            self.update_learning_rate(new_lr)

        # Start weavelet process
        self.weavelet.start()

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        time.sleep(1)

        # Automatic config checking (handlers called automatically)
        self.weavelet.get_config_update()

        # Publish training status to weaver via weavelet process
        self.weavelet.publish_training_status(
            {
                "batch_idx": batch_idx,
                "loss": float(loss),
                "optimizer_type": self.optimizer_type,
                "learning_rate": self.lr,
            }
        )

        return loss

    def _create_optimizer(self):
        if self.optimizer_type == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def configure_optimizers(self):
        self.optimizer = self._create_optimizer()
        return self.optimizer

    def update_optimizer(self, optimizer_type: str) -> None:
        """Update optimizer type - called when config changes are received."""
        if optimizer_type == self.optimizer_type:
            return

        print(f"Updating optimizer from {self.optimizer_type} to {optimizer_type}")
        self.optimizer_type = optimizer_type
        new_opt = self._create_optimizer()

        if self.trainer is not None:
            self.trainer.optimizers = [new_opt]
        self.optimizer = new_opt

    def update_learning_rate(self, learning_rate: float) -> None:
        """Update learning rate - called when config changes are received."""
        if learning_rate == self.lr:
            return

        print(f"Updating learning rate from {self.lr} to {learning_rate}")
        self.lr = learning_rate

        # Update optimizer learning rate
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate

    def on_train_end(self):
        """Clean up weavelet when training ends."""
        if hasattr(self, "weavelet"):
            try:
                # Stop weavelet process
                self.weavelet.stop()
            except Exception as e:
                print(f"Warning: Error stopping weavelet: {e}")


class EnhancedLightningTransformer(EnhancedWeaveletLightningModule):
    """New enhanced Lightning implementation using automatic integration."""

    def __init__(self, vocab_size: int, replica_id: str):
        super().__init__(replica_id=replica_id)
        self.model = Transformer(vocab_size=vocab_size)
        self.optimizer_type = "SGD"
        self.lr = 0.1
        self.optimizer = None

    def _user_training_step(self, batch, batch_idx):
        """Implementation of training logic (replaces training_step)."""
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        time.sleep(1)
        return loss

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def configure_optimizers(self):
        self.optimizer = self._create_optimizer()
        return self.optimizer

    def _create_optimizer(self):
        if self.optimizer_type == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def _collect_additional_status(self):
        """Add custom status information."""
        return {
            "optimizer_type": self.optimizer_type,
            "learning_rate": self.lr,
        }

    # Handlers using the global decorator (automatically registered)
    @weavelet_handler("optimizer_type")
    def update_optimizer(self, new_type: str):
        """Update optimizer type - called automatically when config changes."""
        if new_type == self.optimizer_type:
            return

        print(f"[Enhanced] Updating optimizer from {self.optimizer_type} to {new_type}")
        self.optimizer_type = new_type
        new_opt = self._create_optimizer()

        if self.trainer is not None:
            self.trainer.optimizers = [new_opt]
        self.optimizer = new_opt

    @weavelet_handler("learning_rate")
    def update_learning_rate(self, new_lr: float):
        """Update learning rate - called automatically when config changes."""
        if new_lr == self.lr:
            return

        print(f"[Enhanced] Updating learning rate from {self.lr} to {new_lr}")
        self.lr = new_lr

        # Update optimizer learning rate
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr


class WeaveletCallback(Callback):
    """Callback to monitor weavelet process lifecycle."""

    def on_train_start(self, trainer, pl_module):
        if hasattr(pl_module, "weavelet"):
            print(
                f"Training started with weavelet process for replica: {pl_module.weavelet._replica_id}"
            )

    def on_train_epoch_start(self, trainer, pl_module):
        """Check for config updates at the start of each epoch."""
        if hasattr(pl_module, "weavelet"):
            # For original implementation, manually check for updates
            if hasattr(pl_module, "get_config_update"):
                pl_module.weavelet.get_config_update()

    def on_train_end(self, trainer, pl_module):
        if hasattr(pl_module, "weavelet"):
            print("Training ended, stopping weavelet process")
            try:
                pl_module.weavelet.stop()
            except Exception as e:
                print(f"Warning: Error stopping weavelet: {e}")


def demo_original_integration():
    """Demonstrate the original integration with enhanced handlers."""
    print("🔥 Demo: Original Integration with Enhanced Handlers")
    print("=" * 60)

    dataset = RandomTextDataset(vocab_size=1000)
    dataloader = DataLoader(dataset, batch_size=32)

    model = LightningTransformer(
        vocab_size=dataset.vocab_size, replica_id="lightning_trainer_1"
    )

    callback = WeaveletCallback()
    trainer = L.Trainer(fast_dev_run=5, callbacks=[callback])
    trainer.fit(model=model, train_dataloaders=dataloader)


def demo_enhanced_integration():
    """Demonstrate the new enhanced integration."""
    print("\n🚀 Demo: Enhanced Automatic Integration")
    print("=" * 60)

    dataset = RandomTextDataset(vocab_size=1000)
    dataloader = DataLoader(dataset, batch_size=32)

    model = EnhancedLightningTransformer(
        vocab_size=dataset.vocab_size, replica_id="enhanced_trainer_1"
    )

    # No callback needed - everything is automatic!
    trainer = L.Trainer(fast_dev_run=5, enable_progress_bar=False)
    trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == "__main__":
    # Set multiprocessing start method for better compatibility
    try:
        if os.name == "posix":
            mp.set_start_method("spawn", force=True)
        else:
            mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass  # Start method may already be set

    # Demo both integration approaches
    demo_original_integration()
    demo_enhanced_integration()

    print("\n✅ All demos completed!")
    print("\nKey improvements demonstrated:")
    print("  ✓ Decorator-based handler registration")
    print("  ✓ Automatic handler dispatch")
    print("  ✓ Type validation for config parameters")
    print("  ✓ Enhanced Lightning integration classes")
    print("  ✓ Automatic config checking and status publishing")
