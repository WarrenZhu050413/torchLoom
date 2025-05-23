import multiprocessing as mp
import os
import time
from typing import Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.demos import Transformer
from torch.utils.data import DataLoader, Dataset

from torchLoom.weavelet import WeaveletProcess


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
    def __init__(self, vocab_size, replica_id: Optional[str] = None):
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size)
        self.optimizer_type = "SGD"
        self.lr = 0.1
        self.optimizer = None

        # Initialize process-based weavelet for weaver communication
        self.weavelet = WeaveletProcess(replica_id=replica_id)

        # Start weavelet process
        self.weavelet.start()

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        time.sleep(1)

        # Check for config updates from weavelet process
        config_update = self.weavelet.get_config_update()
        if config_update:
            self.handle_config_update(config_update)

        # Publish training status to weaver via weavelet process
        self.weavelet.publish_training_status(
            {
                "batch_idx": batch_idx,
                "loss": float(loss),
                "optimizer_type": self.optimizer_type,
            }
        )

        return loss

    def handle_config_update(self, config_params: dict) -> None:
        """Handle configuration updates from the weavelet process."""
        for key, value in config_params.items():
            if key == "optimizer_type":
                self.update_optimizer(value)
            # Add other config parameter handlers here as needed

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

    def on_train_end(self):
        """Clean up weavelet when training ends."""
        if hasattr(self, "weavelet"):
            try:
                # Stop weavelet process
                self.weavelet.stop()
            except Exception as e:
                print(f"Warning: Error stopping weavelet: {e}")


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
            config_update = pl_module.weavelet.get_config_update()
            if config_update:
                pl_module.handle_config_update(config_update)

    def on_train_end(self, trainer, pl_module):
        if hasattr(pl_module, "weavelet"):
            print("Training ended, stopping weavelet process")
            try:
                pl_module.weavelet.stop()
            except Exception as e:
                print(f"Warning: Error stopping weavelet: {e}")


if __name__ == "__main__":
    # Set multiprocessing start method for better compatibility
    try:
        if os.name == "posix":
            mp.set_start_method('spawn', force=True)
        else: 
            mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass  # Start method may already be set
    
    dataset = RandomTextDataset(vocab_size=1000)
    dataloader = DataLoader(dataset, batch_size=32)

    model = LightningTransformer(
        vocab_size=dataset.vocab_size, replica_id="lightning_trainer_1"
    )

    callback = WeaveletCallback()
    trainer = L.Trainer(fast_dev_run=100, callbacks=[callback])
    trainer.fit(model=model, train_dataloaders=dataloader)
