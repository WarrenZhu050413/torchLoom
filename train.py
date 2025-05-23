import multiprocessing as mp
import time
from typing import Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.demos import Transformer
from torch.utils.data import DataLoader, Dataset

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
    def __init__(self, vocab_size, replica_id: Optional[str] = None):
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size)
        self.optimizer_type = "SGD"
        self.lr = 0.1
        self.optimizer = None

        # Initialize weavelet for weaver communication
        self.weavelet = Weavelet(replica_id=replica_id)
        self.weavelet.register_config_handler("optimizer_type", self.update_optimizer)

        # Start weavelet in background
        self.weavelet.start()

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        time.sleep(1)

        # Publish training status to weaver
        self.weavelet.publish_training_status(
            {
                "batch_idx": batch_idx,
                "loss": float(loss),
                "optimizer_type": self.optimizer_type,
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
        """Update optimizer type - called by weavelet when config changes."""
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
            self.weavelet.stop()


dataset = RandomTextDataset(vocab_size=1000)
dataloader = DataLoader(dataset, batch_size=32)


class WeaveletCallback(Callback):
    """Simplified callback that just ensures weavelet is running."""

    def on_train_start(self, trainer, pl_module):
        if hasattr(pl_module, "weavelet"):
            print(
                f"Training started with weavelet for replica: {pl_module.weavelet._replica_id}"
            )

    def on_train_end(self, trainer, pl_module):
        if hasattr(pl_module, "weavelet"):
            print("Training ended, stopping weavelet")
            pl_module.weavelet.stop()


model = LightningTransformer(
    vocab_size=dataset.vocab_size, replica_id="lightning_trainer_1"
)

if __name__ == "__main__":
    callback = WeaveletCallback()
    trainer = L.Trainer(fast_dev_run=100, callbacks=[callback])
    trainer.fit(model=model, train_dataloaders=dataloader)
