import lightning as L
import torch
import time
import multiprocessing as mp
from lightning.pytorch.demos import Transformer
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, Dataset

from torchLoom.weavelet import weavelet_process

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
    def __init__(self, vocab_size):
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size)
        self.optimizer_type = "SGD"
        self.lr = 0.1
        self.optimizer = None

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        time.sleep(1)
        return loss

    def _create_optimizer(self):
        if self.optimizer_type == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def configure_optimizers(self):
        self.optimizer = self._create_optimizer()
        return self.optimizer

    def update_optimizer(self, optimizer_type: str) -> None:
        if optimizer_type == self.optimizer_type:
            return
        self.optimizer_type = optimizer_type
        new_opt = self._create_optimizer()
        if self.trainer is not None:
            self.trainer.optimizers = [new_opt]
        self.optimizer = new_opt
    
dataset = RandomTextDataset(vocab_size=1000)
dataloader = DataLoader(dataset, batch_size=32)
class WeaveletCallback(Callback):
    def __init__(self, queue: mp.Queue, process: mp.Process):
        self.queue = queue
        self.process = process

    def on_train_epoch_start(self, trainer, pl_module):
        while True:
            try:
                opt_type = self.queue.get_nowait()
            except Exception:
                break
            pl_module.update_optimizer(opt_type)
        if not self.process.is_alive():
            raise RuntimeError("Weavelet process terminated")


model = LightningTransformer(vocab_size=dataset.vocab_size)

if __name__ == "__main__":
    q = mp.Queue()
    proc = mp.Process(target=weavelet_process, args=(q,))
    proc.start()

    callback = WeaveletCallback(q, proc)
    trainer = L.Trainer(fast_dev_run=100, callbacks=[callback])
    trainer.fit(model=model, train_dataloaders=dataloader)

    proc.terminate()
    proc.join()
