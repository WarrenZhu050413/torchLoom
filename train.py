import lightning as L
import torch

from lightning.pytorch.demos import Transformer
from torch.utils.data import DataLoader, Dataset

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

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)
    
dataset = RandomTextDataset(vocab_size=1000)
dataloader = DataLoader(dataset, batch_size=32)
model = LightningTransformer(vocab_size=dataset.vocab_size)

trainer = L.Trainer(fast_dev_run=100)
trainer.fit(model=model, train_dataloaders=dataloader)