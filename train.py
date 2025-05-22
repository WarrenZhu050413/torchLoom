import logging
import os
# from datetime import timedelta # No longer needed for ProcessGroup timeouts

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision # torchvision.datasets and torchvision.transforms are imported directly
from torchvision import datasets
import torchvision.transforms as transforms
# import torch.nn as nn # Already imported above
import torch.optim as optim # For the new optimizer
from torch.utils.data import DataLoader # Added for MNISTDataModule
# from torch.distributed.elastic.multiprocessing.errors import record # No longer using torch-elastic
# from torchdata.stateful_dataloader import StatefulDataLoader # No longer used
import pytorch_lightning as pl # For LightningModule
from pytorch_lightning import Trainer # Explicitly import Trainer

# from torchft import ( # All torchft components removed
#     DistributedDataParallel,
#     DistributedSampler,
#     Manager,
#     Optimizer,
#     ProcessGroupGloo,
#     ProcessGroupNCCL,
# )
# from torchft.checkpointing.pg_transport import PGTransport

logging.basicConfig(level=logging.INFO)

# @record # No longer using torch-elastic
def main() -> None:
    # Old torchft and CIFAR-10 setup is removed.
    # REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
    # NUM_REPLICA_GROUPS = int(os.environ.get("NUM_REPLICA_GROUPS", 2))

    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )
    # trainset = torchvision.datasets.CIFAR10(
    #     root="./cifar", train=True, download=True, transform=transform
    # )
    # sampler = DistributedSampler(...)
    # trainloader = StatefulDataLoader(...)
    # def load_state_dict(state_dict): ...
    # def state_dict(): ...
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # pg = ProcessGroupNCCL(...) / ProcessGroupGloo(...)
    # transport = PGTransport(...)
    # manager = Manager(...)

    # Instantiate DataModule
    mnist_data_module = MNISTDataModule(batch_size=64)

    # Instantiate LightningModule
    mnist_model = MNISTLightningModule(lr=1.0) # lr from original mnist example

    # Instantiate Trainer
    # Using 'auto' for accelerator and devices for broader compatibility.
    # Limiting epochs for quicker runs during testing.
    trainer = Trainer(
        max_epochs=10,
        accelerator='auto',
        devices='auto',
        logger=True # Default TensorBoard logger
    )

    # Train the model
    trainer.fit(model=mnist_model, datamodule=mnist_data_module)

    # Test the model (optional)
    trainer.test(model=mnist_model, datamodule=mnist_data_module)

    logging.info("Training and testing finished.")

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
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

class MNISTLightningModule(pl.LightningModule):
    def __init__(self, lr=1.0): # Added lr argument as in mnist.py
        super().__init__()
        self.lr = lr
        self.model = MNISTNet()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output, target)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        # Using Adadelta as in the original MNIST example
        optimizer = optim.Adadelta(self.parameters(), lr=self.lr)
        return optimizer

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './mnist_data', batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # Attributes for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # download
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.val_dataset = datasets.MNIST(self.data_dir, train=False, transform=self.transform) # Using test set for validation
        
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    # class Net(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.cnn = nn.Sequential(
    #             nn.Conv2d(3, 6, 5),
    #             nn.ReLU(),
    #             nn.MaxPool2d(2, 2),
    #             nn.Conv2d(6, 16, 5),
    #             nn.ReLU(),
    #             nn.MaxPool2d(2, 2),
    #         )

    #         final_dim = 10
    #         # We add a useless 1GB intermediate layer so we spend more time in dist
    #         # communication so injected failures are more likely to cause issues
    #         # if they exist.
    #         target_size = 1_000_000_000
    #         self.useless = nn.Embedding(target_size // final_dim // 4, final_dim)

    #         self.classifier = nn.Sequential(
    #             nn.Linear(16 * 5 * 5, 120),
    #             nn.ReLU(),
    #             nn.Linear(120, 84),
    #             nn.ReLU(),
    #             nn.Linear(84, final_dim),
    #         )

    #     def forward(self, x):
    #         x = self.cnn(x)
    #         x = torch.flatten(x, 1)  # flatten all dimensions except batch
    #         x = self.classifier(x)
    #         x += self.useless.weight[0]
    #         return x

    # m = Net().to(device) # Commented out old Net instantiation
    # m = DistributedDataParallel(manager, m) # Commented out old Net DDP wrapping
    # optimizer = Optimizer(manager, optim.AdamW(m.parameters())) # Commented out old optimizer
    # criterion = nn.CrossEntropyLoss() # No longer needed, handled in LightningModule

    # print(m) # 'm' is no longer the model instance name
    # num_params = sum(p.numel() for p in m.parameters()) # Can be done inside LightningModule if needed
    # print(f"Total number of parameters: {num_params}")

    # sort_by_keyword = "self_" + device + "_time_total" # Profiler specific

    # def trace_handler(p): # Profiler specific
    #     output = p.key_averages().table(
    #         sort_by=sort_by_keyword,
    #         row_limit=100,
    #     )
    #     print(output)
    #     p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

    # # You can use an epoch based training but with faults it's easier to use step
    # # based training.
    # prof = torch.profiler.profile( # Profiler specific
    #     schedule=torch.profiler.schedule(wait=5, warmup=1, active=10, repeat=2),
    #     on_trace_ready=trace_handler,
    #     record_shapes=True,
    #     profile_memory=True,
    # )

    # prof.start() # Profiler specific
    # Old training loop removed
    # while True:
    #     for i, (inputs, labels) in enumerate(trainloader):
    #         prof.step()
    #         ... (rest of the loop) ...
    #         if manager.current_step() >= 10000:
    #             prof.stop()
    #             exit()


if __name__ == "__main__":
    main()