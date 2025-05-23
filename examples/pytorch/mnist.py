import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Remove torchvision imports
# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import asyncio
import nats
from torchLoom.common.constants import torchLoomConstants
import time
import threading
from nats.js.api import StreamConfig
from torch.utils.data import Dataset, DataLoader

# Define a simple random dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples, input_shape=(1, 28, 28), num_classes=10):
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.num_classes = num_classes
        # Generate random data and labels
        # Data: random tensors of shape input_shape
        # Labels: random integers between 0 and num_classes-1
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
        # The input size 9216 is specific to 28x28 input after conv/pool
        # Keep it for now as we'll generate 28x28 random data
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


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # Use len(train_loader.dataset) for total samples
    total_samples = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # Print progress based on batch index and total batches
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), total_samples,
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # Use len(test_loader.dataset) for total samples
    total_samples = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total_samples

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total_samples,
        100. * correct / total_samples))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Random Data Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
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
    # Add arguments for random dataset size
    parser.add_argument('--train-samples', type=int, default=60000,
                        help='Number of random training samples (default: 60000)')
    parser.add_argument('--test-samples', type=int, default=10000,
                        help='Number of random test samples (default: 10000)')
    args = parser.parse_args()

    # Check for accelerator availability using torch.cuda or torch.backends.mps
    use_accel = not args.no_accel and (torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()))

    torch.manual_seed(args.seed)

    if use_accel:
        # Use CUDA if available, otherwise MPS if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
             device = torch.device("mps")
        else:
             device = torch.device("cpu") # Fallback if check was wrong
        print(f"Using accelerator: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")


    async def check_jetstream_for_lr_change(optimizer):
        # from nats.js.api import StreamConfig # Already imported

        try:
            # 连接 NATS 并获取 JetStream 客户端
            nc = await nats.connect(torchLoomConstants.DEFAULT_ADDR)
            js = nc.jetstream()

            # ✅ 确保创建了 stream（只需创建一次，可放到初始化阶段）
            # This should ideally be done once outside the loop, but keeping it here for simplicity
            # in this example rewrite. In a real app, manage stream creation separately.
            try:
                await js.add_stream(StreamConfig(
                    name="LR_STREAM",
                    subjects=["torchLoom.training.reset_lr"]
                ))
            except Exception as e:
                 # Ignore error if stream already exists
                 if "stream exists" not in str(e):
                     print(f"Warning: Could not add stream, might already exist: {e}")


            # ✅ 拉取最新一条消息（从 durable consumer）
            # Use a try-except block for fetch timeout
            try:
                consumer = await js.pull_subscribe("torchLoom.training.reset_lr", durable="lr_durable")
                messages = await consumer.fetch(1, timeout=0.1) # Use a small timeout
                for msg in messages:
                    new_lr = float(msg.data.decode())
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"✅ [JetStream] Applied new learning rate: {new_lr}")
                    await msg.ack()
            except asyncio.TimeoutError:
                # print("⏳ [JetStream] No new learning rate message this round.") # Suppress frequent message
                pass # No message is normal
            except Exception as e:
                print(f"❌ [JetStream] Error fetching/processing message: {e}")
            finally:
                 # Ensure NATS connection is closed
                 if nc and nc.is_connected:
                     await nc.drain()


        except Exception as e:
            print(f"❌ [JetStream] Error connecting to NATS: {e}")


    # Create random datasets
    train_dataset = RandomDataset(args.train_samples)
    test_dataset = RandomDataset(args.test_samples)

    # Create DataLoaders
    # No need for num_workers, pin_memory, shuffle with random data
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)


    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma) # Keep commented out

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        # scheduler.step() # Keep commented out
        time.sleep(0.01) # Keep sleep if desired, or remove for faster random data training
        # asyncio.run(check_nats_for_lr_change(optimizer)) # Keep commented out
        asyncio.run(check_jetstream_for_lr_change(optimizer))
        print("Current learning rate:", optimizer.param_groups[0]['lr'])

    if args.save_model:
        torch.save(model.state_dict(), "random_cnn.pt") # Change filename


if __name__ == '__main__':
    main()