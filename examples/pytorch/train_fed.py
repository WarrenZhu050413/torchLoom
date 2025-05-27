import argparse
import os
import time
import uuid

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchLoom.common import (
    TrainingStatus,
    create_batch_update_status,
    create_epoch_complete_status,
    create_epoch_start_status,
    create_training_complete_status,
    create_training_start_status,
    deviceStatus,
)

# torchLoom imports
from torchLoom.threadlet import Threadlet


def simulate_device_status(
    replica_id: str,
    batch_idx: int,
    base_utilization: float = 60.0,
    base_temperature: float = 50.0,
) -> deviceStatus:
    """Simulate realistic device status based on training progress."""
    # Simulate varying device metrics
    utilization = base_utilization + (batch_idx % 30)  # 60-90%
    temperature = base_temperature + (batch_idx % 25)  # 50-75°C
    memory_used = 2.0 + (batch_idx % 6) * 0.5  # 2-5 GB

    return deviceStatus(
        device_id=f"device_{replica_id}",
        replica_id=replica_id,
        server_id="local_server",
        status="active",
        utilization=utilization,
        temperature=temperature,
        memory_used=memory_used,
        memory_total=8.0,
    )


# Simple random dataset for testing
class RandomDataset(Dataset):
    def __init__(self, num_samples, input_shape=(1, 28, 28), num_classes=10):
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.num_classes = num_classes
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


class IntegratedTrainer:
    """Training class with full torchLoom integration and comprehensive status publishing."""

    def __init__(self, args, replica_id=None):
        self.args = args

        self.sync_frequency = getattr(args, "sync_frequency", 5)

        self.replica_id = replica_id or f"train_integrated_{uuid.uuid4().hex[:8]}"

        # Training parameters (configurable via threadlet)
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.training_enabled = True
        self.verbose = getattr(args, "verbose", False)  # Initialize from args

        # Training tracking
        self.start_time = time.time()
        self.current_epoch = 0
        self.global_step = 0

        # Setup device
        self.device = self._setup_device()

        # Initialize model and optimizer
        self.model = Net().to(self.device)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.learning_rate)

        # Setup datasets
        train_dataset = RandomDataset(args.train_samples)
        test_dataset = RandomDataset(args.test_samples)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)

        # Initialize threadlet for torchLoom integration
        self.threadlet = Threadlet(replica_id=self.replica_id)
        self._register_handlers()
        self.threadlet.start()

        print(f"🧵 Integrated trainer initialized with replica_id: {self.replica_id}")
        print(f"📊 Will publish comprehensive status to torchLoom UI")

    def _setup_device(self):
        """Setup compute device."""
        use_accel = not self.args.no_accel and (
            torch.cuda.is_available()
            or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        )

        if use_accel:
            if torch.cuda.is_available():
                group_id = getattr(self.args, "replica_group_id", 0)
                rank = getattr(self.args, "rank", 0)
                device = _get_device_for_rank(rank, group_id)
                # device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

        return device

    def _register_handlers(self):
        """Register threadlet handlers for dynamic configuration."""

        def update_learning_rate(new_lr: float):
            print(f"📈 Learning rate updated: {self.learning_rate} → {new_lr}")
            self.learning_rate = new_lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

        def update_batch_size(new_batch_size: int):
            print(f"📦 Batch size updated: {self.batch_size} → {new_batch_size}")
            self.batch_size = new_batch_size
            # Note: Would need to recreate DataLoader for this to take effect

        def toggle_training(enabled: bool):
            print(f"⏯️ Training {'enabled' if enabled else 'paused'}")
            self.training_enabled = enabled

        def toggle_verbose(enabled: bool):
            print(f"🗣️ Verbose mode {'enabled' if enabled else 'disabled'}")
            self.verbose = enabled

        def update_sync_frequency(new_freq: int):
            print(f"🔁 Sync frequency updated: {self.sync_frequency} → {new_freq}")
            self.sync_frequency = new_freq

        # Register all handlers directly with the threadlet
        self.threadlet.register_handler("learning_rate", update_learning_rate, float)
        self.threadlet.register_handler("batch_size", update_batch_size, int)
        self.threadlet.register_handler("training_enabled", toggle_training, bool)
        self.threadlet.register_handler("verbose", toggle_verbose, bool)
        self.threadlet.register_handler("sync_frequency", update_sync_frequency, int)

        print(f"✅ Registered 5 configuration handlers with threadlet")

    def train_epoch(self, epoch):
        """Train for one epoch with comprehensive status reporting."""
        self.current_epoch = epoch
        self.model.train()

        # Handle dataset length safely
        # Use the known size from the dataset since RandomDataset implements __len__
        total_samples = self.args.train_samples
        # Publish epoch start status
        epoch_start_status = create_epoch_start_status(
            replica_id=self.replica_id,
            epoch=epoch,
            total_batches=len(self.train_loader),
        )
        self.publish_status(epoch_start_status)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            time.sleep(0.1)
            # Configuration updates are handled automatically by the threadlet
            # Skip training if disabled
            if not self.training_enabled:
                print("⏸️ Training paused - skipping batch")
                time.sleep(0.1)
                continue
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.global_step += 1

            # Publish comprehensive status updates
            if batch_idx % self.args.log_interval == 0:
                progress_percent = 100.0 * batch_idx / len(self.train_loader)

                if self.verbose:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            total_samples,
                            progress_percent,
                            loss.item(),
                        )
                    )

                # Enhanced training status with more detailed metrics
                training_status = create_batch_update_status(
                    replica_id=self.replica_id,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    step=self.global_step,
                    loss=loss.item(),
                    learning_rate=self.optimizer.param_groups[0]["lr"],
                    step_progress=progress_percent,
                )
                # Add comprehensive training metrics
                training_status.training_time = time.time() - self.start_time
                training_status.epoch_progress = progress_percent
                training_status.metrics.update(
                    {
                        "batch_size": str(self.batch_size),
                        "optimizer_type": "Adadelta",
                        "device": str(self.device),
                        "samples_processed": str(batch_idx * len(data)),
                        "total_samples": str(total_samples),
                        "batches_completed": str(batch_idx),
                        "total_batches": str(len(self.train_loader)),
                        "model_parameters": str(
                            sum(p.numel() for p in self.model.parameters())
                        ),
                        "trainable_parameters": str(
                            sum(
                                p.numel()
                                for p in self.model.parameters()
                                if p.requires_grad
                            )
                        ),
                        "gradient_norm": str(
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_norm=float("inf")
                            )
                        ),
                        "memory_allocated": (
                            str(torch.cuda.memory_allocated() / 1024**3)
                            if torch.cuda.is_available()
                            else "0"
                        ),
                        "memory_reserved": (
                            str(torch.cuda.memory_reserved() / 1024**3)
                            if torch.cuda.is_available()
                            else "0"
                        ),
                        "training_enabled": str(self.training_enabled),
                        "verbose_mode": str(self.verbose),
                    }
                )
                self.publish_status(training_status)
                print(f"F6: batchidx {batch_idx}")

                # Only publish device status every 5th batch to reduce message frequency
                if batch_idx % 5 == 0:
                    # Enhanced device status with more realistic simulation
                    device_status = simulate_device_status(
                        replica_id=self.replica_id, batch_idx=batch_idx
                    )
                    # Add comprehensive training configuration to device status
                    device_status.config.update(
                        {
                            "batch_size": str(self.batch_size),
                            "learning_rate": str(self.learning_rate),
                            "optimizer_type": "Adadelta",
                            "epoch": str(epoch),
                            "current_step": str(self.global_step),
                            "training_enabled": str(self.training_enabled),
                            "device_type": str(self.device),
                            "model_name": "CNN",
                            "dataset_name": "RandomDataset",
                        }
                    )
                    self.publish_status(device_status)

                if self.args.dry_run:
                    break

        print("F7")
        # Publish epoch completion status
        epoch_complete_status = create_epoch_complete_status(
            replica_id=self.replica_id, epoch=epoch
        )
        self.publish_status(epoch_complete_status)
        print("F8")

    def test_model(self):
        """Test the model and publish results."""
        self.model.eval()
        test_loss = 0
        correct = 0

        # Handle dataset length safely
        # Use the known size from the dataset since RandomDataset implements __len__
        total_samples = self.args.test_samples

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= total_samples
        accuracy = 100.0 * correct / total_samples

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, total_samples, accuracy
            )
        )

        # Publish comprehensive test results
        test_status = TrainingStatus(
            replica_id=self.replica_id,
            status_type="test_complete",
            epoch=self.current_epoch,
            current_step=self.global_step,
            status="training",
            metrics={
                "test_loss": str(test_loss),
                "test_accuracy": str(accuracy),
                "correct_predictions": str(correct),
                "total_test_samples": str(total_samples),
                "test_error_rate": str(100.0 - accuracy),
                "model_confidence": str(accuracy / 100.0),
                "test_batches": str(len(self.test_loader)),
                "avg_loss_per_sample": str(test_loss),
            },
        )
        self.publish_status(test_status)

    def publish_status(self, status):
        """Publish any type of status (TrainingStatus, deviceStatus)."""
        try:
            # For protobuf objects, publish directly
            if hasattr(status, "replica_id"):
                # This is likely a TrainingStatus
                self.threadlet.publish_metrics(
                    step=getattr(status, "current_step", 0),
                    epoch=getattr(status, "epoch", 0),
                    loss=(
                        float(status.metrics.get("loss", 0))
                        if hasattr(status, "metrics")
                        else None
                    ),
                    **(
                        {k: v for k, v in status.metrics.items() if k != "loss"}
                        if hasattr(status, "metrics")
                        else {}
                    ),
                )
            elif hasattr(status, "device_id"):
                # This is likely a deviceStatus - for now just log it
                if self.verbose:
                    print(f"📱 Device status: {status.device_id} - {status.status}")
            else:
                # Fallback for other status types
                if self.verbose:
                    print(f"📊 Status update: {status}")

        except Exception as e:
            print(f"Warning: Failed to publish status: {e}")

    def run_training(self):
        """Main training loop with comprehensive status reporting."""
        try:
            print(f"🚀 Starting training for {self.args.epochs} epochs...")

            # Publish training start status
            training_start_status = create_training_start_status(
                replica_id=self.replica_id, epochs=self.args.epochs
            )
            self.publish_status(training_start_status)

            # Initial device status
            initial_device_status = simulate_device_status(self.replica_id, 0)
            initial_device_status.config.update(
                {
                    "batch_size": str(self.batch_size),
                    "learning_rate": str(self.learning_rate),
                    "optimizer_type": "Adadelta",
                }
            )
            self.publish_status(initial_device_status)

            for epoch in range(1, self.args.epochs + 1):
                self.train_epoch(epoch)
                self._perform_model_sync()
                self.test_model()
                time.sleep(0.1)  # Brief pause between epochs

            # Publish training completion with final metrics
            final_metrics = {
                "total_epochs": self.args.epochs,
                "total_steps": self.global_step,
                "training_time": time.time() - self.start_time,
            }

            training_complete_status = create_training_complete_status(
                replica_id=self.replica_id, final_metrics=final_metrics
            )
            self.publish_status(training_complete_status)

            # Final device status
            final_device_status = simulate_device_status(
                self.replica_id, self.global_step
            )
            final_device_status.status = "completed"
            self.publish_status(final_device_status)

            if self.args.save_model:
                torch.save(self.model.state_dict(), "random_cnn_integrated.pt")
                print("💾 Model saved")

        except KeyboardInterrupt:
            print("🛑 Training interrupted by user")
            interrupted_status = TrainingStatus(
                replica_id=self.replica_id,
                status_type="training_interrupted",
                current_step=self.global_step,
                epoch=self.current_epoch,
                status="stopped",
                metrics={"reason": "user_interrupt"},
            )
            self.publish_status(interrupted_status)
        except Exception as e:
            print(f"❌ Training failed: {e}")
            failed_status = TrainingStatus(
                replica_id=self.replica_id,
                status_type="training_failed",
                current_step=self.global_step,
                epoch=self.current_epoch,
                status="failed",
                metrics={"error": str(e)},
            )
            self.publish_status(failed_status)
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, "threadlet"):
                self.threadlet.stop()
                print("🧵 Threadlet stopped")
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Random Data Example with Comprehensive torchLoom Integration"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument("--no-accel", action="store_true", help="disables accelerator")
    parser.add_argument(
        "--dry-run", action="store_true", help="quickly check a single pass"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model", action="store_true", help="For Saving the current Model"
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=60000,
        help="Number of random training samples (default: 60000)",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=10000,
        help="Number of random test samples (default: 10000)",
    )
    parser.add_argument(
        "--replica-id",
        type=str,
        default=None,
        help="Replica ID for torchLoom (auto-generated if not provided)",
    )
    parser.add_argument(
        "--world-size", type=int, default=1, help="Number of processes (default: 1)"
    )
    parser.add_argument(
        "--replica-group-id", type=int, default=0, help="Replica group ID (default: 0)"
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=1,
        help="Number of replica groups to simulate (default: 1)",
    )
    return parser.parse_args()


def get_device_for_rank(
    rank: int, group_id: int, gpus_per_group: int = 8
) -> torch.device:
    """
    Map rank within a replica group to a physical device index.
    """
    base_gpu_index = group_id * gpus_per_group
    assigned_gpu = base_gpu_index + (rank % gpus_per_group)
    return torch.device(f"cuda:{assigned_gpu}")


def _get_device_for_rank(
    rank: int, group_id: int, gpus_per_group: int = 1
) -> torch.device:
    """Always return cuda:0 for simulated multi-GPU on single GPU machine."""
    return torch.device("cuda:0")


def init_distributed(rank: int, world_size: int, group_id: int):
    if torch.cuda.is_available():
        backend = "nccl"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        backend = "gloo"
    else:
        backend = "gloo"

    master_addr = "127.0.0.1"
    master_port = 23456 + group_id
    init_method = f"tcp://{master_addr}:{master_port}"

    print(
        f"[group {group_id}][rank {rank}] Initializing process group on {init_method}"
    )
    time.sleep(3)
    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )


def main_worker(rank: int, args: argparse.Namespace):
    torch.manual_seed(args.seed)

    args.rank = rank
    args.replica_id = args.replica_id or f"group{args.replica_group_id}-rank{rank}"

    init_distributed(rank, args.world_size, args.replica_group_id)

    trainer = IntegratedTrainer(args, replica_id=args.replica_id)
    trainer.run_training()

    if torch.distributed.is_initialized():
        for param in trainer.model.parameters():
            if param.requires_grad:
                torch.distributed.all_reduce(
                    param.data, op=torch.distributed.ReduceOp.SUM
                )
                param.data /= args.world_size


if __name__ == "__main__":
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_SHM_DISABLE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    mp.set_start_method("spawn", force=True)

    args = parse_args()

    num_gpus_in_each_group = np.round(
        np.linspace(0, args.world_size, args.num_groups + 1)
    )
    num_gpus_in_each_group = num_gpus_in_each_group[1:] - num_gpus_in_each_group[:-1]
    num_gpus_in_each_group = num_gpus_in_each_group.astype(int)

    procs = []
    global_rank = 0
    for group_id, num_gpus in enumerate(num_gpus_in_each_group):
        for local_rank in range(num_gpus):
            # Create per-process arguments
            group_args = argparse.Namespace(**vars(args))
            group_args.replica_group_id = group_id
            group_args.rank = global_rank
            group_args.local_rank_in_group = local_rank
            group_args.world_size = args.world_size

            print(
                f"🚀 Launching rank {global_rank} in group {group_id} (local rank: {local_rank})"
            )
            p = mp.Process(target=main_worker, args=(global_rank, group_args))
            p.start()
            procs.append(p)
            global_rank += 1

    for p in procs:
        p.join()
