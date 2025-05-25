import argparse
import time
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchLoom.common import (
    deviceStatus,
    TrainingStatus,
    create_batch_update_status,
    create_epoch_complete_status,
    create_epoch_start_status,
    create_training_complete_status,
    create_training_start_status,
)

# torchLoom imports
from torchLoom.threadlet import Threadlet, threadlet_handler


def simulate_device_status(
    replica_id: str, 
    batch_idx: int, 
    base_utilization: float = 60.0,
    base_temperature: float = 50.0
) -> deviceStatus:
    """Simulate realistic device status based on training progress."""
    # Simulate varying device metrics
    utilization = base_utilization + (batch_idx % 30)  # 60-90%
    temperature = base_temperature + (batch_idx % 25)   # 50-75°C
    memory_used = 2.0 + (batch_idx % 6) * 0.5  # 2-5 GB
    
    return deviceStatus(
        device_id=f"device_{replica_id}",
        replica_id=replica_id,
        server_id="local_server",
        status="active",
        utilization=utilization,
        temperature=temperature,
        memory_used=memory_used,
        memory_total=8.0
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
        self.replica_id = replica_id or f"train_integrated_{uuid.uuid4().hex[:8]}"

        # Training parameters (configurable via threadlet)
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.training_enabled = True
        self.verbose = getattr(args, 'verbose', False)  # Initialize from args

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
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

        return device

    def _register_handlers(self):
        """Register threadlet handlers for dynamic configuration."""

        @threadlet_handler("learning_rate", float)
        def update_learning_rate(new_lr: float):
            print(f"📈 Learning rate updated: {self.learning_rate} → {new_lr}")
            self.learning_rate = new_lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

        @threadlet_handler("batch_size", int)
        def update_batch_size(new_batch_size: int):
            print(f"📦 Batch size updated: {self.batch_size} → {new_batch_size}")
            self.batch_size = new_batch_size
            # Note: Would need to recreate DataLoader for this to take effect

        @threadlet_handler("training_enabled", bool)
        def toggle_training(enabled: bool):
            print(f"⏯️ Training {'enabled' if enabled else 'paused'}")
            self.training_enabled = enabled

        @threadlet_handler("verbose", bool)
        def toggle_verbose(enabled: bool):
            print(f"🗣️ Verbose mode {'enabled' if enabled else 'disabled'}")
            self.verbose = enabled

        # Register all handlers with the threadlet
        for local_var in locals().values():
            if callable(local_var) and hasattr(local_var, "_threadlet_config_key"):
                config_key = local_var._threadlet_config_key
                expected_type = getattr(local_var, "_threadlet_expected_type", None)
                self.threadlet.register_handler(config_key, local_var, expected_type)
                print(f"✅ Registered handler: {config_key}")

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
            time.sleep(0.01)
            # Check for configuration updates
            self.threadlet.check_and_apply_updates()
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
            # Convert status to dictionary if needed
            if hasattr(status, "to_dict"):
                status_dict = status.to_dict()
            else:
                status_dict = status

            # # Check if threadlet is available and not stopping
            if hasattr(self.threadlet, '_status_sender') and self.threadlet._status_sender:
                # Use non-blocking send to prevent hanging
                if hasattr(self.threadlet._status_sender, 'poll'):
                    # Check if pipe is ready for writing (not full)
                    if self.threadlet._status_sender.poll(0):
                        # Pipe might be ready for reading, but we want to write
                        # Fall back to normal send with error handling
                        pass
                    
                # Try to publish with error handling
                self.threadlet.publish_status(status_dict)
            else:
                if self.verbose:
                    print("Warning: Threadlet not available for status publishing")

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
                self.test_model()
                time.sleep(0.01)  # Brief pause between epochs

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
            final_device_status = simulate_device_status(self.replica_id, self.global_step)
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


def main():
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

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Create and run integrated trainer
    trainer = IntegratedTrainer(args, replica_id=args.replica_id)
    trainer.run_training()


if __name__ == "__main__":
    main()
