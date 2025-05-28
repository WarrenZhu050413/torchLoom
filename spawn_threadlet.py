#!/usr/bin/env python3
"""
Standalone script to spawn a threadlet that listens to weaver commands.
This script creates a threadlet instance and keeps it running to receive
configuration updates and send status/metrics updates to the weaver.
"""

import asyncio
import logging
import os
import platform
import random
import signal
import sys
import threading
import time
import uuid
from typing import Any, Dict

import psutil

from torchLoom.common.constants import NatsConstants
from torchLoom.threadlet.threadlet import Threadlet

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("spawn_threadlet")


class DemoTrainingProcess:
    """Mock training process that simulates training with metrics."""

    def __init__(self, threadlet: Threadlet, process_id: str):
        self.threadlet = threadlet
        self.process_id = process_id
        self.current_step = 0
        self.current_epoch = 0
        self.is_training = True
        self.learning_rate = 0.001
        self.batch_size = 32
        self.model_name = "demo_model"
        self.max_step = 10000  # Total training steps
        self.max_epoch = 100  # Total epochs
        self.training_start_time = time.time()

        # Mock device specifications
        self.device_uuid = f"mock-gpu-{uuid.uuid4().hex[:8]}"
        self.server_id = platform.node()  # System hostname
        self.mock_gpu_memory_total = 16.0  # 16GB total memory

        # Training configuration that can be updated
        self.config = {
            "learning_rate": str(self.learning_rate),
            "batch_size": str(self.batch_size),
            "optimizer": "AdamW",
            "scheduler": "cosine",
            "model_name": self.model_name,
        }

    def update_learning_rate(self, new_lr):
        """Handler for learning rate updates."""
        try:
            old_lr = self.learning_rate
            new_lr_value = float(new_lr)
            self.learning_rate = new_lr_value
            self.config["learning_rate"] = str(self.learning_rate)
            logger.info(
                f"PROCESS_ID: {self.process_id} - 🔄 Learning rate updated: {old_lr} -> {self.learning_rate} (received: '{new_lr}' as {type(new_lr)})"
            )
        except (ValueError, TypeError) as e:
            logger.error(
                f"PROCESS_ID: {self.process_id} - ❌ Failed to update learning rate: invalid value '{new_lr}' ({type(new_lr)}): {e}"
            )

    def update_batch_size(self, new_batch_size):
        """Handler for batch size updates."""
        try:
            old_batch = self.batch_size
            new_batch_size_value = int(new_batch_size)
            self.batch_size = new_batch_size_value
            self.config["batch_size"] = str(self.batch_size)
            logger.info(
                f"PROCESS_ID: {self.process_id} - 🔄 Batch size updated: {old_batch} -> {self.batch_size} (received: '{new_batch_size}' as {type(new_batch_size)})"
            )
        except (ValueError, TypeError) as e:
            logger.error(
                f"PROCESS_ID: {self.process_id} - ❌ Failed to update batch size: invalid value '{new_batch_size}' ({type(new_batch_size)}): {e}"
            )

    def pause_training(self):
        """Handler for pause training command."""
        self.is_training = False
        logger.info(f"PROCESS_ID: {self.process_id} - Training paused")

    def resume_training(self):
        """Handler for resume training command."""
        self.is_training = True
        logger.info(f"PROCESS_ID: {self.process_id} - Training resumed")

    def get_mock_device_status(self):
        """Generate mock GPU device status metrics."""
        # Mock GPU utilization (varies based on training state)
        if self.is_training:
            base_utilization = 75.0 + random.uniform(-15.0, 20.0)
        else:
            base_utilization = 5.0 + random.uniform(-3.0, 10.0)

        utilization = max(0.0, min(100.0, base_utilization))

        # Mock GPU temperature (higher when training)
        if self.is_training:
            base_temp = 65.0 + random.uniform(-5.0, 15.0)
        else:
            base_temp = 35.0 + random.uniform(-5.0, 10.0)

        temperature = max(30.0, min(85.0, base_temp))

        # Mock memory usage (increases with batch size and utilization)
        memory_factor = (utilization / 100.0) * (self.batch_size / 32.0)
        memory_used = self.mock_gpu_memory_total * (
            0.3 + 0.5 * memory_factor + random.uniform(-0.1, 0.1)
        )
        memory_used = max(1.0, min(self.mock_gpu_memory_total, memory_used))

        # device_uuid and process_id are handled by Threadlet.publish_device_status
        # device_uuid is passed as a named argument, process_id is self._process_id from Threadlet
        return {
            "server_id": self.server_id,
            "utilization": utilization,
            "temperature": temperature,
            "memory_used": memory_used,
            "memory_total": self.mock_gpu_memory_total,
        }

    def simulate_training_step(self):
        """Simulate a training step with complete metrics."""
        if not self.is_training:
            return

        # Simulate training progress
        self.current_step += 1
        if self.current_step % 100 == 0:
            self.current_epoch += 1

        # Calculate training time
        training_time = time.time() - self.training_start_time

        # Simulate metrics (decreasing loss, improving accuracy)
        base_loss = max(0.1, 2.0 - (self.current_step * 0.001))
        loss = base_loss + random.uniform(-0.05, 0.05)

        base_accuracy = min(0.95, 0.5 + (self.current_step * 0.0005))
        accuracy = base_accuracy + random.uniform(-0.02, 0.02)

        # Additional training metrics
        gradient_norm = random.uniform(0.5, 2.0)
        lr_current = self.learning_rate * (
            0.95 ** (self.current_epoch // 10)
        )  # Decay LR

        # Prepare complete training status data
        metrics = {
            "loss": str(round(loss, 4)),
            "accuracy": str(round(accuracy, 4)),
            "learning_rate": str(round(lr_current, 6)),
            "batch_size": str(self.batch_size),
            "gradient_norm": str(round(gradient_norm, 3)),
            "throughput": str(
                round(self.batch_size / (2.0 + random.uniform(-0.5, 0.5)), 2)
            ),  # samples/sec
        }

        status_message = f"Training | LR: {lr_current:.6f}, Batch: {self.batch_size}, Loss: {loss:.4f}, Acc: {accuracy:.4f}"

        # Send comprehensive training status update
        logger.info(
            f"PROCESS_ID: {self.process_id} - Publishing training status: step={self.current_step}, epoch={self.current_epoch}"
        )
        status_data = {
            "current_step": self.current_step,
            "epoch": self.current_epoch,
            "message": status_message,
            "metrics": metrics,
            "training_time": training_time,
            "max_step": self.max_step,
            "max_epoch": self.max_epoch,
            "config": self.config,
        }
        self.threadlet.publish_training_status(status_data=status_data)

        # Send device status update every few steps
        if self.current_step % 5 == 0:  # Update device status every 5 training steps
            device_status_metrics = self.get_mock_device_status()
            logger.info(
                f"PROCESS_ID: {self.process_id} - Publishing device status: utilization={device_status_metrics['utilization']:.1f}%, temp={device_status_metrics['temperature']:.1f}°C for device_uuid={self.device_uuid}"
            )
            self.threadlet.publish_device_status(
                device_uuid=self.device_uuid,  # Pass device_uuid explicitly
                status_data=device_status_metrics,  # Pass metrics in status_data dict
            )


class ThreadletRunner:
    """Main runner for the threadlet process."""

    def __init__(
        self,
        process_id: str = None,
        device_uuid: str = None,
        torchLoom_addr: str = None,
    ):
        self.process_id = process_id or f"demo-threadlet-{uuid.uuid4().hex[:8]}"
        self.device_uuid = device_uuid or f"demo-replica-{uuid.uuid4().hex[:8]}"
        self.torchLoom_addr = torchLoom_addr or NatsConstants.DEFAULT_ADDR
        self.threadlet = None
        self.training_process = None
        self.running = True
        self._shutdown_event = threading.Event()

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating immediate shutdown...")
        self.running = False
        self._shutdown_event.set()

        # Immediately stop the threadlet
        if self.threadlet:
            try:
                logger.info("Stopping threadlet immediately...")
                self.threadlet.stop()
                logger.info("Threadlet stopped by signal handler")
            except Exception as e:
                logger.exception(f"Error stopping threadlet in signal handler: {e}")

    def setup_threadlet(self):
        """Initialize and configure the threadlet."""
        logger.info(
            f"Setting up threadlet with process_id: {self.process_id}, device_uuid: {self.device_uuid}"
        )

        # Create threadlet instance
        self.threadlet = Threadlet(
            process_id=self.process_id,
            device_uuid=self.device_uuid,
            torchLoom_addr=self.torchLoom_addr,
        )

        # Create mock training process
        self.training_process = DemoTrainingProcess(self.threadlet, self.process_id)

        # Register configuration handlers
        self.threadlet.register_handler(
            "learning_rate", self.training_process.update_learning_rate
        )
        self.threadlet.register_handler(
            "batch_size", self.training_process.update_batch_size
        )
        self.threadlet.register_handler(
            "pause_training", lambda: self.training_process.pause_training()
        )
        self.threadlet.register_handler(
            "resume_training", lambda: self.training_process.resume_training()
        )

        logger.info("Threadlet configuration handlers registered")

    def start_threadlet(self):
        """Start the threadlet process."""
        try:
            logger.info("Starting threadlet...")
            self.threadlet.start()
            logger.info("Threadlet started successfully")

        except Exception as e:
            logger.exception(f"Failed to start threadlet: {e}")
            raise

    def run_training_loop(self):
        """Main training simulation loop with responsive shutdown handling."""
        logger.info("Starting training simulation loop...")

        step_interval = 2.0  # Seconds between training steps
        sleep_chunk = 0.1  # Check for shutdown every 100ms

        try:
            while self.running and not self._shutdown_event.is_set():
                # Simulate training step
                if self.running:
                    self.training_process.simulate_training_step()

                # Responsive sleep - check shutdown signal every 100ms
                slept = 0.0
                while (
                    slept < step_interval
                    and self.running
                    and not self._shutdown_event.is_set()
                ):
                    time.sleep(sleep_chunk)
                    slept += sleep_chunk

        except KeyboardInterrupt:
            logger.info("Training loop interrupted by user")
        except Exception as e:
            logger.exception(f"Error in training loop: {e}")
        finally:
            logger.info("Training loop stopped")

    def stop_threadlet(self):
        """Stop the threadlet and clean up."""
        if self.threadlet:
            try:
                logger.info("Stopping threadlet...")
                self.threadlet.stop()
                logger.info("Threadlet stopped successfully")
            except Exception as e:
                logger.exception(f"Error stopping threadlet: {e}")

    def run(self):
        """Main run method."""
        try:
            logger.info("=== torchLoom Threadlet Runner ===")
            logger.info(f"Replica ID: {self.process_id}")
            logger.info(f"Device UUID: {self.device_uuid}")
            logger.info(f"torchLoom Address: {self.torchLoom_addr}")
            logger.info("\nThis script will:")
            logger.info("1. Create and start a threadlet process")
            logger.info("2. Register configuration handlers")
            logger.info("3. Simulate training with metrics")
            logger.info("4. Listen for weaver commands")
            logger.info("\nPress Ctrl+C to exit\n")

            # Setup and start threadlet
            self.setup_threadlet()
            self.start_threadlet()

            # Run the training simulation
            self.run_training_loop()

        except Exception as e:
            logger.exception(f"Error in main run: {e}")
        finally:
            # Clean up
            self.stop_threadlet()
            logger.info("Shutdown complete")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Spawn a threadlet to listen to weaver commands"
    )
    parser.add_argument("--process-id", help="Replica ID for the threadlet")
    parser.add_argument(
        "--device-uuid",
        help="Device UUID for the threadlet (defaults to demo-replica-<random>)",
    )
    parser.add_argument("--torchLoom-addr", help="torchLoom NATS server address")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Create and run threadlet
    runner = ThreadletRunner(
        process_id=args.process_id,
        device_uuid=args.device_uuid,
        torchLoom_addr=args.torchLoom_addr,
    )

    runner.run()


if __name__ == "__main__":
    main()
