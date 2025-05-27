#!/usr/bin/env python3
"""
Standalone script to spawn a threadlet that listens to weaver commands.
This script creates a threadlet instance and keeps it running to receive
configuration updates and send status/metrics updates to the weaver.
"""

import asyncio
import logging
import signal
import sys
import time
import uuid
from typing import Any, Dict
import random
import psutil

from torchLoom.threadlet.threadlet import Threadlet
from torchLoom.common.constants import NatsConstants

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("spawn_threadlet")


class DemoTrainingProcess:
    """Mock training process that simulates training with metrics."""
    
    def __init__(self, threadlet: Threadlet):
        self.threadlet = threadlet
        self.current_step = 0
        self.current_epoch = 0
        self.is_training = True
        self.learning_rate = 0.001
        self.batch_size = 32
        self.model_name = "demo_model"
        self.max_step = 10000  # Total training steps
        self.max_epoch = 100   # Total epochs
        self.training_start_time = time.time()
        
        # Mock device specifications
        self.device_uuid = f"mock-gpu-{uuid.uuid4().hex[:8]}"
        self.server_id = os.hostname()  # System hostname
        self.mock_gpu_memory_total = 16.0  # 16GB total memory
        
        # Training configuration that can be updated
        self.config = {
            "learning_rate": str(self.learning_rate),
            "batch_size": str(self.batch_size),
            "optimizer": "AdamW",
            "scheduler": "cosine",
            "model_name": self.model_name
        }
        
    def update_learning_rate(self, new_lr: float):
        """Handler for learning rate updates."""
        old_lr = self.learning_rate
        self.learning_rate = float(new_lr)
        self.config["learning_rate"] = str(self.learning_rate)
        logger.info(f"Learning rate updated: {old_lr} -> {self.learning_rate}")
        
    def update_batch_size(self, new_batch_size: int):
        """Handler for batch size updates."""
        old_batch = self.batch_size
        self.batch_size = int(new_batch_size)
        self.config["batch_size"] = str(self.batch_size)
        logger.info(f"Batch size updated: {old_batch} -> {self.batch_size}")
        
    def pause_training(self):
        """Handler for pause training command."""
        self.is_training = False
        logger.info("Training paused")
        
    def resume_training(self):
        """Handler for resume training command."""
        self.is_training = True
        logger.info("Training resumed")
        
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
        memory_used = self.mock_gpu_memory_total * (0.3 + 0.5 * memory_factor + random.uniform(-0.1, 0.1))
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
        lr_current = self.learning_rate * (0.95 ** (self.current_epoch // 10))  # Decay LR
        
        # Prepare complete training status data
        metrics = {
            "loss": str(round(loss, 4)),
            "accuracy": str(round(accuracy, 4)),
            "learning_rate": str(round(lr_current, 6)),
            "batch_size": str(self.batch_size),
            "gradient_norm": str(round(gradient_norm, 3)),
            "throughput": str(round(self.batch_size / (2.0 + random.uniform(-0.5, 0.5)), 2)),  # samples/sec
        }
        
        status_message = f"Training | LR: {lr_current:.6f}, Batch: {self.batch_size}, Loss: {loss:.4f}, Acc: {accuracy:.4f}"
        
        # Send comprehensive training status update
        logger.info(f"Publishing training status: step={self.current_step}, epoch={self.current_epoch}")
        self.threadlet.publish_training_status(
            current_step=self.current_step,
            epoch=self.current_epoch,
            message=status_message,
            metrics=metrics,
            training_time=training_time,
            max_step=self.max_step,
            max_epoch=self.max_epoch,
            config=self.config,
        )
        
        # Send device status update every few steps
        if self.current_step % 5 == 0:  # Update device status every 5 training steps
            device_status_metrics = self.get_mock_device_status()
            logger.info(f"Publishing device status: utilization={device_status_metrics['utilization']:.1f}%, temp={device_status_metrics['temperature']:.1f}°C for device_uuid={self.device_uuid}")
            self.threadlet.publish_device_status(
                device_uuid=self.device_uuid, # Pass device_uuid explicitly
                **device_status_metrics   # Pass other metrics as kwargs
            )


class ThreadletRunner:
    """Main runner for the threadlet process."""
    
    def __init__(self, process_id: str = None, torchLoom_addr: str = None):
        self.process_id = process_id or f"demo-threadlet-{uuid.uuid4().hex[:8]}"
        self.torchLoom_addr = torchLoom_addr or NatsConstants.DEFAULT_ADDR
        self.threadlet = None
        self.training_process = None
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    def setup_threadlet(self):
        """Initialize and configure the threadlet."""
        logger.info(f"Setting up threadlet with process_id: {self.process_id}")
        
        # Create threadlet instance
        self.threadlet = Threadlet(
            process_id=self.process_id,
            torchLoom_addr=self.torchLoom_addr
        )
        
        # Create mock training process
        self.training_process = DemoTrainingProcess(self.threadlet)
        
        # Register configuration handlers
        self.threadlet.register_handler("learning_rate", self.training_process.update_learning_rate)
        self.threadlet.register_handler("batch_size", self.training_process.update_batch_size)
        self.threadlet.register_handler("pause_training", lambda: self.training_process.pause_training())
        self.threadlet.register_handler("resume_training", lambda: self.training_process.resume_training())
        
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
        """Main training simulation loop."""
        logger.info("Starting training simulation loop...")
        
        step_interval = 2.0  # Seconds between training steps
        
        try:
            while self.running:
                current_time = time.time()
                
                # Simulate training step
                self.training_process.simulate_training_step()
                
                # Wait before next step
                time.sleep(step_interval)
                
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
    
    parser = argparse.ArgumentParser(description="Spawn a threadlet to listen to weaver commands")
    parser.add_argument("--process-id", help="Replica ID for the threadlet")
    parser.add_argument("--torchLoom-addr", help="torchLoom NATS server address")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and run threadlet
    runner = ThreadletRunner(
        process_id=args.process_id,
        torchLoom_addr=args.torchLoom_addr
    )
    
    runner.run()


if __name__ == "__main__":
    main() 