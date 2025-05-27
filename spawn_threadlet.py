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

from torchLoom.threadlet.threadlet import Threadlet
from torchLoom.common.constants import torchLoomConstants

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
        
    def update_learning_rate(self, new_lr: float):
        """Handler for learning rate updates."""
        old_lr = self.learning_rate
        self.learning_rate = float(new_lr)
        logger.info(f"Learning rate updated: {old_lr} -> {self.learning_rate}")
        
    def update_batch_size(self, new_batch_size: int):
        """Handler for batch size updates."""
        old_batch = self.batch_size
        self.batch_size = int(new_batch_size)
        logger.info(f"Batch size updated: {old_batch} -> {self.batch_size}")
        
    def pause_training(self):
        """Handler for pause training command."""
        self.is_training = False
        logger.info("Training paused")
        
    def resume_training(self):
        """Handler for resume training command."""
        self.is_training = True
        logger.info("Training resumed")
        
    def simulate_training_step(self):
        """Simulate a training step with metrics."""
        if not self.is_training:
            return
            
        # Simulate training progress
        self.current_step += 1
        if self.current_step % 100 == 0:
            self.current_epoch += 1
            
        # Simulate metrics (decreasing loss, improving accuracy)
        import random
        base_loss = max(0.1, 2.0 - (self.current_step * 0.001))
        loss = base_loss + random.uniform(-0.05, 0.05)
        
        base_accuracy = min(0.95, 0.5 + (self.current_step * 0.0005))
        accuracy = base_accuracy + random.uniform(-0.02, 0.02)
        
        # Send status update with metrics included
        logger.info(f"Calling publish_status with step: {self.current_step}")
        self.threadlet.publish_status(
            current_step=self.current_step,
            epoch=self.current_epoch,
            message=f"LR: {self.learning_rate}, Batch: {self.batch_size}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}",
            metrics={
                "loss": loss,
                "accuracy": accuracy,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size
            }
        )


class ThreadletRunner:
    """Main runner for the threadlet process."""
    
    def __init__(self, replica_id: str = None, torchLoom_addr: str = None):
        self.replica_id = replica_id or f"demo-threadlet-{uuid.uuid4().hex[:8]}"
        self.torchLoom_addr = torchLoom_addr or torchLoomConstants.DEFAULT_ADDR
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
        logger.info(f"Setting up threadlet with replica_id: {self.replica_id}")
        
        # Create threadlet instance
        self.threadlet = Threadlet(
            replica_id=self.replica_id,
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
            logger.info(f"Replica ID: {self.replica_id}")
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
    parser.add_argument("--replica-id", help="Replica ID for the threadlet")
    parser.add_argument("--torchLoom-addr", help="torchLoom NATS server address")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and run threadlet
    runner = ThreadletRunner(
        replica_id=args.replica_id,
        torchLoom_addr=args.torchLoom_addr
    )
    
    runner.run()


if __name__ == "__main__":
    main() 