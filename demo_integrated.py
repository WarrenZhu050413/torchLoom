#!/usr/bin/env python3
"""
Comprehensive demo of the integrated torchLoom system.

This script demonstrates the full integration between:
- Enhanced Weavelet with status reporting
- Weaver with UI support and WebSocket server
- Vue.js UI with real-time updates
- NATS messaging for coordination

The demo shows:
1. Starting the Weaver with UI support
2. Running enhanced weavelets that report status
3. Real-time UI updates via WebSocket
4. Interactive GPU deactivation/reactivation
5. Configuration updates flowing through the system
"""

import asyncio
import multiprocessing as mp
import time
import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from torchLoom.weaver.core import Weaver
from torchLoom.log.logger import setup_logger
from torchLoom.constants import torchLoomConstants
from demo_weavelet import DemoTrainer
import lightning as L
from torch.utils.data import DataLoader
from train import RandomTextDataset

logger = setup_logger(name="integrated_demo")


class EnhancedDemoTrainer(DemoTrainer):
    """Enhanced demo trainer with status reporting to weaver."""
    
    def __init__(self, vocab_size: int, replica_id: str = "demo_replica_1"):
        super().__init__(vocab_size)
        self.replica_id = replica_id
        self.status_report_interval = 5  # Report every 5 training steps
        self.step_counter = 0
        
    def training_step(self, batch, batch_idx):
        """Enhanced training step with status reporting."""
        # Call parent training step
        loss = self._user_training_step(batch, batch_idx)
        
        # Report status periodically
        self.step_counter += 1
        if self.step_counter % self.status_report_interval == 0:
            asyncio.create_task(self._report_status())
        
        return loss
    
    async def _report_status(self):
        """Report training status to weaver via NATS."""
        try:
            import nats
            from torchLoom.proto.torchLoom_pb2 import EventEnvelope, GPUStatus, TrainingProgress
            
            # Connect to NATS (reuse connection if available)
            nc = await nats.connect(torchLoomConstants.DEFAULT_ADDR)
            
            # Create GPU status update
            env = EventEnvelope()
            gpu_status = env.gpu_status
            gpu_status.gpu_id = f"gpu-{self.replica_id.split('_')[-1]}-0"
            gpu_status.replica_id = self.replica_id
            gpu_status.server_id = f"server-{self.replica_id.split('_')[-1]}-0"
            gpu_status.status = "active"
            gpu_status.utilization = 75.0 + (self.step_counter % 20)  # 75-95%
            gpu_status.temperature = 60.0 + (self.step_counter % 15)  # 60-75°C
            
            # Add current config
            gpu_status.config["batch_size"] = str(self.batch_size)
            gpu_status.config["learning_rate"] = str(self.learning_rate)
            gpu_status.config["optimizer_type"] = self.optimizer_type
            
            # Publish GPU status
            await nc.publish(torchLoomConstants.subjects.GPU_STATUS, env.SerializeToString())
            
            # Create training progress update
            env2 = EventEnvelope()
            progress = env2.training_progress
            progress.replica_id = self.replica_id
            progress.current_step = self.step_counter
            progress.step_progress = (self.step_counter % 100)
            progress.status = "training"
            progress.last_active_step = self.step_counter
            
            # Publish training progress
            await nc.publish(torchLoomConstants.subjects.TRAINING_PROGRESS, env2.SerializeToString())
            
            await nc.close()
            logger.debug(f"Reported status for {self.replica_id}: step {self.step_counter}")
            
        except Exception as e:
            logger.warning(f"Failed to report status: {e}")


async def run_weaver_with_ui():
    """Run the weaver with UI support."""
    logger.info("🌟 Starting Weaver with UI integration")
    
    try:
        # Create weaver with UI enabled
        weaver = Weaver(enable_ui=True, ui_host="0.0.0.0", ui_port=8080)
        await weaver.initialize()
        
        logger.info("✅ Weaver initialized successfully")
        logger.info("🌐 UI server will be available at: http://localhost:8080")
        logger.info("🔗 WebSocket endpoint: ws://localhost:8080/ws")
        
        # Start all weaver services
        async with asyncio.TaskGroup() as tg:
            # NATS subscriptions
            tg.create_task(weaver.subscribe_js(
                torchLoomConstants.weaver_stream.STREAM,
                torchLoomConstants.weaver_stream.subjects.DR_SUBJECT,
                torchLoomConstants.weaver_stream.CONSUMER,
                weaver.message_handler
            ))
            tg.create_task(weaver.subscribe_nc(
                subject=torchLoomConstants.subjects.CONFIG_INFO,
                message_handler=weaver.message_handler
            ))
            
            # UI-related subscriptions
            tg.create_task(weaver.subscribe_nc(
                subject=torchLoomConstants.subjects.GPU_STATUS,
                message_handler=weaver.message_handler
            ))
            tg.create_task(weaver.subscribe_nc(
                subject=torchLoomConstants.subjects.TRAINING_PROGRESS,
                message_handler=weaver.message_handler
            ))
            tg.create_task(weaver.subscribe_nc(
                subject=torchLoomConstants.subjects.UI_COMMAND,
                message_handler=weaver.message_handler
            ))
            
            # UI WebSocket server
            tg.create_task(weaver.start_ui_server())
            
            # Demo simulation
            tg.create_task(weaver.start_demo_simulation())
            
            logger.info("🚀 All weaver services started successfully")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("⏹️  Weaver stopped by user")
    except Exception as e:
        logger.exception(f"❌ Error in weaver: {e}")
        raise
    finally:
        await weaver.stop()


def run_enhanced_weavelet(replica_id: str = "demo_replica_1", vocab_size: int = 10):
    """Run an enhanced weavelet with status reporting."""
    logger.info(f"🔧 Starting enhanced weavelet: {replica_id}")
    
    try:
        # Create dataset and trainer
        dataset = RandomTextDataset(vocab_size=vocab_size, num_samples=100)
        dataloader = DataLoader(dataset, batch_size=4)
        
        # Create enhanced trainer
        trainer_model = EnhancedDemoTrainer(vocab_size=vocab_size, replica_id=replica_id)
        
        logger.info(f"✅ Enhanced trainer created: {replica_id}")
        logger.info(f"🔧 Initial config: optimizer={trainer_model.optimizer_type}, "
                   f"lr={trainer_model.learning_rate}, batch={trainer_model.batch_size}")
        
        # Create Lightning trainer
        lightning_trainer = L.Trainer(
            max_epochs=1,
            max_steps=50,  # Limit steps for demo
            enable_progress_bar=True,
            logger=False,
            enable_checkpointing=False
        )
        
        logger.info(f"🚀 Starting training for {replica_id}")
        lightning_trainer.fit(model=trainer_model, train_dataloaders=dataloader)
        
        logger.info(f"✅ Training completed for {replica_id}")
        
    except Exception as e:
        logger.exception(f"❌ Error in weavelet {replica_id}: {e}")


def run_ui_instructions():
    """Print instructions for running the UI."""
    print("\n" + "="*80)
    print("🌐 TORCHLOOM UI INSTRUCTIONS")
    print("="*80)
    print("\n📋 To start the Vue.js UI:")
    print("   1. Open a new terminal")
    print("   2. cd torchLoom-ui")
    print("   3. npm install  (if not already done)")
    print("   4. npm run dev")
    print("   5. Open http://localhost:5173 in your browser")
    print("\n✨ The UI will automatically connect to the backend via WebSocket")
    print("🔄 You should see real-time updates of GPU status and training progress")
    print("🎮 Try deactivating GPUs and reactivating replica groups in the UI")
    print("\n" + "="*80 + "\n")


async def main():
    """Main demo function."""
    print("\n🌟 TORCHLOOM INTEGRATED SYSTEM DEMO")
    print("="*50)
    print("\nThis demo shows the complete integrated torchLoom system:")
    print("✅ Enhanced Weaver with UI support")
    print("✅ WebSocket server for real-time updates") 
    print("✅ Enhanced Weavelets with status reporting")
    print("✅ Vue.js UI for monitoring and control")
    print("✅ NATS messaging coordination")
    
    # Show UI instructions
    run_ui_instructions()
    
    # Ask user if they want to continue, or auto-start if -y is provided
    auto_start = '-y' in sys.argv or '--yes' in sys.argv

    if auto_start:
        print("⚡ Auto-starting backend services due to -y flag.")
        # Proceed without prompt
    else:
        try:
            response = input("⚡ Start the backend services? (y/n): ").lower().strip()
            if response not in ['y', 'yes']:
                print("Demo cancelled.")
                return
        except KeyboardInterrupt:
            print("\nDemo cancelled.")
            return
        print("\nDemo cancelled.")
        return
    
    print("\n🚀 Starting integrated demo...")
    
    try:
        # Set multiprocessing start method
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        
        # Start weavelet processes
        weavelet_processes = []
        
        for i in range(1, 4):  # Start 3 weavelets
            replica_id = f"demo_replica_{i}"
            process = mp.Process(
                target=run_enhanced_weavelet,
                args=(replica_id, 10),
                name=f"weavelet-{replica_id}"
            )
            process.start()
            weavelet_processes.append(process)
            logger.info(f"🔧 Started weavelet process: {replica_id}")
            
            # Small delay between starting weavelets
            time.sleep(1)
        
        # Start weaver with UI (this blocks)
        await run_weaver_with_ui()
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Demo interrupted by user")
    except Exception as e:
        logger.exception(f"❌ Demo failed: {e}")
    finally:
        # Clean up weavelet processes
        logger.info("🧹 Cleaning up weavelet processes...")
        for process in weavelet_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
        
        logger.info("✅ Demo cleanup completed")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Suppress some verbose logs
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")
        sys.exit(1) 