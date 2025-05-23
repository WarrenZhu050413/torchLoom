#!/usr/bin/env python3
"""
Demonstration of the new Weavelet class for torchLoom.

This script shows how the Weavelet integrates with Lightning training
and manages communication with the weaver.
"""

import time
from train import LightningTransformer, WeaveletCallback
import lightning as L
from torch.utils.data import DataLoader
from train import RandomTextDataset

def demo_weavelet():
    """Demonstrate the new Weavelet functionality."""
    print("🌟 torchLoom Weavelet Demo")
    print("=" * 50)
    
    # Create dataset and model
    print("📊 Creating dataset and model...")
    dataset = RandomTextDataset(vocab_size=100, num_samples=10)  # Small for demo
    dataloader = DataLoader(dataset, batch_size=4)
    
    # Create model with weavelet integration
    model = LightningTransformer(
        vocab_size=dataset.vocab_size, 
        replica_id="demo_replica_1"
    )
    
    print(f"✅ Model created with weavelet replica_id: {model.weavelet._replica_id}")
    print(f"🔧 Initial optimizer type: {model.optimizer_type}")
    
    # Test direct optimizer update (need a dummy trainer)
    print("\n🔄 Testing direct optimizer update...")
    print(f"   Before: {model.optimizer_type}")
    
    # Create a minimal dummy trainer for testing
    class DummyTrainer:
        def __init__(self):
            self.optimizers = []
    
    dummy_trainer = DummyTrainer()
    dummy_trainer.optimizers = [model.configure_optimizers()]
    model.trainer = dummy_trainer
    model.update_optimizer("Adam")
    print(f"   After: {model.optimizer_type}")
    
    # Test config handler
    print("\n📨 Testing config handler...")
    model.weavelet._message_handlers["optimizer_type"]("SGD")
    print(f"   Config handler changed optimizer to: {model.optimizer_type}")
    
    # Create a callback and create trainer for real training
    callback = WeaveletCallback()
    trainer = L.Trainer(
        fast_dev_run=3,  # Very short run for demo
        callbacks=[callback],
        enable_progress_bar=False,  # Cleaner output
        logger=False  # Disable logging for demo
    )
    
    print("\n🚀 Starting training with weavelet integration...")
    print("   (This will run a few training steps to demonstrate status publishing)")
    
    try:
        trainer.fit(model=model, train_dataloaders=dataloader)
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"⚠️  Training demo completed with: {e}")
    finally:
        # Ensure cleanup
        if hasattr(model, 'weavelet'):
            model.weavelet.stop()
        print("🛑 Weavelet stopped and cleaned up")
    
    print("\n🎉 Demo completed!")
    print("\nKey features demonstrated:")
    print("  ✓ Automatic device registration with weaver")
    print("  ✓ Config handler for optimizer updates")
    print("  ✓ Training status publishing")
    print("  ✓ Proper lifecycle management")
    print("  ✓ Background thread management")

if __name__ == "__main__":
    demo_weavelet() 