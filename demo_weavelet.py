#!/usr/bin/env python3
"""
Demonstration of the enhanced process-based Weavelet class for torchLoom.

This script shows the new decorator-based handler system, automatic integration,
and enhanced Lightning classes for seamless dynamic configuration management.
"""

import multiprocessing as mp
import time
from typing import Any, Dict

import lightning as L
import torch
from torch.utils.data import DataLoader

from torchLoom.lightning_integration import (
    EnhancedWeaveletLightningModule,
    weavelet_handler,
)
from torchLoom.weavelet import Weavelet
from train import RandomTextDataset


class DemoTrainer(EnhancedWeaveletLightningModule):
    """Demo trainer using the enhanced automatic integration."""

    def __init__(self, vocab_size: int):
        super().__init__(replica_id="demo_enhanced_trainer")

        # Simple demo model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(vocab_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, vocab_size),
        )

        # Configuration parameters
        self.optimizer_type = "SGD"
        self.learning_rate = 0.01
        self.batch_size = 4

    def _user_training_step(self, batch, batch_idx):
        """Training logic (replaces training_step)."""
        inputs, targets = batch

        # Simple demo computation
        outputs = self.model(inputs.float())
        loss = torch.nn.functional.mse_loss(outputs, targets.float())

        return loss

    def configure_optimizers(self):
        if self.optimizer_type == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def _collect_additional_status(self) -> Dict[str, Any]:
        """Add custom status information."""
        return {
            "optimizer_type": self.optimizer_type,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        }

    # Handlers using the global decorator (automatically registered)
    @weavelet_handler("optimizer_type")
    def update_optimizer(self, new_type: str):
        """Update optimizer type - called automatically when config changes."""
        print(f"🔧 Handler: Updating optimizer {self.optimizer_type} → {new_type}")
        self.optimizer_type = new_type

        # Update the actual optimizer
        new_opt = self.configure_optimizers()
        if self.trainer is not None:
            self.trainer.optimizers = [new_opt]

    @weavelet_handler("learning_rate")
    def update_learning_rate(self, new_lr: float):
        """Update learning rate - called automatically when config changes."""
        print(f"📈 Handler: Updating learning rate {self.learning_rate} → {new_lr}")
        self.learning_rate = new_lr

        # Update optimizer learning rate
        if hasattr(self, "optimizers") and self.optimizers:
            for optimizer in self.optimizers():
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr

    @weavelet_handler("batch_size")
    def update_batch_size(self, new_size: int):
        """Update batch size - called automatically when config changes."""
        print(f"📦 Handler: Updating batch size {self.batch_size} → {new_size}")
        self.batch_size = new_size
        # Note: In real usage, you'd need to recreate the dataloader


def demo_enhanced_weavelet():
    """Demonstrate the enhanced weavelet functionality."""
    print("🌟 torchLoom Enhanced Weavelet Demo")
    print("=" * 50)

    # Create dataset and model
    print("📊 Creating dataset and enhanced trainer...")
    dataset = RandomTextDataset(vocab_size=10, num_samples=20)  # Small for demo
    dataloader = DataLoader(dataset, batch_size=4)

    # Create enhanced trainer with automatic weavelet integration
    trainer_model = DemoTrainer(vocab_size=dataset.vocab_size)

    print(
        f"✅ Enhanced trainer created with replica_id: {trainer_model.weavelet._replica_id}"
    )
    print(f"🔧 Initial optimizer: {trainer_model.optimizer_type}")
    print(f"📈 Initial learning rate: {trainer_model.learning_rate}")
    print(f"📦 Initial batch size: {trainer_model.batch_size}")

    # Test manual configuration changes (simulating weaver updates)
    print("\n🧪 Testing Handler System:")
    print("-" * 30)

    # Test optimizer change
    print("1. Testing optimizer type change...")
    trainer_model.weavelet._dispatch_handlers({"optimizer_type": "Adam"})

    # Test learning rate change
    print("2. Testing learning rate change...")
    trainer_model.weavelet._dispatch_handlers({"learning_rate": 0.001})

    # Test batch size change
    print("3. Testing batch size change...")
    trainer_model.weavelet._dispatch_handlers({"batch_size": 8})

    # Test type validation
    print("4. Testing type validation...")
    try:
        trainer_model.weavelet._dispatch_handlers({"learning_rate": "invalid_float"})
    except Exception as e:
        print(f"   ✅ Type validation caught error: {e}")

    # Test multiple simultaneous changes
    print("5. Testing multiple simultaneous changes...")
    trainer_model.weavelet._dispatch_handlers(
        {"optimizer_type": "SGD", "learning_rate": 0.1, "batch_size": 2}
    )

    # Test Lightning training with automatic integration
    print("\n🚀 Testing Lightning Training with Automatic Integration:")
    print("-" * 55)

    lightning_trainer = L.Trainer(
        fast_dev_run=3,  # Very short run for demo
        enable_progress_bar=False,  # Cleaner output
        logger=False,  # Disable logging for demo
    )

    print(
        "   Starting training (config changes and status publishing are automatic)..."
    )
    try:
        lightning_trainer.fit(model=trainer_model, train_dataloaders=dataloader)
        print("   ✅ Training completed successfully!")
    except Exception as e:
        print(f"   ⚠️  Training completed with: {e}")

    print("\n🎉 Enhanced Demo completed!")
    print("\nNew features demonstrated:")
    print("  ✓ Decorator-based handler registration (@weavelet_handler)")
    print("  ✓ Automatic handler dispatch with type validation")
    print("  ✓ Enhanced Lightning integration (EnhancedWeaveletLightningModule)")
    print("  ✓ Automatic config checking and status publishing")
    print("  ✓ Multiple parameter handling in single update")
    print("  ✓ Type conversion and validation")


def demo_basic_handler_system():
    """Demonstrate the basic enhanced handler system."""
    print("\n🔥 Basic Handler System Demo")
    print("=" * 30)

    # Create a basic weavelet with handlers
    weavelet = Weavelet(replica_id="basic_demo")

    # Configuration state
    config_state = {"optimizer_type": "SGD", "learning_rate": 0.01, "enabled": True}

    # Register handlers using decorators
    @weavelet.handler("optimizer_type")
    def handle_optimizer_change(new_type: str):
        config_state["optimizer_type"] = new_type
        print(f"  🔧 Optimizer changed to: {new_type}")

    @weavelet.handler("learning_rate")
    def handle_lr_change(new_lr: float):
        config_state["learning_rate"] = new_lr
        print(f"  📈 Learning rate changed to: {new_lr}")

    @weavelet.handler("enabled")
    def handle_enabled_change(enabled: bool):
        config_state["enabled"] = enabled
        print(f"  🔀 Training enabled: {enabled}")

    print("✅ Handlers registered")
    print(f"📋 Initial state: {config_state}")

    # Test various configuration changes
    test_configs = [
        {"optimizer_type": "Adam"},
        {"learning_rate": 0.001},
        {"enabled": False},
        {"optimizer_type": "RMSprop", "learning_rate": 0.01, "enabled": True},
    ]

    for i, config in enumerate(test_configs, 1):
        print(f"\n{i}. Applying config: {config}")
        weavelet._dispatch_handlers(config)
        print(f"   📋 New state: {config_state}")

    print("\n✅ Basic handler system demo completed!")


if __name__ == "__main__":
    # Set multiprocessing start method for better compatibility
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Start method may already be set

    # Run demos
    demo_basic_handler_system()
    demo_enhanced_weavelet()

    print("\n🏆 All enhanced demos completed successfully!")
    print("\n🎯 Integration Summary:")
    print("   • Enhanced API: @weavelet.handler() decorators")
    print("   • Automatic dispatch: Handlers called automatically")
    print("   • Type safety: Runtime type validation and conversion")
    print("   • Lightning integration: EnhancedWeaveletLightningModule")
    print("   • Minimal code: Just inherit and add @weavelet_handler decorators")
