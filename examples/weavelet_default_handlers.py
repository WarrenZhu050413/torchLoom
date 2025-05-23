#!/usr/bin/env python3
"""
Example demonstrating default handlers for torchLoom Weavelets.

This example shows how weavelets now come with comprehensive default handlers
for common configuration parameters, eliminating the need for manual handler setup.
"""

import asyncio
import time
from typing import Dict, Any

import pytorch_lightning as L
import torch
import torch.nn as nn

from torchLoom.lightning_wrapper import WeaveletWrapper, make_weavelet
from torchLoom.weavelet import Weavelet, weavelet_handler


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 32, output_size: int = 1):
        super().__init__()
        self.dropout_rate = 0.1
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class BasicTrainer(L.LightningModule):
    """Basic Lightning module that benefits from default handlers."""
    
    def __init__(self):
        super().__init__()
        
        # These attributes will be automatically updated by default handlers
        self.learning_rate = 0.001
        self.batch_size = 32
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.dropout_rate = 0.1
        self.training_enabled = True
        self.verbose = False
        
        # Model
        self.model = SimpleModel()
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        if not self.training_enabled:
            return None  # Skip training if disabled
            
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        if self.verbose:
            print(f"Batch {batch_idx}: loss = {loss.item():.4f}")
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )


class CustomTrainer(L.LightningModule):
    """Trainer with custom handlers alongside default ones."""
    
    def __init__(self):
        super().__init__()
        
        # Default handlers will manage these
        self.learning_rate = 0.001
        self.dropout_rate = 0.1
        
        # Custom parameter - needs custom handler
        self.custom_multiplier = 1.0
        
        self.model = SimpleModel()
        self.loss_fn = nn.MSELoss()
    
    @weavelet_handler("custom_multiplier", float)
    def update_custom_multiplier(self, new_value: float):
        """Custom handler for our custom parameter."""
        print(f"🎛️ Custom multiplier updated to: {new_value}")
        self.custom_multiplier = new_value
        
        # Custom logic: adjust model weights based on multiplier
        with torch.no_grad():
            for param in self.model.parameters():
                param.data *= new_value
    
    def forward(self, x):
        return self.model(x) * self.custom_multiplier
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def basic_usage_example():
    """Demonstrate zero-configuration usage with default handlers."""
    print("=== Basic Usage Example ===")
    
    # Create a basic trainer
    trainer = BasicTrainer()
    
    # Wrap with weavelet - default handlers are automatically enabled!
    weavelet_trainer = WeaveletWrapper(trainer, replica_id="basic_trainer")
    
    # List all available configuration handlers
    handlers = weavelet_trainer.list_config_handlers()
    print(f"\n✅ {len(handlers)} default handlers automatically registered:")
    for param, description in handlers.items():
        print(f"  📝 {param}: {description}")
    
    # Show registered handler types
    registered = weavelet_trainer.get_registered_handlers()
    print(f"\n📋 Handler types:")
    for param, param_type in registered.items():
        print(f"  {param}: {param_type.__name__}")
    
    # Simulate configuration updates (these would normally come from the weaver)
    print(f"\n🔄 Simulating configuration updates...")
    
    # Test learning rate update
    print(f"Original learning_rate: {trainer.learning_rate}")
    weavelet_trainer.weavelet._handler_registry.dispatch_handlers({"learning_rate": 0.01})
    print(f"Updated learning_rate: {trainer.learning_rate}")
    
    # Test batch size update  
    print(f"Original batch_size: {trainer.batch_size}")
    weavelet_trainer.weavelet._handler_registry.dispatch_handlers({"batch_size": 64})
    print(f"Updated batch_size: {trainer.batch_size}")
    
    # Test training control
    print(f"Original training_enabled: {trainer.training_enabled}")
    weavelet_trainer.weavelet._handler_registry.dispatch_handlers({"training_enabled": False})
    print(f"Updated training_enabled: {trainer.training_enabled}")
    
    # Cleanup
    weavelet_trainer.cleanup()
    print("✅ Basic example completed!")


def custom_handler_example():
    """Demonstrate combining default handlers with custom ones."""
    print("\n=== Custom Handler Example ===")
    
    # Create trainer with custom handlers
    trainer = CustomTrainer()
    
    # Wrap with weavelet
    weavelet_trainer = make_weavelet(trainer, replica_id="custom_trainer")
    
    # List all handlers (default + custom)
    handlers = weavelet_trainer.get_registered_handlers()
    print(f"✅ {len(handlers)} total handlers registered:")
    
    default_params = weavelet_trainer.list_config_handlers().keys()
    for param, param_type in handlers.items():
        if param in default_params:
            print(f"  📝 {param}: {param_type.__name__} (default)")
        else:
            print(f"  🎛️ {param}: {param_type.__name__} (custom)")
    
    # Test default handler
    print(f"\n🔄 Testing default handler - learning_rate:")
    print(f"Before: {trainer.learning_rate}")
    weavelet_trainer.weavelet._handler_registry.dispatch_handlers({"learning_rate": 0.005})
    print(f"After: {trainer.learning_rate}")
    
    # Test custom handler
    print(f"\n🔄 Testing custom handler - custom_multiplier:")
    print(f"Before: {trainer.custom_multiplier}")
    weavelet_trainer.weavelet._handler_registry.dispatch_handlers({"custom_multiplier": 2.0})
    print(f"After: {trainer.custom_multiplier}")
    
    # Cleanup
    weavelet_trainer.cleanup()
    print("✅ Custom handler example completed!")


def standalone_weavelet_example():
    """Demonstrate using Weavelet directly (without Lightning) with default handlers."""
    print("\n=== Standalone Weavelet Example ===")
    
    # Create a simple object to receive config updates
    class SimpleConfig:
        def __init__(self):
            self.learning_rate = 0.001
            self.batch_size = 32
            self.verbose = False
    
    config = SimpleConfig()
    
    # Create weavelet with default handlers
    weavelet = Weavelet(replica_id="standalone_example")
    weavelet.enable_default_handlers(config)
    
    # Start weavelet (normally you'd start it, but for demo we'll just use handlers directly)
    # weavelet.start()  # Commented out for demo
    
    # Show available handlers
    handlers = weavelet.get_supported_config_parameters()
    print(f"✅ {len(handlers)} default handlers available")
    
    # Test direct handler dispatch
    print(f"\n🔄 Testing configuration updates:")
    print(f"Before - learning_rate: {config.learning_rate}, batch_size: {config.batch_size}")
    
    # Dispatch updates
    weavelet._handler_registry.dispatch_handlers({
        "learning_rate": 0.01,
        "batch_size": 128,
        "verbose": True
    })
    
    print(f"After - learning_rate: {config.learning_rate}, batch_size: {config.batch_size}")
    print(f"Verbose mode: {config.verbose}")
    
    # weavelet.stop()  # Commented out for demo
    print("✅ Standalone example completed!")


def main():
    """Run all examples."""
    print("🎉 torchLoom Weavelet Default Handlers Demo")
    print("=" * 50)
    
    try:
        basic_usage_example()
        custom_handler_example() 
        standalone_weavelet_example()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print("🎉 torchLoom Weavelets now work with ZERO configuration!")
        print("📚 Default handlers cover all common training parameters")
        print("🎛️ Custom handlers can be added for specialized needs")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 