#!/usr/bin/env python3
"""
Example showing the new WeaveletWrapper pattern.
Users define normal Lightning modules and wrap them with weavelet functionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torchLoom.lightning import WeaveletWrapper, make_weavelet, weavelet_handler


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MyLightningTrainer(L.LightningModule):
    """
    Step 1: Define a NORMAL Lightning module.
    
    This is just a regular Lightning module - no special inheritance needed!
    Users can keep their existing Lightning modules unchanged.
    """
    
    def __init__(self):
        super().__init__()
        
        # Training configuration
        self.learning_rate = 0.001
        self.batch_size = 32
        self.optimizer_type = "adam"
        self.dropout_rate = 0.1
        
        # Create model
        self.model = SimpleModel()
        self.criterion = nn.CrossEntropyLoss()
        
        print("✅ Normal Lightning module created")
    
    def training_step(self, batch, batch_idx):
        """Standard Lightning training step - no changes needed."""
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        
        # Log metrics
        self.log('train_loss', loss)
        self.log('learning_rate', self.learning_rate)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Standard Lightning validation step."""
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        
        # Calculate accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer based on current settings."""
        if self.optimizer_type == "adam":
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "sgd":
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    # ==========================================
    # WEAVELET HANDLERS - Just add these to existing Lightning modules!
    # ==========================================
    
    @weavelet_handler("learning_rate", float)
    def update_learning_rate(self, new_lr: float):
        """Dynamically update learning rate during training."""
        print(f"📈 Handler called: Learning rate updated to {new_lr}")
        self.learning_rate = new_lr
        
        # Update optimizer if it exists
        if hasattr(self, 'trainer') and self.trainer is not None:
            for optimizer in self.trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
    
    @weavelet_handler("batch_size", int)  
    def update_batch_size(self, new_batch_size: int):
        """Update batch size (takes effect on next epoch)."""
        print(f"📦 Handler called: Batch size updated to {new_batch_size}")
        self.batch_size = new_batch_size
    
    @weavelet_handler("optimizer_type", str)
    def change_optimizer(self, opt_type: str):
        """Switch optimizer type during training."""
        print(f"⚙️ Handler called: Optimizer type updated to {opt_type}")
        self.optimizer_type = opt_type.lower()
    
    @weavelet_handler("dropout_rate", float)
    def update_dropout(self, new_dropout: float):
        """Update dropout rate."""
        print(f"🎭 Handler called: Dropout rate updated to {new_dropout}")
        self.dropout_rate = new_dropout
        
        # Find and update dropout layers
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = new_dropout
    
    # ==========================================
    # OPTIONAL: Custom weavelet hooks (safe to define)
    # ==========================================
    
    def on_weavelet_train_epoch_start(self) -> None:
        """Custom logic at epoch start - this is SAFE to define."""
        print(f"🚀 User Hook: Starting epoch {self.current_epoch}")
        print(f"   Current learning rate: {self.learning_rate}")
    
    def on_weavelet_train_epoch_end(self) -> None:
        """Custom logic at epoch end - this is SAFE to define."""
        print(f"✅ User Hook: Completed epoch {self.current_epoch}")
        
        # Example: Custom checkpoint logic
        if self.current_epoch % 5 == 0:
            print(f"   📸 Custom checkpoint logic for epoch {self.current_epoch}")
    
    def on_weavelet_train_batch_start(self, batch, batch_idx):
        """Custom logic before each batch - this is SAFE to define."""
        if batch_idx % 100 == 0:
            print(f"⚡ User Hook: Processing batch {batch_idx} (lr={self.learning_rate})")
        return None  # Don't skip any batches
    
    def on_weavelet_train_batch_end(self, outputs, batch, batch_idx):
        """Custom logic after each batch - this is SAFE to define."""
        if batch_idx % 100 == 0:
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            print(f"🎯 User Hook: Batch {batch_idx} completed with loss: {loss:.4f}")
        return None


def demo_wrapper_pattern():
    """Demonstrate the wrapper pattern."""
    print("🎯 DEMONSTRATING WRAPPER PATTERN")
    print("="*50)
    
    # Step 1: Create a normal Lightning module
    print("\n📋 Step 1: Creating normal Lightning module...")
    lightning_trainer = MyLightningTrainer()
    print(f"   Type: {type(lightning_trainer)}")
    print(f"   Is Lightning module: {isinstance(lightning_trainer, L.LightningModule)}")
    
    # Step 2: Wrap it with weavelet functionality
    print("\n📋 Step 2: Wrapping with weavelet functionality...")
    weavelet_trainer = WeaveletWrapper(lightning_trainer, replica_id="demo_trainer")
    print(f"   Type: {type(weavelet_trainer)}")
    print(f"   Is Lightning module: {isinstance(weavelet_trainer, L.LightningModule)}")  # Should be True due to __getattr__
    
    # Step 3: Show that handlers were registered
    print("\n📋 Step 3: Checking registered handlers...")
    if hasattr(weavelet_trainer, 'weavelet') and hasattr(weavelet_trainer.weavelet, 'handler_registry'):
        handlers = weavelet_trainer.weavelet.handler_registry.list_handlers()
        print(f"   📋 Automatically registered handlers:")
        for config_key, expected_type in handlers.items():
            print(f"      • {config_key} -> {expected_type.__name__}")
    
    # Step 4: Show that we can access Lightning module attributes
    print("\n📋 Step 4: Testing attribute delegation...")
    print(f"   learning_rate: {weavelet_trainer.learning_rate}")
    print(f"   optimizer_type: {weavelet_trainer.optimizer_type}")
    print(f"   batch_size: {weavelet_trainer.batch_size}")
    
    # Step 5: Test the hooks
    print("\n📋 Step 5: Testing hook integration...")
    fake_batch = (torch.randn(4, 784), torch.randint(0, 10, (4,)))
    fake_outputs = torch.tensor(0.5)
    
    # Test epoch start (don't set current_epoch directly since it's read-only)
    print("   Calling on_train_epoch_start()...")
    try:
        weavelet_trainer._wrapped_on_train_epoch_start()
        print("   ✅ Epoch start hook successful")
    except Exception as e:
        print(f"   ❌ Epoch start hook failed: {e}")
    
    # Test batch start
    print("   Calling on_train_batch_start()...")
    try:
        weavelet_trainer._wrapped_on_train_batch_start(fake_batch, 0)
        print("   ✅ Batch start hook successful")
    except Exception as e:
        print(f"   ❌ Batch start hook failed: {e}")
    
    return weavelet_trainer


def demo_convenience_function():
    """Demonstrate the convenience function."""
    print("\n\n🎯 DEMONSTRATING CONVENIENCE FUNCTION")
    print("="*50)
    
    # One-liner to wrap any Lightning module
    print("\n📋 Using make_weavelet() convenience function...")
    lightning_trainer = MyLightningTrainer()
    weavelet_trainer = make_weavelet(lightning_trainer, replica_id="convenience_trainer")
    
    print(f"   ✅ Created with one line: make_weavelet(trainer, replica_id)")
    print(f"   Type: {type(weavelet_trainer)}")
    
    return weavelet_trainer


def main():
    """Demonstrate the new wrapper pattern."""
    print("🚀 WEAVELET WRAPPER PATTERN DEMO")
    print("This shows how to wrap ANY existing Lightning module with weavelet functionality")
    print("=" * 80)
    
    # Demo 1: Manual wrapper
    wrapper_trainer = demo_wrapper_pattern()
    
    # Demo 2: Convenience function
    convenience_trainer = demo_convenience_function()
    
    print("\n\n" + "=" * 80)
    print("🎉 SUMMARY - WHY THIS PATTERN IS BETTER")
    print("=" * 80)
    print("✅ No inheritance required - wrap ANY Lightning module")
    print("✅ Existing Lightning modules work unchanged")
    print("✅ Add weavelet functionality with one line")
    print("✅ Users can't accidentally break weavelet functionality")
    print("✅ Cleaner separation of concerns")
    print("✅ Works with any existing Lightning project")
    
    print("\n📚 USAGE PATTERNS:")
    print("   Pattern 1: WeaveletWrapper(my_trainer, replica_id='my_id')")
    print("   Pattern 2: make_weavelet(my_trainer, replica_id='my_id')")
    
    print("\n🔧 MIGRATION FROM OLD APPROACH:")
    print("   OLD: class MyTrainer(WeaveletLightningModule)")
    print("   NEW: class MyTrainer(L.LightningModule) + make_weavelet(trainer)")


if __name__ == "__main__":
    main() 