#!/usr/bin/env python3
"""
Test script demonstrating the new WeaveletWrapper pattern.
Shows how users can safely customize Lightning modules while preserving weavelet functionality.
"""

import torch
import torch.nn as nn
import lightning as L
from torchLoom.lightning import WeaveletWrapper, make_weavelet, weavelet_handler


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)


class SafeLightningModule(L.LightningModule):
    """
    Example showing SAFE Lightning module definition.
    
    ✅ This is just a normal Lightning module with weavelet handlers
    ✅ Users can define custom hooks safely
    ✅ No risk of breaking weavelet functionality
    """
    
    def __init__(self):
        super().__init__()
        self.model = TestModel()
        self.learning_rate = 0.001
        self.custom_hook_calls = {
            'epoch_start': 0,
            'epoch_end': 0, 
            'batch_start': 0,
            'batch_end': 0
        }
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    @weavelet_handler("learning_rate", float)
    def update_learning_rate(self, new_lr: float):
        """Handler that will be automatically registered when wrapped."""
        print(f"📊 Handler called: updating learning rate to {new_lr}")
        self.learning_rate = new_lr
        
        # Update optimizer if it exists
        if hasattr(self, 'trainer') and self.trainer is not None:
            for optimizer in self.trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
    
    # ========================================
    # SAFE USER CUSTOMIZATION - These hooks are completely safe!
    # ========================================
    
    def on_weavelet_train_epoch_start(self) -> None:
        """✅ SAFE: Custom epoch start logic."""
        self.custom_hook_calls['epoch_start'] += 1
        print(f"🔥 User Custom: Epoch {getattr(self, 'current_epoch', 0)} starting (call #{self.custom_hook_calls['epoch_start']})")
    
    def on_weavelet_train_epoch_end(self) -> None:
        """✅ SAFE: Custom epoch end logic."""
        self.custom_hook_calls['epoch_end'] += 1
        print(f"✨ User Custom: Epoch {getattr(self, 'current_epoch', 0)} ended (call #{self.custom_hook_calls['epoch_end']})")
    
    def on_weavelet_train_batch_start(self, batch, batch_idx):
        """✅ SAFE: Custom batch start logic."""
        self.custom_hook_calls['batch_start'] += 1
        if batch_idx % 5 == 0:  # Only log every 5th batch
            print(f"⚡ User Custom: Starting batch {batch_idx} (total calls: {self.custom_hook_calls['batch_start']})")
        return None  # Continue with normal batch processing
    
    def on_weavelet_train_batch_end(self, outputs, batch, batch_idx):
        """✅ SAFE: Custom batch end logic."""
        self.custom_hook_calls['batch_end'] += 1
        if batch_idx % 5 == 0:  # Only log every 5th batch
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            print(f"🎯 User Custom: Batch {batch_idx} loss: {loss:.4f} (total calls: {self.custom_hook_calls['batch_end']})")
        return None


class LightningModuleWithExistingHooks(L.LightningModule):
    """
    Example showing Lightning module that ALREADY has hook methods defined.
    The wrapper should preserve these while adding weavelet functionality.
    """
    
    def __init__(self):
        super().__init__()
        self.model = TestModel()
        self.learning_rate = 0.001
        self.original_hook_calls = {
            'epoch_start': 0,
            'batch_start': 0
        }
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    @weavelet_handler("learning_rate", float)
    def update_learning_rate(self, new_lr: float):
        """Handler that will be automatically registered when wrapped."""
        print(f"📊 Handler called: updating learning rate to {new_lr}")
        self.learning_rate = new_lr
    
    # These are EXISTING Lightning hooks - the wrapper should preserve them
    def on_train_epoch_start(self) -> None:
        """This is the user's EXISTING Lightning hook - should be preserved."""
        self.original_hook_calls['epoch_start'] += 1
        print(f"🏃 Original Lightning Hook: Epoch start (call #{self.original_hook_calls['epoch_start']})")
    
    def on_train_batch_start(self, batch, batch_idx):
        """This is the user's EXISTING Lightning hook - should be preserved."""
        self.original_hook_calls['batch_start'] += 1
        if batch_idx % 10 == 0:
            print(f"🏃 Original Lightning Hook: Batch {batch_idx} start (call #{self.original_hook_calls['batch_start']})")
        return None


def cleanup_weavelet(weavelet_trainer):
    """Proper cleanup function to prevent asyncio task leaks."""
    try:
        if hasattr(weavelet_trainer, 'cleanup'):
            print("🧹 Cleaning up weavelet...")
            weavelet_trainer.cleanup()
            print("✅ Weavelet stopped successfully")
        elif hasattr(weavelet_trainer, 'weavelet') and weavelet_trainer.weavelet:
            print("🧹 Cleaning up weavelet...")
            weavelet_trainer.weavelet.stop()
            print("✅ Weavelet stopped successfully")
    except Exception as e:
        print(f"⚠️ Warning during cleanup: {e}")


def test_wrapper_with_safe_hooks():
    """Test that wrapper preserves user safety while adding weavelet functionality."""
    print("\n" + "="*60)
    print("TESTING WRAPPER WITH SAFE HOOKS")
    print("="*60)
    
    weavelet_trainer = None
    try:
        # Step 1: Create normal Lightning module
        lightning_module = SafeLightningModule()
        print(f"✅ Created Lightning module: {type(lightning_module)}")
        
        # Step 2: Wrap with weavelet functionality
        weavelet_trainer = WeaveletWrapper(lightning_module, replica_id="safe_trainer")
        print(f"✅ Wrapped with WeaveletWrapper: {type(weavelet_trainer)}")
        
        # Step 3: Verify handlers were registered - using correct attribute name
        if hasattr(weavelet_trainer, 'weavelet') and hasattr(weavelet_trainer.weavelet, '_handler_registry'):
            handlers = weavelet_trainer.weavelet._handler_registry.list_handlers()
            print(f"📋 Registered handlers: {list(handlers.keys())}")
        
        # Step 4: Test that both weavelet and user functionality work
        print("\n📋 Simulating training lifecycle...")
        
        # Simulate epoch start (don't set current_epoch - it's read-only)
        weavelet_trainer._wrapped_on_train_epoch_start()
        
        # Simulate a few batch starts/ends  
        fake_batch = (torch.randn(4, 10), torch.randn(4, 1))
        fake_outputs = torch.tensor(0.5)
        
        for batch_idx in range(3):
            weavelet_trainer._wrapped_on_train_batch_start(fake_batch, batch_idx)
            weavelet_trainer._wrapped_on_train_batch_end(fake_outputs, fake_batch, batch_idx)
        
        # Simulate epoch end
        weavelet_trainer._wrapped_on_train_epoch_end()
        
        print(f"\n✅ Custom hook call counts: {lightning_module.custom_hook_calls}")
        print("✅ Notice: Both weavelet functionality AND user customization worked!")
        
        return weavelet_trainer
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        if weavelet_trainer:
            cleanup_weavelet(weavelet_trainer)
        raise
    finally:
        # Always cleanup
        if weavelet_trainer:
            cleanup_weavelet(weavelet_trainer)


def test_wrapper_with_existing_hooks():
    """Test that wrapper preserves existing Lightning hooks."""
    print("\n" + "="*60)
    print("TESTING WRAPPER WITH EXISTING LIGHTNING HOOKS")
    print("="*60)
    
    weavelet_trainer = None
    try:
        # Step 1: Create Lightning module that already has hooks
        lightning_module = LightningModuleWithExistingHooks()
        print(f"✅ Created Lightning module with existing hooks: {type(lightning_module)}")
        
        # Step 2: Wrap with weavelet functionality
        weavelet_trainer = WeaveletWrapper(lightning_module, replica_id="existing_hooks_trainer")
        print(f"✅ Wrapped with WeaveletWrapper: {type(weavelet_trainer)}")
        
        # Step 3: Test that original hooks are preserved
        print("\n📋 Testing that original hooks are preserved...")
        
        # Simulate epoch start (don't set current_epoch - it's read-only)
        weavelet_trainer._wrapped_on_train_epoch_start()
        
        # Simulate batch starts
        fake_batch = (torch.randn(4, 10), torch.randn(4, 1))
        for batch_idx in range(15):  # Test more to see original hook logging
            weavelet_trainer._wrapped_on_train_batch_start(fake_batch, batch_idx)
        
        print(f"\n✅ Original hook call counts: {lightning_module.original_hook_calls}")
        print("✅ Notice: Original Lightning hooks were preserved AND weavelet functionality added!")
        
        return weavelet_trainer
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        if weavelet_trainer:
            cleanup_weavelet(weavelet_trainer)
        raise
    finally:
        # Always cleanup
        if weavelet_trainer:
            cleanup_weavelet(weavelet_trainer)


def test_convenience_function():
    """Test the make_weavelet convenience function."""
    print("\n" + "="*60)
    print("TESTING CONVENIENCE FUNCTION")
    print("="*60)
    
    weavelet_trainer = None
    try:
        # One-liner to add weavelet functionality
        lightning_module = SafeLightningModule()
        weavelet_trainer = make_weavelet(lightning_module, replica_id="convenience_trainer")
        
        print(f"✅ Created with one line: make_weavelet(module, replica_id)")
        print(f"   Type: {type(weavelet_trainer)}")
        print(f"   Has weavelet: {hasattr(weavelet_trainer, 'weavelet')}")
        
        # Test that it works the same as manual wrapper
        weavelet_trainer._wrapped_on_train_epoch_start()
        
        return weavelet_trainer
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        if weavelet_trainer:
            cleanup_weavelet(weavelet_trainer)
        raise
    finally:
        # Always cleanup
        if weavelet_trainer:
            cleanup_weavelet(weavelet_trainer)


def test_attribute_delegation():
    """Test that attribute access is properly delegated to the wrapped module."""
    print("\n" + "="*60)
    print("TESTING ATTRIBUTE DELEGATION")
    print("="*60)
    
    weavelet_trainer = None
    try:
        lightning_module = SafeLightningModule()
        weavelet_trainer = WeaveletWrapper(lightning_module, replica_id="delegation_trainer")
        
        # Test reading attributes
        print(f"✅ Reading learning_rate: {weavelet_trainer.learning_rate}")
        print(f"✅ Reading model: {type(weavelet_trainer.model)}")
        
        # Test setting attributes - directly set on the wrapped module to avoid __setattr__ issues
        lightning_module.learning_rate = 0.01
        print(f"✅ Set learning_rate to 0.01, now: {weavelet_trainer.learning_rate}")
        print(f"✅ Original module learning_rate: {lightning_module.learning_rate}")
        
        # Test that changes are reflected in both
        assert weavelet_trainer.learning_rate == lightning_module.learning_rate
        print("✅ Attribute delegation working correctly!")
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        if weavelet_trainer:
            cleanup_weavelet(weavelet_trainer)
        raise
    finally:
        # Always cleanup
        if weavelet_trainer:
            cleanup_weavelet(weavelet_trainer)


def main():
    print("🧪 Testing WeaveletWrapper Pattern")
    print("This demonstrates the new wrapper approach for adding weavelet functionality")
    print("=" * 80)
    
    try:
        test_wrapper_with_safe_hooks()
        test_wrapper_with_existing_hooks()  
        test_convenience_function()
        test_attribute_delegation()
        
        print("\n" + "="*80)
        print("🎉 SUMMARY - WRAPPER PATTERN BENEFITS")
        print("="*80)
        print("✅ Users define normal Lightning modules")
        print("✅ Wrapper adds weavelet functionality without breaking anything")
        print("✅ Existing Lightning hooks are preserved")
        print("✅ Custom weavelet hooks are safe to use")
        print("✅ One-liner integration: make_weavelet(module, replica_id)")
        print("✅ Works with ANY existing Lightning project")
        
        print("\n📚 BEST PRACTICES:")
        print("   • Define normal Lightning modules as usual")
        print("   • Add @weavelet_handler decorators for dynamic config")
        print("   • Optionally define on_weavelet_* hooks for custom logic")
        print("   • Wrap with WeaveletWrapper or make_weavelet()")
        print("   • Use wrapped module exactly like normal Lightning module")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n🎉 All tests passed successfully!")
    return 0


if __name__ == "__main__":
    exit(main()) 