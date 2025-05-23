"""
Lightning integration module for torchLoom Weavelet.

This module provides enhanced Lightning classes that automatically integrate
weavelet functionality for dynamic configuration management.
"""

from typing import Any, Dict, Optional

import lightning as L

from torchLoom.weavelet import Weavelet


class WeaveletLightningModule(L.LightningModule):
    """Enhanced Lightning module with built-in weavelet support.
    
    This base class automatically handles:
    - Weavelet process lifecycle (start/stop)
    - Configuration checking during training
    - Status publishing to weaver
    - Handler registration via decorators
    
    Usage:
        class MyTrainer(WeaveletLightningModule):
            def __init__(self, vocab_size: int):
                super().__init__(replica_id="my_trainer")
                # Your initialization here
                
            @weavelet_handler("optimizer_type")
            def update_optimizer(self, new_type: str):
                # Handler called automatically when config changes
                pass
    """
    
    def __init__(self, replica_id: str, torchLoom_addr: Optional[str] = None, **kwargs):
        """Initialize the Lightning module with weavelet integration.
        
        Args:
            replica_id: Unique identifier for this training replica
            torchLoom_addr: NATS server address (uses default if not provided)
            **kwargs: Additional arguments passed to parent LightningModule
        """
        super().__init__(**kwargs)
        
        # Initialize weavelet
        self.weavelet = Weavelet(
            replica_id=replica_id,
            torchLoom_addr=torchLoom_addr or "nats://localhost:4222"
        )
        
        # Start weavelet process
        self.weavelet.start()
        
        # Track training state for status publishing
        self._current_epoch = 0
        self._current_batch_idx = 0
        
    def weavelet_handler(self, config_key: str, expected_type=None):
        """Decorator for registering weavelet configuration handlers.
        
        This is a convenience method that delegates to the weavelet's handler decorator.
        
        Args:
            config_key: Configuration parameter name
            expected_type: Expected type for the parameter value
            
        Usage:
            @self.weavelet_handler("optimizer_type")
            def update_optimizer(self, new_type: str):
                pass
        """
        return self.weavelet.handler(config_key, expected_type)
    
    def training_step(self, batch, batch_idx):
        """Enhanced training step with automatic weavelet integration.
        
        This method automatically:
        - Checks for configuration updates
        - Publishes training status
        - Calls the user's training_step implementation
        """
        # Store current batch index for status publishing
        self._current_batch_idx = batch_idx
        
        # Check for configuration updates before training step
        self.weavelet.check_and_apply_updates()
        
        # Call the actual training step (to be implemented by subclass)
        result = self._user_training_step(batch, batch_idx)
        
        # Publish training status after successful training step
        self._publish_training_status(batch_idx, result)
        
        return result
    
    def _user_training_step(self, batch, batch_idx):
        """Override this method to implement your training logic.
        
        This is called automatically by the enhanced training_step method.
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            "Subclasses must implement _user_training_step() method. "
            "This replaces the standard training_step() method."
        )
    
    def _publish_training_status(self, batch_idx: int, result: Any) -> None:
        """Publish training status to the weaver.
        
        Args:
            batch_idx: Current batch index
            result: Result from training step (typically contains loss)
        """
        try:
            # Extract loss from result
            loss_value = None
            if isinstance(result, dict) and "loss" in result:
                loss_value = float(result["loss"])
            elif hasattr(result, "item"):  # Tensor
                loss_value = float(result.item())
            elif isinstance(result, (int, float)):
                loss_value = float(result)
            
            # Collect status information
            status = {
                "epoch": self._current_epoch,
                "batch_idx": batch_idx,
                "replica_id": self.weavelet._replica_id,
            }
            
            if loss_value is not None:
                status["loss"] = loss_value
            
            # Add any additional status from user implementation
            user_status = self._collect_additional_status()
            if user_status:
                status.update(user_status)
            
            # Publish to weavelet
            self.weavelet.publish_training_status(status)
            
        except Exception as e:
            print(f"Warning: Failed to publish training status: {e}")
    
    def _collect_additional_status(self) -> Optional[Dict[str, Any]]:
        """Override this method to add custom status information.
        
        Returns:
            Dictionary of additional status information to publish
        """
        return None
    
    def on_train_epoch_start(self) -> None:
        """Called at the start of each training epoch."""
        self._current_epoch = self.current_epoch
        
        # Check for configuration updates at epoch start
        self.weavelet.check_and_apply_updates()
        
        # Call parent implementation
        super().on_train_epoch_start()
    
    def on_train_end(self) -> None:
        """Called at the end of training to clean up weavelet."""
        try:
            self.weavelet.stop()
            print("Weavelet process stopped successfully")
        except Exception as e:
            print(f"Warning: Error stopping weavelet: {e}")
        
        # Call parent implementation
        super().on_train_end()
    
    def __del__(self):
        """Ensure weavelet is stopped when module is destroyed."""
        if hasattr(self, 'weavelet'):
            try:
                self.weavelet.stop()
            except:
                pass  # Ignore errors during cleanup


def weavelet_handler(config_key: str, expected_type=None):
    """Global decorator function for weavelet handlers.
    
    This can be used as a standalone decorator when the weavelet instance
    is not yet available during class definition.
    
    Args:
        config_key: Configuration parameter name
        expected_type: Expected type for the parameter value
        
    Usage:
        class MyTrainer(WeaveletLightningModule):
            @weavelet_handler("optimizer_type")
            def update_optimizer(self, new_type: str):
                pass
    """
    def decorator(func):
        # Store handler metadata on the function
        func._weavelet_config_key = config_key
        func._weavelet_expected_type = expected_type
        return func
    
    return decorator


class AutoWeaveletMixin:
    """Mixin class that automatically registers handlers decorated with @weavelet_handler.
    
    This mixin scans for methods decorated with @weavelet_handler and automatically
    registers them with the weavelet instance.
    """
    
    def _register_weavelet_handlers(self):
        """Scan and register all methods decorated with @weavelet_handler."""
        if not hasattr(self, 'weavelet'):
            return
        
        # Scan all methods for weavelet handler decorations
        for name in dir(self):
            # Skip special methods and properties that might cause issues during init
            if name.startswith('_') or name in ['trainer', 'optimizers', 'device']:
                continue
                
            try:
                method = getattr(self, name)
                # Only process callable methods with handler metadata
                if callable(method) and hasattr(method, '_weavelet_config_key'):
                    config_key = method._weavelet_config_key
                    expected_type = getattr(method, '_weavelet_expected_type', None)
                    
                    # Register the handler
                    self.weavelet.register_handler(config_key, method, expected_type)
                    print(f"Auto-registered weavelet handler: {name} for '{config_key}'")
            except Exception:
                # Skip any methods that cause issues during initialization
                continue


class EnhancedWeaveletLightningModule(WeaveletLightningModule, AutoWeaveletMixin):
    """Enhanced Lightning module with automatic handler registration.
    
    This class combines WeaveletLightningModule with AutoWeaveletMixin to provide
    the most convenient integration experience.
    
    Usage:
        class MyTrainer(EnhancedWeaveletLightningModule):
            def __init__(self, vocab_size: int):
                super().__init__(replica_id="my_trainer")
                # Handlers are automatically registered
                
            @weavelet_handler("optimizer_type")
            def update_optimizer(self, new_type: str):
                # This is automatically registered
                pass
    """
    
    def __init__(self, replica_id: str, torchLoom_addr: Optional[str] = None, **kwargs):
        super().__init__(replica_id=replica_id, torchLoom_addr=torchLoom_addr, **kwargs)
        
        # Automatically register decorated handlers
        self._register_weavelet_handlers() 