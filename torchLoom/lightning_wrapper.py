"""
Lightning integration module for torchLoom Threadlet.

This module provides a wrapper that can wrap any existing Lightning module
to add threadlet functionality without requiring inheritance changes.
"""

import time
from typing import Any, Dict, Optional, Union

import pytorch_lightning as L

from torchLoom.threadlet import Threadlet, threadlet_handler


class ThreadletWrapper(L.LightningModule):
    """Wrapper that adds threadlet functionality to any Lightning module.

    This wrapper uses composition over inheritance, allowing users to keep
    their existing Lightning modules unchanged while adding threadlet capabilities.

    Usage:
        # Step 1: Define your normal Lightning module
        class MyTrainer(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.learning_rate = 0.001
                self.model = MyModel()

            @threadlet_handler("learning_rate", float)
            def update_lr(self, new_lr: float):
                self.learning_rate = new_lr

            def training_step(self, batch, batch_idx):
                return self.compute_loss(batch)

        # Step 2: Wrap it with threadlet functionality
        trainer = MyTrainer()
        threadlet_trainer = ThreadletWrapper(trainer, replica_id="my_trainer")

        # Step 3: Use as normal Lightning module
        lightning_trainer = L.Trainer()
        lightning_trainer.fit(threadlet_trainer)
    """

    def __init__(
        self,
        lightning_module: L.LightningModule,
        replica_id: str,
        torchLoom_addr: Optional[str] = None,
    ):
        """Initialize the wrapper around a Lightning module.

        Args:
            lightning_module: The Lightning module to wrap
            replica_id: Unique identifier for this training replica
            torchLoom_addr: NATS server address (uses default if not provided)
        """
        # Set these first using object.__setattr__ to avoid PyTorch Module restrictions
        object.__setattr__(self, "_wrapped_module", lightning_module)
        object.__setattr__(self, "_replica_id", replica_id)

        super().__init__()  # Initialize as Lightning module

        # Initialize threadlet
        self.threadlet = Threadlet(
            replica_id=replica_id,
            torchLoom_addr=torchLoom_addr or "nats://localhost:4222",
        )

        # Enable default handlers with the wrapped module as target
        self.threadlet.enable_default_handlers(self._wrapped_module)

        # Start threadlet process
        self.threadlet.start()

        # Scan the wrapped module for handlers and register them
        self._register_threadlet_handlers()

        # Set up hook integration
        self._setup_hooks()

        print(f"ThreadletWrapper initialized for replica: {replica_id}")
        print(f"Default handlers enabled for common config parameters")

    def _register_threadlet_handlers(self):
        """Scan the wrapped module for @threadlet_handler decorated methods."""
        if self._wrapped_module is None:
            return

        skip_attrs = {
            # Lightning-specific attributes that may not be available during init
            "trainer",
            "optimizers",
            "optimizer",
            "device",
            "global_rank",
            "local_rank",
            "fabric",
            "logger",
            "loggers",
            "checkpoint_callback",
            "callbacks",
            "datamodule",
            "current_epoch",
            "global_step",
            "automatic_optimization",
            # PyTorch module attributes that could cause issues
            "training",
            "named_parameters",
            "named_modules",
            "named_children",
            "parameters",
            "modules",
            "children",
            "state_dict",
            "load_state_dict",
            # Common Python special attributes
            "__class__",
            "__dict__",
            "__doc__",
            "__module__",
            "__weakref__",
        }

        # Scan all methods in the wrapped module for threadlet handler decorations
        for name in dir(self._wrapped_module):
            # Skip private methods and known problematic attributes
            if name.startswith("_") or name in skip_attrs:
                continue

            try:
                if not hasattr(self._wrapped_module, name):
                    continue

                method = getattr(self._wrapped_module, name)

                # Only process callable methods with handler metadata
                if not callable(method):
                    continue

                if not hasattr(method, "_threadlet_config_key"):
                    continue

                config_key = method._threadlet_config_key
                expected_type = getattr(method, "_threadlet_expected_type", None)

                # Register the handler
                self.threadlet.register_handler(config_key, method, expected_type)
                print(f"Auto-registered threadlet handler: {name} for '{config_key}'")

            except (AttributeError, RuntimeError, TypeError) as e:
                print(f"Skipping attribute '{name}' during handler registration: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error processing attribute '{name}': {e}")
                continue

    def _setup_hooks(self):
        """Set up method interception for Lightning hooks."""
        if self._wrapped_module is None:
            return

        # Store original methods
        self._original_on_train_batch_start = getattr(
            self._wrapped_module, "on_train_batch_start", None
        )
        self._original_on_train_batch_end = getattr(
            self._wrapped_module, "on_train_batch_end", None
        )
        self._original_on_train_epoch_start = getattr(
            self._wrapped_module, "on_train_epoch_start", None
        )
        self._original_on_train_epoch_end = getattr(
            self._wrapped_module, "on_train_epoch_end", None
        )
        self._original_on_train_end = getattr(
            self._wrapped_module, "on_train_end", None
        )

        # Replace with our wrapped versions
        if hasattr(self._wrapped_module, "__dict__"):
            self._wrapped_module.on_train_batch_start = (
                self._wrapped_on_train_batch_start
            )
            self._wrapped_module.on_train_batch_end = self._wrapped_on_train_batch_end
            self._wrapped_module.on_train_epoch_start = (
                self._wrapped_on_train_epoch_start
            )
            self._wrapped_module.on_train_epoch_end = self._wrapped_on_train_epoch_end
            self._wrapped_module.on_train_end = self._wrapped_on_train_end

    def _wrapped_on_train_batch_start(self, batch, batch_idx):
        """Wrapped batch start - adds threadlet functionality."""
        # Always check for configuration updates first
        self.threadlet.check_and_apply_updates()

        # Call user's custom hook if they defined one
        user_result = None
        if self._wrapped_module and hasattr(
            self._wrapped_module, "on_threadlet_train_batch_start"
        ):
            user_result = self._wrapped_module.on_threadlet_train_batch_start(
                batch, batch_idx
            )

        # Call original method if it existed
        original_result = None
        if self._original_on_train_batch_start:
            original_result = self._original_on_train_batch_start(batch, batch_idx)

        return user_result if user_result is not None else original_result

    def _wrapped_on_train_batch_end(self, outputs, batch, batch_idx):
        """Wrapped batch end - adds threadlet functionality."""
        # Always publish training status
        self._publish_training_status(batch_idx, outputs)

        # Call user's custom hook if they defined one
        user_result = None
        if self._wrapped_module and hasattr(
            self._wrapped_module, "on_threadlet_train_batch_end"
        ):
            user_result = self._wrapped_module.on_threadlet_train_batch_end(
                outputs, batch, batch_idx
            )

        # Call original method if it existed
        original_result = None
        if self._original_on_train_batch_end:
            original_result = self._original_on_train_batch_end(
                outputs, batch, batch_idx
            )

        return user_result if user_result is not None else original_result

    def _wrapped_on_train_epoch_start(self):
        """Wrapped epoch start - adds threadlet functionality."""
        # Always check for configuration updates
        self.threadlet.check_and_apply_updates()

        # Call user's custom hook if they defined one
        if self._wrapped_module and hasattr(
            self._wrapped_module, "on_threadlet_train_epoch_start"
        ):
            self._wrapped_module.on_threadlet_train_epoch_start()

        # Call original method if it existed
        if self._original_on_train_epoch_start:
            self._original_on_train_epoch_start()

    def _wrapped_on_train_epoch_end(self):
        """Wrapped epoch end - adds threadlet functionality."""
        # Publish epoch-end status
        self._publish_epoch_status()

        # Call user's custom hook if they defined one
        if self._wrapped_module and hasattr(
            self._wrapped_module, "on_threadlet_train_epoch_end"
        ):
            self._wrapped_module.on_threadlet_train_epoch_end()

        # Call original method if it existed
        if self._original_on_train_epoch_end:
            self._original_on_train_epoch_end()

    def _wrapped_on_train_end(self):
        """Wrapped train end - cleanup threadlet."""
        try:
            self.threadlet.stop()
            print("Threadlet process stopped successfully")
        except Exception as e:
            print(f"Warning: Error stopping threadlet: {e}")

        # Call original method if it existed
        if self._original_on_train_end:
            self._original_on_train_end()

    def _publish_training_status(self, batch_idx: int, result: Any) -> None:
        """Publish training status to the weaver."""
        try:
            # Extract loss from result
            loss_value = None
            if isinstance(result, dict) and "loss" in result:
                loss_value = float(result["loss"])
            elif (
                not isinstance(result, dict)
                and hasattr(result, "item")
                and callable(getattr(result, "item", None))
            ):
                loss_value = float(result.item())
            elif isinstance(result, (int, float)):
                loss_value = float(result)

            # Collect status information
            status = {
                "epoch": (
                    getattr(self._wrapped_module, "current_epoch", 0)
                    if self._wrapped_module
                    else 0
                ),
                "batch_idx": batch_idx,
                "replica_id": self._replica_id,
            }

            if loss_value is not None:
                status["loss"] = loss_value

            # Add any additional status from user implementation
            if self._wrapped_module and hasattr(
                self._wrapped_module, "_collect_additional_status"
            ):
                user_status = self._wrapped_module._collect_additional_status()
                if user_status:
                    status.update(user_status)

            # Publish to threadlet
            self.threadlet.publish_training_status(status)

        except Exception as e:
            print(f"Warning: Failed to publish training status: {e}")

    def _publish_epoch_status(self) -> None:
        """Publish epoch-level training status."""
        if hasattr(self, "threadlet") and self.threadlet:
            status_data = {
                "type": "epoch_end",
                "epoch": (
                    getattr(self._wrapped_module, "current_epoch", 0)
                    if self._wrapped_module
                    else 0
                ),
                "global_step": (
                    getattr(self._wrapped_module, "global_step", 0)
                    if self._wrapped_module
                    else 0
                ),
                "trainer_id": self._replica_id,
                "timestamp": time.time(),
            }

            # Add any available metrics (safely check if trainer exists)
            try:
                if (
                    self._wrapped_module
                    and hasattr(self._wrapped_module, "trainer")
                    and self._wrapped_module.trainer is not None
                ):
                    if hasattr(self._wrapped_module.trainer, "logged_metrics"):
                        status_data["metrics"] = dict(
                            self._wrapped_module.trainer.logged_metrics
                        )
            except (RuntimeError, AttributeError):
                # Module not attached to trainer yet - that's fine for testing
                pass

            try:
                self.threadlet.publish_training_status(status_data)
            except Exception as e:
                print(f"Failed to publish epoch status: {e}")

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped Lightning module."""
        # Check if we have the wrapped module attribute to avoid recursion during init
        if (
            "_wrapped_module" in self.__dict__
            and self._wrapped_module
            and hasattr(self._wrapped_module, name)
        ):
            return getattr(self._wrapped_module, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name, value):
        """Handle attribute setting - delegate to wrapped module for Lightning attributes."""
        # These are our wrapper's own attributes - always set on wrapper
        if name in (
            "_wrapped_module",
            "_replica_id",
            "threadlet",
            "_original_on_train_batch_start",
            "_original_on_train_batch_end",
            "_original_on_train_epoch_start",
            "_original_on_train_epoch_end",
            "_original_on_train_end",
        ):
            super().__setattr__(name, value)
        # Lightning framework attributes that should stay on wrapper
        elif name in (
            "_device_type",
            "_logger",
            "_fabric",
            "_precision",
            "_accelerator",
            "prepare_data_per_node",
            "allow_zero_length_dataloader_with_multiple_devices",
            "automatic_optimization",
            "_forward_cache",
            "_running_stage",
            "_compile_config",
        ):
            super().__setattr__(name, value)
        # Lightning properties that can't be set (read-only) - set on wrapper
        elif name in (
            "current_epoch",
            "global_step",
            "trainer",
            "device",
            "global_rank",
            "local_rank",
        ):
            try:
                super().__setattr__(name, value)
            except (AttributeError, TypeError):
                # If we can't set it, just ignore - it's probably a read-only property
                pass
        # User-defined attributes - delegate to wrapped module if it exists
        elif (
            hasattr(self, "_wrapped_module")
            and self._wrapped_module
            and hasattr(self._wrapped_module, name)
        ):
            setattr(self._wrapped_module, name, value)
        # Everything else during initialization or new attributes - set on wrapper
        else:
            super().__setattr__(name, value)

    def __del__(self):
        """Ensure threadlet is stopped when wrapper is destroyed."""
        self.cleanup()

    def cleanup(self):
        """Explicit cleanup method to stop threadlet and clean up resources."""
        if hasattr(self, "threadlet") and self.threadlet:
            try:
                self.threadlet.stop()
            except Exception as e:
                print(f"Warning during threadlet cleanup: {e}")

        # Clear references to help with garbage collection
        if hasattr(self, "_wrapped_module"):
            self._wrapped_module = None

    def list_config_handlers(self) -> Dict[str, str]:
        """List all registered configuration handlers.
        
        Returns:
            Dictionary mapping config parameter names to their descriptions
        """
        if hasattr(self, "threadlet") and self.threadlet:
            return self.threadlet.get_supported_config_parameters()
        return {}

    def get_registered_handlers(self) -> Dict[str, Any]:
        """Get all currently registered handlers.
        
        Returns:
            Dictionary mapping config keys to their expected types
        """
        if hasattr(self, "threadlet") and self.threadlet:
            return self.threadlet.get_registered_handlers()
        return {}


def make_threadlet(
    lightning_module: L.LightningModule,
    replica_id: str,
    torchLoom_addr: Optional[str] = None,
) -> ThreadletWrapper:
    """Convenience function to wrap a Lightning module with threadlet functionality.

    Args:
        lightning_module: The Lightning module to wrap
        replica_id: Unique identifier for this training replica
        torchLoom_addr: NATS server address (uses default if not provided)

    Returns:
        ThreadletWrapper that can be used as a Lightning module

    Usage:
        trainer = MyLightningModule()
        threadlet_trainer = make_threadlet(trainer, replica_id="my_trainer")

        lightning_trainer = L.Trainer()
        lightning_trainer.fit(threadlet_trainer)
    """
    return ThreadletWrapper(lightning_module, replica_id, torchLoom_addr)


# Export for convenience
__all__ = ["ThreadletWrapper", "make_threadlet", "threadlet_handler"]
