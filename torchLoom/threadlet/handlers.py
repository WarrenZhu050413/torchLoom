"""
Handler registration and dispatch system for threadlet configuration management.
"""

import inspect
import logging
from typing import Any, Callable, Dict, Optional, Type

logger = logging.getLogger(__name__)


def threadlet_handler(config_key: str, expected_type: Optional[Type] = None):
    """Global decorator function for threadlet handlers.

    This can be used as a standalone decorator when the threadlet instance
    is not yet available during class definition.

    Args:
        config_key: Configuration parameter name
        expected_type: Expected type for the parameter value (ignored - no type checking)

    Usage:
        class MyTrainer(ThreadletLightningModule):
            @threadlet_handler("optimizer_type")
            def update_optimizer(self, new_type: str):
                pass
    """

    def decorator(func):
        # Store handler metadata on the function
        func._threadlet_config_key = config_key
        func._threadlet_expected_type = expected_type
        return func

    return decorator


class HandlerRegistry:
    """Centralized registry for configuration handlers."""

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._handler_types: Dict[str, Type] = (
            {}
        )  # Kept for backward compatibility but not used

    def register_handler(
        self, config_key: str, handler: Callable, expected_type: Optional[Type] = None
    ) -> None:
        """Register a handler for a specific configuration parameter.

        Args:
            config_key: The configuration parameter name (e.g., 'optimizer_type')
            handler: Function to call when this parameter changes
            expected_type: Expected type for the parameter value (ignored - no type checking)
        """
        # Store handler without any type checking or inference
        self._handlers[config_key] = handler
        # Store None to indicate no type checking
        self._handler_types[config_key] = None
        print(f"Registered handler for '{config_key}' (no type checking)")

    def has_handler(self, config_key: str) -> bool:
        """Check if a handler is registered for the given config key."""
        return config_key in self._handlers

    def get_handler(self, config_key: str) -> Optional[Callable]:
        """Get the handler for the given config key."""
        return self._handlers.get(config_key)

    def get_handler_type(self, config_key: str) -> Optional[Type]:
        """Get the expected type for the given config key."""
        return self._handler_types.get(config_key)

    def dispatch_handlers(self, config_updates: Dict[str, Any]) -> None:
        """Automatically dispatch handlers for configuration updates."""
        for config_key, value in config_updates.items():
            if config_key in self._handlers:
                try:
                    # Call the handler directly without type checking or conversion
                    handler = self._handlers[config_key]
                    print(f"Calling handler for '{config_key}' with value: {value}")
                    handler(value)

                except Exception as e:
                    print(f"Error in handler for '{config_key}': {e}")
            else:
                print(f"No handler registered for config key: {config_key}")

    def clear_handlers(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()
        self._handler_types.clear()

    def list_handlers(self) -> Dict[str, Type]:
        """List all registered handlers (no type information since type checking is disabled)."""
        # Return handler names with None types since we don't do type checking
        return {key: None for key in self._handlers.keys()}

    def register_default_handlers(self, target_object: Optional[Any] = None) -> None:
        """Register simple logging handlers for common configuration parameters.

        Args:
            target_object: The object that will receive configuration updates (e.g., Lightning module)
                          If None, only logging handlers are registered.
        """
        # Store reference to target object for handlers that need it
        self._target_object = target_object

        # Register simple logging handlers for common parameters
        common_params = [
            "learning_rate",
            "lr",
            "batch_size",
            "momentum",
            "weight_decay",
            "optimizer_type",
            "optimizer",
            "training_enabled",
            "pause_training",
            "resume_training",
            "dropout_rate",
            "dropout",
            "log_level",
            "logging_interval",
            "verbose",
            "gradient_clip_val",
            "accumulate_grad_batches",
        ]

        registered_count = 0
        for param in common_params:
            if not self.has_handler(param):
                handler = self._create_simple_handler(param)
                self.register_handler(param, handler)
                registered_count += 1

        logger.info(f"Registered {registered_count} simple logging handlers")

    def _create_simple_handler(self, config_key: str):
        """Create a simple handler for a specific config key."""

        def handler(value: Any) -> None:
            logger.info(f"🔄 Configuration '{config_key}' updated to: {value}")

            # Optionally set attribute on target object if it exists
            if self._target_object and hasattr(self._target_object, config_key):
                setattr(self._target_object, config_key, value)
                logger.debug(f"Set {config_key} attribute on target object")

        return handler
