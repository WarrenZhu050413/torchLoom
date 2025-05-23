"""
Handler registration and dispatch system for weavelet configuration management.
"""

import inspect
import logging
from typing import Any, Callable, Dict, Optional, Type

from .config import TypeConverter

logger = logging.getLogger(__name__)


def weavelet_handler(config_key: str, expected_type: Optional[Type] = None):
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


class HandlerRegistry:
    """Centralized registry for configuration handlers."""

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._handler_types: Dict[str, Type] = {}
        self._type_converter = TypeConverter()

    def register_handler(
        self, config_key: str, handler: Callable, expected_type: Optional[Type] = None
    ) -> None:
        """Register a handler for a specific configuration parameter.

        Args:
            config_key: The configuration parameter name (e.g., 'optimizer_type')
            handler: Function to call when this parameter changes
            expected_type: Expected type for the parameter value (inferred if not provided)
        """
        # Infer type from handler signature if not provided
        if expected_type is None:
            sig = inspect.signature(handler)
            params = list(sig.parameters.values())
            if len(params) >= 1:
                param = params[0]  # First parameter (after self if it's a method)
                if param.annotation != inspect.Parameter.empty:
                    expected_type = param.annotation
                else:
                    expected_type = str  # Default to string

        self._handlers[config_key] = handler
        self._handler_types[config_key] = expected_type or str
        print(f"Registered handler for '{config_key}' with type {expected_type}")

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
                    # Validate and convert the value
                    expected_type = self._handler_types[config_key]
                    converted_value = self._type_converter.validate_and_convert_value(
                        config_key, value, expected_type
                    )

                    # Call the handler
                    handler = self._handlers[config_key]
                    print(
                        f"Calling handler for '{config_key}' with value: {converted_value}"
                    )
                    handler(converted_value)

                except Exception as e:
                    print(f"Error in handler for '{config_key}': {e}")
            else:
                print(f"No handler registered for config key: {config_key}")

    def clear_handlers(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()
        self._handler_types.clear()

    def list_handlers(self) -> Dict[str, Type]:
        """List all registered handlers and their expected types."""
        return self._handler_types.copy()

    def register_default_handlers(self, target_object: Optional[Any] = None) -> None:
        """Register default handlers for common configuration parameters.

        Args:
            target_object: The object that will receive configuration updates (e.g., Lightning module)
                          If None, only logging handlers are registered.
        """
        # Register default handlers for common configuration parameters
        default_handlers = {
            # Training parameters
            "learning_rate": (self._default_learning_rate_handler, float),
            "lr": (self._default_learning_rate_handler, float),  # Common alias
            "batch_size": (self._default_batch_size_handler, int),
            "momentum": (self._default_momentum_handler, float),
            "weight_decay": (self._default_weight_decay_handler, float),
            # Optimizer parameters
            "optimizer_type": (self._default_optimizer_handler, str),
            "optimizer": (self._default_optimizer_handler, str),  # Alias
            # Training control
            "training_enabled": (self._default_training_control_handler, bool),
            "pause_training": (self._default_pause_handler, bool),
            "resume_training": (self._default_resume_handler, bool),
            # Model parameters
            "dropout_rate": (self._default_dropout_handler, float),
            "dropout": (self._default_dropout_handler, float),  # Alias
            # Logging and debugging
            "log_level": (self._default_log_level_handler, str),
            "logging_interval": (self._default_logging_interval_handler, int),
            "verbose": (self._default_verbose_handler, bool),
            # Advanced parameters
            "gradient_clip_val": (self._default_gradient_clip_handler, float),
            "accumulate_grad_batches": (self._default_grad_accumulation_handler, int),
        }

        # Store reference to target object for handlers that need it
        self._target_object = target_object

        # Register all default handlers
        registered_count = 0
        for config_key, (handler_func, expected_type) in default_handlers.items():
            # Only register if not already registered (allow user overrides)
            if not self.has_handler(config_key):
                self.register_handler(config_key, handler_func, expected_type)
                registered_count += 1

        logger.info(f"Registered {registered_count} default weavelet handlers")

    # Default handler implementations
    def _default_learning_rate_handler(self, new_lr: float) -> None:
        """Default handler for learning rate updates."""
        logger.info(f"🔄 Learning rate updated to: {new_lr}")

        if self._target_object:
            # Try to update learning rate on the target object
            if hasattr(self._target_object, "learning_rate"):
                setattr(self._target_object, "learning_rate", new_lr)
                logger.info(f"Updated learning_rate attribute to {new_lr}")
            elif hasattr(self._target_object, "lr"):
                setattr(self._target_object, "lr", new_lr)
                logger.info(f"Updated lr attribute to {new_lr}")

            # Try to update optimizer learning rate
            if hasattr(self._target_object, "trainer") and self._target_object.trainer:
                try:
                    for optimizer in self._target_object.trainer.optimizers:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = new_lr
                    logger.info(f"Updated optimizer learning rates to {new_lr}")
                except (AttributeError, TypeError):
                    pass  # Trainer not available or optimizers not set up yet

    def _default_batch_size_handler(self, new_batch_size: int) -> None:
        """Default handler for batch size updates."""
        logger.info(f"🔄 Batch size updated to: {new_batch_size}")

        if self._target_object and hasattr(self._target_object, "batch_size"):
            setattr(self._target_object, "batch_size", new_batch_size)
            logger.info(f"Updated batch_size attribute to {new_batch_size}")

        logger.warning("Note: Batch size changes may require dataloader recreation")

    def _default_momentum_handler(self, new_momentum: float) -> None:
        """Default handler for momentum updates."""
        logger.info(f"🔄 Momentum updated to: {new_momentum}")

        if self._target_object and hasattr(self._target_object, "momentum"):
            setattr(self._target_object, "momentum", new_momentum)

        # Try to update optimizer momentum
        if (
            self._target_object
            and hasattr(self._target_object, "trainer")
            and self._target_object.trainer
        ):
            try:
                for optimizer in self._target_object.trainer.optimizers:
                    for param_group in optimizer.param_groups:
                        if "momentum" in param_group:
                            param_group["momentum"] = new_momentum
                logger.info(f"Updated optimizer momentum to {new_momentum}")
            except (AttributeError, TypeError):
                pass

    def _default_weight_decay_handler(self, new_weight_decay: float) -> None:
        """Default handler for weight decay updates."""
        logger.info(f"🔄 Weight decay updated to: {new_weight_decay}")

        if self._target_object and hasattr(self._target_object, "weight_decay"):
            setattr(self._target_object, "weight_decay", new_weight_decay)

        # Try to update optimizer weight decay
        if (
            self._target_object
            and hasattr(self._target_object, "trainer")
            and self._target_object.trainer
        ):
            try:
                for optimizer in self._target_object.trainer.optimizers:
                    for param_group in optimizer.param_groups:
                        if "weight_decay" in param_group:
                            param_group["weight_decay"] = new_weight_decay
                logger.info(f"Updated optimizer weight decay to {new_weight_decay}")
            except (AttributeError, TypeError):
                pass

    def _default_optimizer_handler(self, new_optimizer: str) -> None:
        """Default handler for optimizer type changes."""
        logger.info(f"🔄 Optimizer type updated to: {new_optimizer}")

        if self._target_object and hasattr(self._target_object, "optimizer_type"):
            setattr(self._target_object, "optimizer_type", new_optimizer)

        logger.warning(
            "Note: Optimizer changes typically require model reinitialization"
        )

    def _default_training_control_handler(self, enabled: bool) -> None:
        """Default handler for training enable/disable."""
        status = "enabled" if enabled else "disabled"
        logger.info(f"🔄 Training {status}")

        if self._target_object and hasattr(self._target_object, "training_enabled"):
            setattr(self._target_object, "training_enabled", enabled)

    def _default_pause_handler(self, should_pause: bool) -> None:
        """Default handler for training pause."""
        if should_pause:
            logger.info("⏸️ Training paused")
            if self._target_object and hasattr(self._target_object, "training_paused"):
                setattr(self._target_object, "training_paused", True)

    def _default_resume_handler(self, should_resume: bool) -> None:
        """Default handler for training resume."""
        if should_resume:
            logger.info("▶️ Training resumed")
            if self._target_object and hasattr(self._target_object, "training_paused"):
                setattr(self._target_object, "training_paused", False)

    def _default_dropout_handler(self, new_dropout: float) -> None:
        """Default handler for dropout rate updates."""
        logger.info(f"🔄 Dropout rate updated to: {new_dropout}")

        if self._target_object and hasattr(self._target_object, "dropout_rate"):
            setattr(self._target_object, "dropout_rate", new_dropout)
        elif self._target_object and hasattr(self._target_object, "dropout"):
            setattr(self._target_object, "dropout", new_dropout)

    def _default_log_level_handler(self, new_level: str) -> None:
        """Default handler for log level changes."""
        logger.info(f"🔄 Log level updated to: {new_level}")

        # Update Python logging level
        try:
            numeric_level = getattr(logging, new_level.upper())
            logging.getLogger().setLevel(numeric_level)
            logger.info(f"Python logging level set to {new_level.upper()}")
        except AttributeError:
            logger.warning(f"Invalid log level: {new_level}")

    def _default_logging_interval_handler(self, new_interval: int) -> None:
        """Default handler for logging interval updates."""
        logger.info(f"🔄 Logging interval updated to: {new_interval}")

        if self._target_object and hasattr(self._target_object, "logging_interval"):
            setattr(self._target_object, "logging_interval", new_interval)

    def _default_verbose_handler(self, verbose: bool) -> None:
        """Default handler for verbose mode."""
        status = "enabled" if verbose else "disabled"
        logger.info(f"🔄 Verbose mode {status}")

        if self._target_object and hasattr(self._target_object, "verbose"):
            setattr(self._target_object, "verbose", verbose)

    def _default_gradient_clip_handler(self, clip_val: float) -> None:
        """Default handler for gradient clipping."""
        logger.info(f"🔄 Gradient clipping updated to: {clip_val}")

        if self._target_object and hasattr(self._target_object, "gradient_clip_val"):
            setattr(self._target_object, "gradient_clip_val", clip_val)

    def _default_grad_accumulation_handler(self, accumulate_batches: int) -> None:
        """Default handler for gradient accumulation."""
        logger.info(f"🔄 Gradient accumulation updated to: {accumulate_batches}")

        if self._target_object and hasattr(
            self._target_object, "accumulate_grad_batches"
        ):
            setattr(self._target_object, "accumulate_grad_batches", accumulate_batches)
