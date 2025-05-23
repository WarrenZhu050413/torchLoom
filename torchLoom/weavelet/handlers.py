"""
Handler registration and dispatch system for weavelet configuration management.
"""

import inspect
from typing import Any, Callable, Dict, Optional, Type

from .config import TypeConverter


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