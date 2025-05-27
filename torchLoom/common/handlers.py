"""
Unified handler registry system for torchLoom.

This module provides a comprehensive handler registration and dispatch system
that can be used across all components of torchLoom for consistent event handling.
"""

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type


class BaseHandler(ABC):
    """Abstract base class for all handlers in torchLoom."""

    @abstractmethod
    async def handle(self, *args, **kwargs) -> None:
        """Handle a specific type of message or event."""
        pass


class HandlerRegistry:
    """Unified registry for all types of handlers in torchLoom."""

    def __init__(self, name: str = "default"):
        self.name = name
        self._handlers: Dict[str, Callable] = {}
        self._handler_types: Dict[str, Optional[Type]] = {}
        self._handler_metadata: Dict[str, Dict[str, Any]] = {}
        self._event_type_mapping: Dict[str, str] = (
            {}
        )  # For message handler functionality
        self._logger = logging.getLogger(f"HandlerRegistry.{name}")

    def register_handler(
        self,
        key: str,
        handler: Callable,
        expected_type: Optional[Type] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a handler for a specific key.

        Args:
            key: The handler key (e.g., 'heartbeat', 'optimizer_type')
            handler: Function to call when this key is triggered
            expected_type: Expected type for the parameter value (optional)
            metadata: Additional metadata about the handler
        """
        self._handlers[key] = handler
        self._handler_types[key] = expected_type
        self._handler_metadata[key] = metadata or {}
        self._logger.info(f"Registered handler for '{key}' in registry '{self.name}'")

    def register_message_handler(
        self, handler_key: str, handler: BaseHandler, event_types: Optional[list] = None
    ) -> None:
        """Register a message handler with associated event types.

        Args:
            handler_key: Key for the handler (e.g., 'threadlet', 'ui', 'external')
            handler: Handler instance
            event_types: List of event types this handler processes
        """
        self.register_handler(handler_key, handler)

        if event_types:
            for event_type in event_types:
                self._event_type_mapping[event_type] = handler_key

    def register_default_handlers(self, target_object: Optional[Any] = None) -> None:
        """Register simple logging handlers for common configuration parameters.

        Args:
            target_object: The object that will receive configuration updates (e.g., Lightning module)
                          If None, only logging handlers are registered.
        """
        from torchLoom.common.constants import HandlerConstants

        registered_count = 0
        for param in HandlerConstants.COMMON_CONFIG_PARAMS:
            if not self.has_handler(param):
                handler_func = self._create_handler(param, target_object)
                self.register_handler(param, handler_func)
                registered_count += 1

        self._logger.info(f"Registered {registered_count} simple logging handlers")

    def _create_handler(self, config_key: str, target_object: Optional[Any] = None):
        """Create a simple handler for a specific config key."""

        def handler_func(value: Any) -> None:
            self._logger.info(f"🔄 Configuration '{config_key}' updated to: {value}")

            # Optionally set attribute on target object if it exists
            if target_object and hasattr(target_object, config_key):
                setattr(target_object, config_key, value)
                self._logger.debug(f"Set {config_key} attribute on target object")

        return handler_func

    def has_handler(self, key: str) -> bool:
        """Check if a handler is registered for the given key."""
        return key in self._handlers

    def get_handler(self, key: str) -> Optional[Callable]:
        """Get the handler for the given key."""
        return self._handlers.get(key)

    def get_handler_for_event_type(self, event_type: str) -> Optional[BaseHandler]:
        """Get the appropriate handler for a specific event type."""
        handler_key = self._event_type_mapping.get(event_type)
        if handler_key:
            return self.get_handler(handler_key)
        return None

    def get_handler_type(self, key: str) -> Optional[Type]:
        """Get the expected type for the given key."""
        return self._handler_types.get(key)

    def get_handler_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata for the given key."""
        return self._handler_metadata.get(key, {})

    def get_supported_events(self) -> Dict[str, str]:
        """Get mapping of event types to handler keys."""
        return self._event_type_mapping.copy()

    def dispatch_handler(self, key: str, *args, **kwargs) -> Any:
        """Dispatch a single handler for the given key."""
        if key not in self._handlers:
            self._logger.warning(f"No handler registered for key: {key}")
            return None

        try:
            handler = self._handlers[key]
            self._logger.debug(
                f"Calling handler for '{key}' with args: {args}, kwargs: {kwargs}"
            )

            # Handle both sync and async handlers
            if inspect.iscoroutinefunction(handler):
                return handler(*args, **kwargs)  # Return coroutine for await
            else:
                return handler(*args, **kwargs)

        except Exception as e:
            self._logger.exception(f"Error in handler for '{key}': {e}")
            raise

    def dispatch_handlers(self, updates: Dict[str, Any]) -> None:
        """Automatically dispatch handlers for multiple updates."""
        for key, value in updates.items():
            if key in self._handlers:
                try:
                    self.dispatch_handler(key, value)
                except Exception as e:
                    self._logger.exception(
                        f"Error dispatching handler for '{key}': {e}"
                    )
            else:
                self._logger.debug(f"No handler registered for key: {key}")

    def clear_handlers(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()
        self._handler_types.clear()
        self._handler_metadata.clear()
        self._event_type_mapping.clear()
        self._logger.info(f"Cleared all handlers in registry '{self.name}'")

    def list_handlers(self) -> Dict[str, Dict[str, Any]]:
        """List all registered handlers with their metadata."""
        return {
            key: {
                "handler": (
                    handler.__name__ if hasattr(handler, "__name__") else str(handler)
                ),
                "expected_type": self._handler_types.get(key),
                "metadata": self._handler_metadata.get(key, {}),
            }
            for key, handler in self._handlers.items()
        }


def create_weaver_message_registry() -> HandlerRegistry:
    """Create and configure the main message handler registry for Weaver."""
    from torchLoom.common.constants import HandlerConstants

    registry = HandlerRegistry("weaver_main")

    # Register event type mappings
    registry._event_type_mapping.update(
        {
            # Threadlet events
            **{event: "threadlet" for event in HandlerConstants.THREADLET_EVENTS},
            # External events
            **{event: "external" for event in HandlerConstants.EXTERNAL_EVENTS},
            # UI events
            **{event: "ui" for event in HandlerConstants.UI_EVENTS},
        }
    )

    return registry


# Aliases for backward compatibility (but everything uses the same HandlerRegistry now)
MessageHandlerRegistry = HandlerRegistry
ThreadletHandlerRegistry = HandlerRegistry
