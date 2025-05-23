"""
torchLoom Weaver package.

This package contains the refactored Weaver implementation with separated concerns:
- handlers: Message handling logic
- subscription: NATS subscription management
- core: Main Weaver class
"""

from .handlers import (
    MessageHandler,
    DeviceRegistrationHandler,
    FailureHandler,
    ConfigurationHandler,
    DeviceReplicaMapper,
)
from .subscription import (
    StreamManager,
    SubscriptionManager,
    ConnectionManager,
)
from .core import Weaver

__all__ = [
    "Weaver",
    "MessageHandler",
    "DeviceRegistrationHandler", 
    "FailureHandler",
    "ConfigurationHandler",
    "DeviceReplicaMapper",
    "StreamManager",
    "SubscriptionManager",
    "ConnectionManager",
] 