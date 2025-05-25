"""
torchLoom Weaver package.

This package contains the refactored Weaver implementation with separated concerns:
- handlers: Message handling for incoming messages to the weaver
- publishers: Message publishing from the weaver to other components  
- subscription: NATS subscription management
- core: Main Weaver class
"""

from .core import Weaver

# Message handlers (messages TO the weaver)
from .handlers import (  # Consolidated handlers and utility classes
    DeviceRegistrationHandler,
    DeviceReplicaMapper,
    DrainEventHandler,
    ExternalHandler,
    FailureHandler,
    deviceStatusHandler,
    HeartbeatHandler,
    MessageHandler,
    TrainingStatusHandler,
    UICommandHandler,
    UIHandler,
    ThreadletHandler,
)

# Publishers (messages FROM the weaver)
from .publishers import (  # Weaver -> UI publishers; Weaver -> Threadlet publishers
    Publisher,
    UIUpdatePublisher,
    ThreadletCommandPublisher,
)

# Core infrastructure
from .subscription import ConnectionManager, StreamManager, SubscriptionManager

__all__ = [
    # Core
    "Weaver",
    # Consolidated message handlers (TO weaver)
    "MessageHandler",
    "ThreadletHandler",
    "ExternalHandler",
    "UIHandler",
    "DeviceReplicaMapper",
    # Backward compatibility aliases
    "DeviceRegistrationHandler",
    "HeartbeatHandler",
    "TrainingStatusHandler",
    "deviceStatusHandler",
    "FailureHandler",
    "UICommandHandler",
    "DrainEventHandler",
    # Publishers (FROM weaver)
    "Publisher",
    "UIUpdatePublisher",
    "ThreadletCommandPublisher",
    # Infrastructure
    "StreamManager",
    "SubscriptionManager",
    "ConnectionManager",
]
