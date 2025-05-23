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
from .handlers import (  # Weavelet -> Weaver handlers; External -> Weaver handlers; UI -> Weaver handlers; Utility classes
    ConfigurationHandler,
    DeviceRegistrationHandler,
    DeviceReplicaMapper,
    DrainEventHandler,
    FailureHandler,
    GPUStatusHandler,
    HeartbeatHandler,
    MessageHandler,
    TrainingStatusHandler,
    UICommandHandler,
    WeaverCommandHandler,
)

# Publishers (messages FROM the weaver)
from .publishers import (  # Weaver -> UI publishers; Weaver -> Weavelet publishers; Demo utilities;
    DemoDataSimulator,
    HeartbeatMonitor,
    Publisher,
    UIUpdatePublisher,
    WeaveletCommandPublisher,
)

# Core infrastructure
from .subscription import ConnectionManager, StreamManager, SubscriptionManager

__all__ = [
    # Core
    "Weaver",
    # Message handlers (TO weaver)
    "MessageHandler",
    "DeviceRegistrationHandler",
    "HeartbeatHandler",
    "TrainingStatusHandler",
    "GPUStatusHandler",
    "FailureHandler",
    "UICommandHandler",
    "ConfigurationHandler",
    "DeviceReplicaMapper",
    # Publishers (FROM weaver)
    "Publisher",
    "UIUpdatePublisher",
    "WeaveletCommandPublisher",
    "HeartbeatMonitor",
    "DemoDataSimulator",
    # Infrastructure
    "StreamManager",
    "SubscriptionManager",
    "ConnectionManager",
]
