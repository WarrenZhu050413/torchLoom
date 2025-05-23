"""
torchLoom Weaver package.

This package contains the refactored Weaver implementation with separated concerns:
- handlers: Message handling for incoming messages to the weaver
- publishers: Message publishing from the weaver to other components  
- subscription: NATS subscription management
- core: Main Weaver class
"""

# Message handlers (messages TO the weaver)
from .handlers import (
    MessageHandler,
    # Weavelet -> Weaver handlers
    DeviceRegistrationHandler,
    HeartbeatHandler,
    TrainingStatusHandler,
    GPUStatusHandler,
    # External -> Weaver handlers
    FailureHandler,
    NetworkStatusHandler,
    # UI -> Weaver handlers
    UICommandHandler,
    ConfigurationHandler,
    # Utility classes
    DeviceReplicaMapper,
)

# Publishers (messages FROM the weaver)
from .publishers import (
    Publisher,
    # Weaver -> UI publishers
    UIUpdatePublisher,
    # Weaver -> Weavelet publishers
    WeaveletCommandPublisher,
    HeartbeatMonitor,
    # Demo utilities
    DemoDataSimulator,
    # Legacy compatibility aliases
    UIUpdateHandler,
    WeaveletCommandHandler,
)

# Core infrastructure
from .subscription import (
    StreamManager,
    SubscriptionManager,
    ConnectionManager,
)
from .core import Weaver

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
    "NetworkStatusHandler",
    "UICommandHandler",
    "ConfigurationHandler",
    "DeviceReplicaMapper",
    
    # Publishers (FROM weaver)
    "Publisher",
    "UIUpdatePublisher",
    "WeaveletCommandPublisher",
    "HeartbeatMonitor",
    "DemoDataSimulator",
    # Legacy compatibility
    "UIUpdateHandler",
    "WeaveletCommandHandler",
    
    # Infrastructure
    "StreamManager",
    "SubscriptionManager",
    "ConnectionManager",
] 