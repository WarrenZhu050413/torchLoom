"""
torchLoom Weaver package.

This package contains the refactored Weaver implementation with separated concerns:
- inbound_handlers: Message handling for incoming messages to the weaver
- outbound_handlers: Message publishing from the weaver to other components  
- subscription: NATS subscription management
- core: Main Weaver class
"""

# Inbound handlers (messages TO the weaver)
from .inbound_handlers import (
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

# Outbound handlers (messages FROM the weaver)
from .outbound_handlers import (
    OutboundHandler,
    # Weaver -> UI handlers
    UIUpdateHandler,
    # Weaver -> Weavelet handlers
    WeaveletCommandHandler,
    HeartbeatMonitor,
    # Demo utilities
    DemoDataSimulator,
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
    
    # Inbound handlers (TO weaver)
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
    
    # Outbound handlers (FROM weaver)
    "OutboundHandler",
    "UIUpdateHandler",
    "WeaveletCommandHandler",
    "HeartbeatMonitor",
    "DemoDataSimulator",
    
    # Infrastructure
    "StreamManager",
    "SubscriptionManager",
    "ConnectionManager",
] 