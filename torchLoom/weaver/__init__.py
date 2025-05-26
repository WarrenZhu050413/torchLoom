"""
torchLoom Weaver package.

This package contains the refactored Weaver implementation with separated concerns:
- handlers: Message handling for incoming messages to the weaver
- publishers: Message publishing from the weaver to other components  
- subscription: NATS subscription management
- core: Main Weaver class
"""

# Core infrastructure
from torchLoom.common.subscription import SubscriptionManager

# Message handlers (messages TO the weaver)
from .handlers import (  # Consolidated handlers and utility classes
    DeviceReplicaMapper,
    ExternalHandler,
    MessageHandler,
    ThreadletHandler,
    UIHandler,
)

# Publishers (messages FROM the weaver)
from .publishers import (  # Weaver -> UI publishers; Weaver -> Threadlet publishers
    Publisher,
    ThreadletCommandPublisher,
    UIUpdatePublisher,
)
from .weaver import Weaver

__all__ = [
    # Core
    "Weaver",
    # Consolidated message handlers (TO weaver)
    "MessageHandler",
    "ThreadletHandler",
    "ExternalHandler",
    "UIHandler",
    "DeviceReplicaMapper",
    # Publishers (FROM weaver)
    "Publisher",
    "UIUpdatePublisher",
    "ThreadletCommandPublisher",
    # Infrastructure
    "SubscriptionManager",
]
