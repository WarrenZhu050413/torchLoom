"""
torchLoom Weaver package.

This package contains the refactored Weaver implementation with separated concerns:
- handlers: Individual handler functions for incoming messages to the weaver
- publishers: Message publishing from the weaver to other components  
- subscription: NATS subscription management
- core: Main Weaver class
"""

from torchLoom.common.handlers import BaseHandler, HandlerRegistry

# Core infrastructure
from torchLoom.common.subscription import SubscriptionManager

# Individual handler functions (messages TO the weaver)
from . import handlers

# Publishers (messages FROM the weaver)
from .publishers import ThreadletCommandPublisher  # Weaver -> Threadlet publishers
from .status_tracker import StatusTracker
from .ui_interface import UINotificationManager
from .weaver import Weaver

__all__ = [
    "Weaver",
    "StatusTracker",
    "BaseHandler",
    "HandlerRegistry",
    "handlers",
    "ThreadletCommandPublisher",
    "UINotificationManager",
    "SubscriptionManager",
]
