"""
Common publishers for torchLoom components.

This module contains shared publishing functionality that can be used by both
threadlet and weaver components to promote code reuse.
"""

import time
from abc import abstractmethod
from typing import Any, Dict, Optional

from torchLoom.common.constants import NatsConstants
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

logger = setup_logger(name="common_publishers")


from abc import ABC, abstractmethod

class BasePublisher(ABC):
    """Base publisher for components that publish events to NATS.
    Specific event publishing methods are expected to be implemented in child classes.
    """

    def __init__(self, nats_client=None, js_client=None):
        self.nats_client = nats_client
        self.js_client = js_client
        logger.info("BasePublisher base initialized.")

    @abstractmethod
    async def publish(self, message_type: str, **kwargs) -> None:
        """Generic publish method.
        Child classes should override this or implement specific publish methods
        that are called by their own version of this generic publish dispatcher.
        """
        pass