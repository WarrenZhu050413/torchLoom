"""
Publishers for the torchLoom Weaver.

This module contains publishers for sending messages FROM the weaver to other components:
- UI publishers: Publish updates and responses to the UI
- Threadlet publishers: Publish commands and notifications to threadlets
- Demo utilities: Simulate data for demonstration purposes
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Set

from torchLoom.common.constants import torchLoomConstants
from torchLoom.common.publishers import BasePublisher, EventPublisher, UIStatusPublisher
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

logger = setup_logger(name="publishers")


# Re-export the base publisher for compatibility
Publisher = BasePublisher


# ===========================================
# UI PUBLISHERS (Weaver -> UI)
# ===========================================


class UIUpdatePublisher(UIStatusPublisher):
    """Publisher for sending consolidated UI updates FROM the weaver TO the UI.
    
    This now inherits from the common UIStatusPublisher for maximum code reuse.
    """

    async def publish_ui_update(self) -> Optional[EventEnvelope]:
        """Constructs a UIStatusUpdate message. Does not publish to NATS anymore."""
        try:
            envelope = await self.create_ui_status_update()
            if envelope:
                logger.debug(
                    "[WEAVER->UI] Constructed UIStatusUpdate envelope. Publishing to NATS is removed."
                )
            return envelope

        except Exception as e:
            logger.exception(
                f"[WEAVER->UI] Failed to construct UIStatusUpdate envelope: {e}"
            )
            return None


# ===========================================
# WEAVELET PUBLISHERS (Weaver -> Training Processes)
# ===========================================


class ThreadletCommandPublisher(BasePublisher):
    """Publisher for sending commands FROM the weaver TO threadlets/training processes.

    This now uses the common EventPublisher for maximum code reuse.
    """

    def __init__(self, nats_client=None, threadlet_handler=None):
        self.nats_client = nats_client
        self.threadlet_handler = (
            threadlet_handler  # Reference to threadlet handler for heartbeat monitoring
        )
        
        # Initialize the common event publisher
        self._event_publisher = EventPublisher(
            nats_client=nats_client,
            js_client=nats_client.jetstream() if nats_client else None,
        )

    async def publish_weaver_command(
        self,
        command_type: str,
        target_replica_id: str,
        params: Optional[Dict[str, str]] = None,
    ) -> None:
        """Publish a weaver command to training processes using the common publisher."""
        try:
            await self._event_publisher.publish_weaver_command(
                command_type=command_type,
                target_replica_id=target_replica_id,
                params=params,
            )

            logger.info(
                f"[WEAVER->WEAVELET] Published command: {command_type} to {target_replica_id}"
            )

        except Exception as e:
            logger.exception(
                f"[WEAVER->WEAVELET] Failed to publish weaver command: {e}"
            )

    async def publish_config_update(self, config_params: Dict[str, str]) -> None:
        """Publish configuration updates to all training processes via UI_COMMANDS."""
        try:
            if not self.nats_client:
                logger.warning(
                    "[WEAVER->WEAVELET] Cannot publish config update - no NATS client"
                )
                return

            js = self.nats_client.jetstream()

            envelope = EventEnvelope()
            config_info = envelope.config_info
            for key, value in config_params.items():
                config_info.config_params[key] = str(value)

            # Publish to UI_COMMANDS
            await js.publish(
                torchLoomConstants.subjects.UI_COMMANDS, envelope.SerializeToString()
            )

            logger.info(
                f"[WEAVER->WEAVELET] Published config update via UI_COMMANDS: {config_params}"
            )

        except Exception as e:
            logger.exception(f"[WEAVER->WEAVELET] Failed to publish config update: {e}")

    async def publish(self, message_type: str, **kwargs) -> None:
        """Generic publish method for different message types."""
        if message_type == "weaver_command":
            command_type = kwargs.get("command_type")
            target_replica_id = kwargs.get("target_replica_id")
            params = kwargs.get("params")
            if command_type is not None and target_replica_id is not None:
                await self.publish_weaver_command(
                    command_type, target_replica_id, params
                )
            else:
                logger.warning(
                    "[WEAVER->WEAVELET] Missing required parameters for weaver_command message"
                )
        elif message_type == "config_update":
            config_params = kwargs.get("config_params")
            if config_params is not None:
                await self.publish_config_update(config_params)
            else:
                logger.warning(
                    "[WEAVER->WEAVELET] Missing config_params for config_update message"
                )
        else:
            logger.warning(
                f"Unknown message type for ThreadletCommandPublisher: {message_type}"
            )
