"""
Publishers for the torchLoom Weaver.

This module contains publishers for sending messages FROM the weaver to other components:
- UI publishers: Publish updates and responses to the UI
- Threadlet publishers: Publish commands and notifications to threadlets
"""

import time
from typing import Dict, Optional

from torchLoom.common.constants import NatsConstants
from torchLoom.common.publishers import BasePublisher, EventPublisher
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

logger = setup_logger(name="publishers")


# Re-export the base publisher for compatibility
Publisher = BasePublisher


# ===========================================
# THREADLET PUBLISHERS (Weaver -> Training Processes)
# ===========================================


class ThreadletCommandPublisher(BasePublisher):
    """Publisher for sending commands FROM the weaver TO threadlets/training processes."""

    def __init__(self, nats_client=None):
        self.nats_client = nats_client

        # Initialize the common event publisher
        try:
            self._event_publisher = EventPublisher(
                nats_client=nats_client,
                js_client=nats_client.jetstream() if nats_client else None,
            )
        except Exception as e:
            logger.error(f"Failed to initialize event publisher: {e}")
            self._event_publisher = None

    async def publish_weaver_command(
        self,
        command_type: str,
        target_process_id: str,
        params: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Publish a weaver command to training processes using the common publisher."""
        try:
            if not self._event_publisher:
                logger.error("Event publisher not available for weaver command")
                return False

            await self._event_publisher.publish_weaver_command(
                command_type=command_type,
                target_process_id=target_process_id,
                params=params,
            )

            logger.info(f"Published command: {command_type} to {target_process_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish weaver command: {e}")
            return False

    async def publish(self, message_type: str, **kwargs) -> bool:
        """Generic publish method for different message types."""
        try:
            if message_type == "weaver_command":
                command_type = kwargs.get("command_type")
                target_process_id = kwargs.get("target_process_id")
                params = kwargs.get("params")

                if command_type and target_process_id:
                    return await self.publish_weaver_command(
                        command_type, target_process_id, params
                    )
                else:
                    logger.error("Missing required parameters for weaver_command")
                    return False

            elif message_type == "config_update":
                config_params = kwargs.get("config_params")
                if config_params:
                    return await self.publish_config_update(config_params)
                else:
                    logger.error("Missing config_params for config_update")
                    return False

            else:
                logger.error(f"Unknown message type: {message_type}")
                return False

        except Exception as e:
            logger.error(f"Failed to publish message type {message_type}: {e}")
            return False
