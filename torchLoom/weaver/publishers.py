"""
Publishers for the torchLoom Weaver.

This module contains publishers for sending messages FROM the weaver to other components:
- UI publishers: Publish updates and responses to the UI
- Threadlet publishers: Publish commands and notifications to threadlets
"""

import time
from typing import Dict, Optional

from torchLoom.common.constants import NatsConstants
from torchLoom.common.publishers import BasePublisher
from torchLoom.log.logger import setup_logger
from torchLoom.proto import torchLoom_pb2
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

logger = setup_logger(name="weaver_publisher")


class ThreadletCommandPublisher(BasePublisher):
    """Publisher for sending commands FROM the weaver TO threadlets/training processes.
    Implements specific weaver command publishing logic.
    Direction: Weaver -> Threadlet
    """

    def __init__(self, nats_client=None, js_client=None):
        super().__init__(nats_client=nats_client, js_client=js_client)
        logger.info(
            "ThreadletCommandPublisher initialized with its own publishing method."
        )

    async def publish_weaver_command(
        self,
        command_type: str,
        target_process_id: str,
        params: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Publish a weaver command to training processes.
        This method now contains the actual publishing logic.
        """
        try:
            if not self.js_client:
                logger.error(
                    "Cannot publish weaver command - no JetStream client provided to ThreadletCommandPublisher"
                )
                return False

            envelope = EventEnvelope()
            weaver_command = envelope.weaver_command
            weaver_command.command_type = command_type
            weaver_command.target_process_id = target_process_id

            if params:
                for key, value in params.items():
                    weaver_command.params[key] = str(value)

            await self.js_client.publish(
                NatsConstants.subjects.WEAVER_COMMANDS,
                envelope.SerializeToString(),
            )

            logger.info(
                f"ThreadletCommandPublisher published weaver command: {command_type} to {target_process_id}"
            )
            return True

        except Exception as e:
            logger.error(
                f"ThreadletCommandPublisher failed to publish weaver command: {e}"
            )
            return False

    async def publish(self, message_type: str, **kwargs) -> bool:
        """Generic publish method for different message types.
        Currently only handles 'weaver_command'.
        """
        result = False
        try:
            if message_type == "weaver_command":
                command_type = kwargs.get("command_type")
                target_process_id = kwargs.get("target_process_id")
                params = kwargs.get("params")

                if command_type and target_process_id:
                    # Calls the local publish_weaver_command method
                    result = await self.publish_weaver_command(
                        command_type, target_process_id, params
                    )
                else:
                    logger.error(
                        "Missing required parameters (command_type or target_process_id) for weaver_command"
                    )
            else:
                logger.error(
                    f"Unknown message type for ThreadletCommandPublisher: {message_type}"
                )

        except Exception as e:
            logger.error(
                f"ThreadletCommandPublisher failed to publish message type {message_type}: {e}"
            )
        return result
