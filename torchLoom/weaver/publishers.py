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
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

logger = setup_logger(name="publishers")


class Publisher(ABC):
    """Abstract base class for message publishers."""

    @abstractmethod
    async def publish(self, *args, **kwargs) -> None:
        """Publish a message or update."""
        pass


# ===========================================
# UI PUBLISHERS (Weaver -> UI)
# ===========================================


class UIUpdatePublisher(Publisher):
    """Publisher for sending consolidated UI updates FROM the weaver TO the UI."""

    def __init__(self, status_tracker):
        self.status_tracker = status_tracker

    async def publish_ui_update(self) -> Optional[EventEnvelope]:
        """Constructs a UIStatusUpdate message. Does not publish to NATS anymore."""
        try:
            # Create consolidated UIStatusUpdate
            envelope = EventEnvelope()
            ui_update = envelope.ui_status_update
            # ui_update.communication_status = self.status_tracker.communication_status # This field is not in UIStatusUpdate proto
            ui_update.timestamp = int(time.time())

            # Add all device statuses
            for device_info in self.status_tracker.devices.values():
                device_status = ui_update.devices.add()
                device_status.device_id = device_info.device_id
                device_status.replica_id = device_info.replica_id
                device_status.server_id = device_info.server_id
                device_status.status = device_info.status
                device_status.utilization = device_info.utilization
                device_status.temperature = device_info.temperature
                device_status.memory_used = device_info.memory_used
                device_status.memory_total = device_info.memory_total

                # Add config
                for key, value in device_info.config.items():
                    device_status.config[key] = str(value)

            # Add all training statuses (copy from existing UI state)
            ui_snapshot = self.status_tracker.get_ui_status_snapshot()
            for existing_training_status in ui_snapshot.training_status:
                training_status = ui_update.training_status.add()
                training_status.CopyFrom(existing_training_status)

            # Note: Topology information is not currently tracked in the reorganized StatusTracker
            # If topology is needed, it would need to be added to the StatusTracker or derived from device-replica mappings

            # Publish to UI - This part is removed as UI_UPDATE NATS subject is removed.
            # await self.nats_client.publish(
            #     torchLoomConstants.subjects.UI_UPDATE, envelope.SerializeToString()
            # )

            # logger.debug("[WEAVER->UI] Published UI update to clients") # Logged that it was constructed
            logger.debug(
                "[WEAVER->UI] Constructed UIStatusUpdate envelope. Publishing to NATS is removed."
            )
            return envelope  # Return the constructed envelope

        except Exception as e:
            logger.exception(
                f"[WEAVER->UI] Failed to construct UIStatusUpdate envelope: {e}"
            )
            return None

    async def publish(self) -> None:
        """Implement the abstract publish method."""
        await self.publish_ui_update()  # Now just constructs and logs


# ===========================================
# WEAVELET PUBLISHERS (Weaver -> Training Processes)
# ===========================================


class ThreadletCommandPublisher(Publisher):
    """Publisher for sending commands FROM the weaver TO threadlets/training processes.

    Also includes heartbeat monitoring functionality to detect dead replicas
    and publish failure events for them.
    """

    def __init__(self, nats_client=None, threadlet_handler=None):
        self.nats_client = nats_client
        self.threadlet_handler = (
            threadlet_handler  # Reference to threadlet handler for heartbeat monitoring
        )

    async def publish_weaver_command(
        self,
        command_type: str,
        target_replica_id: str,
        params: Optional[Dict[str, str]] = None,
    ) -> None:
        """Publish a weaver command to training processes."""
        try:
            if not self.nats_client:
                logger.warning(
                    "[WEAVER->WEAVELET] Cannot publish weaver command - no NATS client"
                )
                return

            envelope = EventEnvelope()
            weaver_command = envelope.weaver_command
            weaver_command.command_type = command_type
            weaver_command.target_replica_id = target_replica_id

            if params:
                for key, value in params.items():
                    weaver_command.params[key] = str(value)

            js = self.nats_client.jetstream()
            await js.publish(
                torchLoomConstants.subjects.WEAVER_COMMANDS,
                envelope.SerializeToString(),
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

            logger.info(f"[WEAVER->WEAVELET] Published config update via UI_COMMANDS: {config_params}")

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
