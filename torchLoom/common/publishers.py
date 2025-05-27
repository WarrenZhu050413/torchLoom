"""
Common publishers for torchLoom components.

This module contains shared publishing functionality that can be used by both
threadlet and weaver components to promote code reuse.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from torchLoom.common.constants import torchLoomConstants
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

logger = setup_logger(name="common_publishers")


class BasePublisher(ABC):
    """Abstract base class for all publishers."""

    @abstractmethod
    async def publish(self, *args, **kwargs) -> None:
        """Publish a message or update."""
        pass


class EventPublisher(BasePublisher):
    """Common publisher for publishing events to NATS."""

    def __init__(self, nats_client=None, js_client=None):
        self.nats_client = nats_client
        self.js_client = js_client

    async def publish_device_registration(
        self, device_uuid: str, replica_id: str
    ) -> None:
        """Publish device registration event."""
        try:
            if not self.js_client:
                logger.warning(
                    "Cannot publish device registration - no JetStream client"
                )
                return

            envelope = EventEnvelope()
            envelope.register_device.device_uuid = device_uuid
            envelope.register_device.replica_id = replica_id

            await self.js_client.publish(
                torchLoomConstants.subjects.THREADLET_EVENTS,
                envelope.SerializeToString(),
            )

            logger.info(f"Published device registration: {device_uuid} -> {replica_id}")

        except Exception as e:
            logger.exception(f"Failed to publish device registration: {e}")

    async def publish_heartbeat(
        self,
        replica_id: str,
        device_uuid: str,
        status: str = "active",
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """Publish heartbeat event."""
        try:
            if not self.nats_client:
                logger.warning("Cannot publish heartbeat - no NATS client")
                return

            envelope = EventEnvelope()
            heartbeat = envelope.heartbeat
            heartbeat.replica_id = replica_id
            heartbeat.device_uuid = device_uuid
            heartbeat.timestamp = int(time.time())
            heartbeat.status = status

            if metadata:
                for key, value in metadata.items():
                    heartbeat.metadata[key] = str(value)

            await self.nats_client.publish(
                torchLoomConstants.subjects.THREADLET_EVENTS,
                envelope.SerializeToString(),
            )

            logger.debug(f"Published heartbeat for replica {replica_id}")

        except Exception as e:
            logger.exception(f"Failed to publish heartbeat: {e}")

    async def publish_training_status(
        self, replica_id: str, status_data: Dict[str, Any]
    ) -> None:
        """Publish training status event."""
        try:
            if not self.nats_client:
                logger.warning("Cannot publish training status - no NATS client")
                return

            envelope = EventEnvelope()
            training_status = envelope.training_status

            # Set basic fields
            training_status.replica_id = replica_id
            training_status.status_type = status_data.get("status_type", "update")
            training_status.current_step = status_data.get("current_step", 0)
            training_status.epoch = status_data.get("epoch", 0)
            training_status.status = status_data.get("status", "active")
            training_status.training_time = status_data.get("training_time", 0.0)

            # Add metrics
            metrics = status_data.get("metrics", {})
            for key, value in metrics.items():
                training_status.metrics[key] = str(value)

            await self.nats_client.publish(
                torchLoomConstants.subjects.THREADLET_EVENTS,
                envelope.SerializeToString(),
            )

            logger.debug(f"Published training status for replica {replica_id}")

        except Exception as e:
            logger.exception(f"Failed to publish training status: {e}")

    async def publish_device_status(
        self, device_id: str, replica_id: str, status_data: Dict[str, Any]
    ) -> None:
        """Publish device status event."""
        try:
            if not self.nats_client:
                logger.warning("Cannot publish device status - no NATS client")
                return

            envelope = EventEnvelope()
            device_status = envelope.device_status

            # Set basic fields
            device_status.device_id = device_id
            device_status.replica_id = replica_id
            device_status.server_id = status_data.get("server_id", device_id)
            device_status.status = status_data.get("status", "active")
            device_status.utilization = status_data.get("utilization", 0.0)
            device_status.temperature = status_data.get("temperature", 0.0)
            device_status.memory_used = status_data.get("memory_used", 0.0)
            device_status.memory_total = status_data.get("memory_total", 0.0)

            # Add config
            config = status_data.get("config", {})
            for key, value in config.items():
                device_status.config[key] = str(value)

            await self.nats_client.publish(
                torchLoomConstants.subjects.THREADLET_EVENTS,
                envelope.SerializeToString(),
            )

            logger.debug(f"Published device status for device {device_id}")

        except Exception as e:
            logger.exception(f"Failed to publish device status: {e}")

    async def publish_weaver_command(
        self,
        command_type: str,
        target_replica_id: str,
        params: Optional[Dict[str, str]] = None,
    ) -> None:
        """Publish weaver command to threadlets."""
        try:
            if not self.js_client:
                logger.warning("Cannot publish weaver command - no JetStream client")
                return

            envelope = EventEnvelope()
            weaver_command = envelope.weaver_command
            weaver_command.command_type = command_type
            weaver_command.target_replica_id = target_replica_id

            if params:
                for key, value in params.items():
                    weaver_command.params[key] = str(value)

            await self.js_client.publish(
                torchLoomConstants.subjects.WEAVER_COMMANDS,
                envelope.SerializeToString(),
            )

            logger.info(
                f"Published weaver command: {command_type} to {target_replica_id}"
            )

        except Exception as e:
            logger.exception(f"Failed to publish weaver command: {e}")

    async def publish_ui_command(
        self,
        command_type: str,
        target_id: str,
        params: Optional[Dict[str, str]] = None,
    ) -> None:
        """Publish UI command."""
        try:
            if not self.js_client:
                logger.warning("Cannot publish UI command - no JetStream client")
                return

            envelope = EventEnvelope()
            ui_command = envelope.ui_command
            ui_command.command_type = command_type
            ui_command.target_id = target_id

            if params:
                for key, value in params.items():
                    ui_command.params[key] = str(value)

            await self.js_client.publish(
                torchLoomConstants.subjects.UI_COMMANDS,
                envelope.SerializeToString(),
            )

            logger.info(f"Published UI command: {command_type} to {target_id}")

        except Exception as e:
            logger.exception(f"Failed to publish UI command: {e}")

    async def publish(self, message_type: str, **kwargs) -> None:
        """Generic publish method for different message types."""
        if message_type == "device_registration":
            await self.publish_device_registration(
                kwargs.get("device_uuid"), kwargs.get("replica_id")
            )
        elif message_type == "heartbeat":
            await self.publish_heartbeat(
                kwargs.get("replica_id"),
                kwargs.get("device_uuid"),
                kwargs.get("status", "active"),
                kwargs.get("metadata"),
            )
        elif message_type == "training_status":
            await self.publish_training_status(
                kwargs.get("replica_id"), kwargs.get("status_data", {})
            )
        elif message_type == "device_status":
            await self.publish_device_status(
                kwargs.get("device_id"),
                kwargs.get("replica_id"),
                kwargs.get("status_data", {}),
            )
        elif message_type == "weaver_command":
            await self.publish_weaver_command(
                kwargs.get("command_type"),
                kwargs.get("target_replica_id"),
                kwargs.get("params"),
            )
        elif message_type == "ui_command":
            await self.publish_ui_command(
                kwargs.get("command_type"),
                kwargs.get("target_id"),
                kwargs.get("params"),
            )
        else:
            logger.warning(f"Unknown message type: {message_type}")


class UIStatusPublisher(BasePublisher):
    """Publisher for UI status updates using the same data structure as weaver publishers."""

    def __init__(self, status_tracker):
        self.status_tracker = status_tracker

    async def create_ui_status_update(self) -> Optional[EventEnvelope]:
        """Create a UIStatusUpdate envelope using the same structure as weaver publishers."""
        try:
            envelope = EventEnvelope()
            ui_update = envelope.ui_status_update
            ui_update.communication_status = getattr(
                self.status_tracker, "communication_status", "stable"
            )
            ui_update.timestamp = int(time.time())

            # Add all device statuses
            devices = getattr(self.status_tracker, "devices", {})
            if hasattr(devices, "values"):
                for device_info in devices.values():
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

            # Add all training statuses
            if hasattr(self.status_tracker, "get_ui_status_snapshot"):
                ui_snapshot = self.status_tracker.get_ui_status_snapshot()
                for existing_training_status in ui_snapshot.training_status:
                    training_status = ui_update.training_status.add()
                    training_status.CopyFrom(existing_training_status)

            logger.debug("Created UIStatusUpdate envelope")
            return envelope

        except Exception as e:
            logger.exception(f"Failed to create UIStatusUpdate envelope: {e}")
            return None

    async def publish(self) -> Optional[EventEnvelope]:
        """Implement the abstract publish method."""
        return await self.create_ui_status_update()
