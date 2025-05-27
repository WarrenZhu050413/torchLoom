"""
Common publishers for torchLoom components.

This module contains shared publishing functionality that can be used by both
threadlet and weaver components to promote code reuse.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from torchLoom.common.constants import Config, NatsConstants
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
        self, device_uuid: str, process_id: str
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
            envelope.register_device.process_id = process_id

            await self.js_client.publish(
                NatsConstants.subjects.THREADLET_EVENTS,
                envelope.SerializeToString(),
            )

            logger.info(f"Published device registration: {device_uuid} -> {process_id}")

        except Exception as e:
            logger.exception(f"Failed to publish device registration: {e}")

    async def publish_heartbeat(
        self,
        process_id: str,
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
            heartbeat.process_id = process_id
            heartbeat.device_uuid = device_uuid
            heartbeat.timestamp = int(time.time())
            heartbeat.status = status

            if metadata:
                for key, value in metadata.items():
                    heartbeat.metadata[key] = str(value)

            await self.nats_client.publish(
                NatsConstants.subjects.THREADLET_EVENTS,
                envelope.SerializeToString(),
            )

            logger.debug(f"Published heartbeat for replica {process_id}")

        except Exception as e:
            logger.exception(f"Failed to publish heartbeat: {e}")

    async def publish_training_status(
        self, process_id: str, status_data: Dict[str, Any]
    ) -> None:
        """Publish training status event."""
        try:
            if not self.nats_client:
                logger.warning("Cannot publish training status - no NATS client")
                return

            envelope = EventEnvelope()
            training_status = envelope.training_status

            # Set basic fields
            training_status.process_id = process_id
            training_status.current_step = status_data.get("current_step", 0)
            training_status.epoch = status_data.get("epoch", 0)
            training_status.training_time = status_data.get("training_time", 0.0)
            training_status.max_step = status_data.get("max_step", 0)
            training_status.max_epoch = status_data.get("max_epoch", 0)

            # Add metrics
            metrics = status_data.get("metrics", {})
            for key, value in metrics.items():
                training_status.metrics[key] = str(value)

            # Add config
            config = status_data.get("config", {})
            for key, value in config.items():
                training_status.config[key] = str(value)

            await self.nats_client.publish(
                NatsConstants.subjects.THREADLET_EVENTS,
                envelope.SerializeToString(),
            )

            logger.debug(f"Published training status for replica {process_id}")

        except Exception as e:
            logger.exception(f"Failed to publish training status: {e}")

    async def publish_device_status(
        self, device_id: str, process_id: str, status_data: Dict[str, Any]
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
            device_status.process_id = process_id
            device_status.server_id = status_data.get("server_id", device_id)
            device_status.utilization = status_data.get("utilization", 0.0)
            device_status.temperature = status_data.get("temperature", 0.0)
            device_status.memory_used = status_data.get("memory_used", 0.0)
            device_status.memory_total = status_data.get("memory_total", 0.0)

            # Add config
            config = status_data.get("config", {})
            for key, value in config.items():
                device_status.config[key] = str(value)

            await self.nats_client.publish(
                NatsConstants.subjects.THREADLET_EVENTS,
                envelope.SerializeToString(),
            )

            logger.debug(f"Published device status for device {device_id}")

        except Exception as e:
            logger.exception(f"Failed to publish device status: {e}")

    async def publish_weaver_command(
        self,
        command_type: str,
        target_process_id: str,
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
            weaver_command.target_process_id = target_process_id

            if params:
                for key, value in params.items():
                    weaver_command.params[key] = str(value)

            await self.js_client.publish(
                NatsConstants.subjects.WEAVER_COMMANDS,
                envelope.SerializeToString(),
            )

            logger.info(
                f"Published weaver command: {command_type} to {target_process_id}"
            )

        except Exception as e:
            logger.exception(f"Failed to publish weaver command: {e}")

    async def publish(self, message_type: str, **kwargs) -> None:
        """Generic publish method for different message types."""
        if message_type == "device_registration":
            await self.publish_device_registration(
                kwargs.get("device_uuid"), kwargs.get("process_id")
            )
        elif message_type == "heartbeat":
            await self.publish_heartbeat(
                kwargs.get("process_id"),
                kwargs.get("device_uuid"),
                kwargs.get("status", "active"),
                kwargs.get("metadata"),
            )
        elif message_type == "training_status":
            await self.publish_training_status(
                kwargs.get("process_id"), kwargs.get("status_data", {})
            )
        elif message_type == "device_status":
            await self.publish_device_status(
                kwargs.get("device_id"),
                kwargs.get("process_id"),
                kwargs.get("status_data", {}),
            )
        elif message_type == "weaver_command":
            await self.publish_weaver_command(
                kwargs.get("command_type"),
                kwargs.get("target_process_id"),
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
