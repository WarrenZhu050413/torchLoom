"""
Publishers for the torchLoom Threadlet.

This module contains publishers for sending messages FROM the threadlet to other components,
using the common EventPublisher for maximum code reuse.
"""
import time # Added for heartbeat timestamp
from typing import Any, Dict, Optional

from torchLoom.common.constants import NatsConstants # Added
from torchLoom.proto import torchLoom_pb2 # Added for EventEnvelope
from torchLoom.common.publishers import BasePublisher
from torchLoom.log.logger import setup_logger

logger = setup_logger(name="threadlet_publishers")


class ThreadletEventPublisher(BasePublisher):
    """Publisher for sending events FROM the threadlet TO NATS (e.g., weaver).
    Implements specific event publishing logic.
    """

    def __init__(self, nats_client, js_client, process_id: str, device_uuid: str):
        super().__init__(nats_client=nats_client, js_client=js_client)
        self._process_id = process_id
        self._device_uuid = device_uuid
        logger.info("ThreadletEventPublisher initialized with its own publishing methods.")

    async def publish_device_registration(
        self
    ) -> None:
        """Publish device registration event."""
        try:
            if not self.js_client:
                logger.warning(
                    "Cannot publish device registration - no JetStream client provided to ThreadletEventPublisher"
                )
                return

            envelope = torchLoom_pb2.EventEnvelope()
            envelope.register_device.device_uuid = self._device_uuid
            envelope.register_device.process_id = self._process_id

            await self.js_client.publish(
                NatsConstants.subjects.THREADLET_EVENTS,
                envelope.SerializeToString(),
            )
            logger.info(f"ThreadletEventPublisher published device registration: {self._device_uuid} -> {self._process_id}")
        except Exception as e:
            logger.exception(f"ThreadletEventPublisher failed to publish device registration: {e}")

    async def publish_heartbeat(
        self,
        status: str = "active",
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """Publish heartbeat event."""
        try:
            if not self.nats_client:
                logger.warning("Cannot publish heartbeat - no NATS client provided to ThreadletEventPublisher")
                return

            envelope = torchLoom_pb2.EventEnvelope()
            heartbeat = envelope.heartbeat
            heartbeat.process_id = self._process_id
            heartbeat.device_uuid = self._device_uuid
            heartbeat.timestamp = int(time.time())
            heartbeat.status = status

            if metadata:
                for key, value in metadata.items():
                    heartbeat.metadata[key] = str(value)

            await self.nats_client.publish(
                NatsConstants.subjects.THREADLET_EVENTS,
                envelope.SerializeToString(),
            )
            logger.debug(f"ThreadletEventPublisher published heartbeat for replica {self._process_id}")
        except Exception as e:
            logger.exception(f"ThreadletEventPublisher failed to publish heartbeat: {e}")

    async def publish_training_status(
        self, status_data: Dict[str, Any]
    ) -> None:
        """Publish training status event."""
        try:
            if not self.nats_client:
                logger.warning("Cannot publish training status - no NATS client provided to ThreadletEventPublisher")
                return

            envelope = torchLoom_pb2.EventEnvelope()
            training_status = envelope.training_status
            training_status.process_id = self._process_id
            training_status.current_step = status_data.get("current_step", 0)
            training_status.epoch = status_data.get("epoch", 0)
            training_status.training_time = status_data.get("training_time", 0.0)
            training_status.max_step = status_data.get("max_step", 0)
            training_status.max_epoch = status_data.get("max_epoch", 0)

            metrics = status_data.get("metrics", {})
            for key, value in metrics.items():
                training_status.metrics[key] = str(value)
            config = status_data.get("config", {})
            for key, value in config.items():
                training_status.config[key] = str(value)

            await self.nats_client.publish(
                NatsConstants.subjects.THREADLET_EVENTS,
                envelope.SerializeToString(),
            )
            logger.debug(f"ThreadletEventPublisher published training status for replica {self._process_id}")
        except Exception as e:
            logger.exception(f"ThreadletEventPublisher failed to publish training status: {e}")

    async def publish_device_status(
        self, status_data: Dict[str, Any]
    ) -> None:
        """Publish device status event."""
        try:
            if not self.nats_client:
                logger.warning("Cannot publish device status - no NATS client provided to ThreadletEventPublisher")
                return

            envelope = torchLoom_pb2.EventEnvelope()
            device_status = envelope.device_status
            device_status.device_uuid = self._device_uuid
            device_status.process_id = self._process_id
            device_status.server_id = status_data.get("server_id", self._device_uuid) # Consider if server_id is available
            device_status.utilization = status_data.get("utilization", 0.0)
            device_status.temperature = status_data.get("temperature", 0.0)
            device_status.memory_used = status_data.get("memory_used", 0.0)
            device_status.memory_total = status_data.get("memory_total", 0.0)

            config = status_data.get("config", {})
            for key, value in config.items():
                device_status.config[key] = str(value)

            await self.nats_client.publish(
                NatsConstants.subjects.THREADLET_EVENTS,
                envelope.SerializeToString(),
            )
            logger.debug(f"ThreadletEventPublisher published device status for device {self._device_uuid}")
        except Exception as e:
            logger.exception(f"ThreadletEventPublisher failed to publish device status: {e}")

    async def publish_metrics(
        self,
        current_step: int = 0,
        epoch: int = 0,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        max_step: int = 0,
        max_epoch: int = 0,
        training_time: float = 0.0,
        config: Optional[Dict[str, str]] = None,
        metrics: Optional[Dict[str, str]] = None,
    ) -> None:
        """Publish training metrics by formatting them into a training status event."""
        status_data = {
            "current_step": current_step,
            "epoch": epoch,
            "max_step": max_step,
            "max_epoch": max_epoch,
            "training_time": training_time,
            "metrics": metrics or {},
            "config": config or {},
        }

        if loss is not None:
            status_data["metrics"]["loss"] = str(loss)
        if accuracy is not None:
            status_data["metrics"]["accuracy"] = str(accuracy)
        if gradient_norm is not None:
            status_data["metrics"]["gradient_norm"] = str(gradient_norm)

        # Calls the local publish_training_status method
        await self.publish_training_status(
            process_id=self._process_id,
            status_data=status_data
        )
        logger.debug(f"Published metrics for process {self._process_id} via training_status call in ThreadletEventPublisher.")

    async def publish(self, message_type: str, **kwargs) -> None:
        """Generic publish method, expecting necessary IDs (e.g., process_id, device_uuid) in kwargs."""
        if message_type == "device_registration":
            if "device_uuid" not in kwargs or "process_id" not in kwargs:
                logger.error("device_uuid and process_id missing for publishing device_registration")
                return
            await self.publish_device_registration(
                device_uuid=kwargs["device_uuid"], process_id=kwargs["process_id"]
            )
        elif message_type == "heartbeat":
            if "device_uuid" not in kwargs or "process_id" not in kwargs:
                logger.error("device_uuid and process_id missing for publishing heartbeat")
                return
            await self.publish_heartbeat(
                status=kwargs.get("status", "active"),
                metadata=kwargs.get("metadata"),
            )
        elif message_type == "training_status":
            if "process_id" not in kwargs or "status_data" not in kwargs:
                logger.error("process_id or status_data missing for publishing training_status")
                return
            await self.publish_training_status(
                status_data=kwargs["status_data"]
            )
        elif message_type == "device_status":
            if "device_uuid" not in kwargs or "process_id" not in kwargs or "status_data" not in kwargs:
                logger.error("device_uuid, process_id, or status_data missing for publishing device_status")
                return
            await self.publish_device_status(
                status_data=kwargs["status_data"],
            )
        elif message_type == "metrics":
            if "process_id" not in kwargs: # process_id is the main one checked by publish_metrics itself
                logger.error("process_id missing for publishing metrics")
                return
            # publish_metrics will extract other args from kwargs or use defaults
            await self.publish_metrics(**kwargs)
        else:
            # If EventPublisher base class is intended to have other publish types, call super.
            # Otherwise, this indicates an unknown type for ThreadletEventPublisher.
            logger.warning(f"Unknown message type for ThreadletEventPublisher: {message_type}. Falling back to super.publish if implemented.")
            try:
                await super().publish(message_type, **kwargs)
            except AttributeError: # If super().publish is not implemented or abstract
                 logger.error(f"super().publish not available or message type {message_type} not handled.")
