"""
Publishers for the torchLoom Threadlet.

This module contains publishers for sending messages FROM the threadlet to other components,
using the common EventPublisher for maximum code reuse.
"""

from typing import Any, Dict, Optional

from torchLoom.common.publishers import BasePublisher, EventPublisher
from torchLoom.log.logger import setup_logger

logger = setup_logger(name="threadlet_publishers")


class ThreadletEventPublisher(BasePublisher):
    """Publisher for threadlet events using the common EventPublisher."""

    def __init__(
        self, replica_id: str, device_uuid: str, event_publisher: EventPublisher
    ):
        self.replica_id = replica_id
        self.device_uuid = device_uuid
        self.event_publisher = event_publisher

    async def publish_device_registration(self) -> None:
        """Publish device registration event."""
        await self.event_publisher.publish_device_registration(
            device_uuid=self.device_uuid,
            replica_id=self.replica_id,
        )

    async def publish_heartbeat(
        self, status: str = "active", metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """Publish heartbeat event."""
        await self.event_publisher.publish_heartbeat(
            replica_id=self.replica_id,
            device_uuid=self.device_uuid,
            status=status,
            metadata=metadata,
        )

    async def publish_training_status(self, status_data: Dict[str, Any]) -> None:
        """Publish training status event."""
        await self.event_publisher.publish_training_status(
            replica_id=self.replica_id,
            status_data=status_data,
        )

    async def publish_device_status(self, status_data: Dict[str, Any]) -> None:
        """Publish device status event."""
        await self.event_publisher.publish_device_status(
            device_id=self.device_uuid,
            replica_id=self.replica_id,
            status_data=status_data,
        )

    async def publish_metrics(
        self,
        step: int = 0,
        epoch: int = 0,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Publish training metrics as a training status event."""
        status_data = {
            "status_type": "batch_update",
            "current_step": step,
            "epoch": epoch,
            "status": "training",
            "metrics": {
                **kwargs,
            },
        }

        # Add metrics if provided
        if loss is not None:
            status_data["metrics"]["loss"] = str(loss)
        if accuracy is not None:
            status_data["metrics"]["accuracy"] = str(accuracy)
        if gradient_norm is not None:
            status_data["metrics"]["gradient_norm"] = str(gradient_norm)

        await self.publish_training_status(status_data)

    async def publish(self, message_type: str, **kwargs) -> None:
        """Generic publish method for different message types."""
        if message_type == "device_registration":
            await self.publish_device_registration()
        elif message_type == "heartbeat":
            await self.publish_heartbeat(
                kwargs.get("status", "active"),
                kwargs.get("metadata"),
            )
        elif message_type == "training_status":
            await self.publish_training_status(kwargs.get("status_data", {}))
        elif message_type == "device_status":
            await self.publish_device_status(kwargs.get("status_data", {}))
        elif message_type == "metrics":
            await self.publish_metrics(
                kwargs.get("step", 0),
                kwargs.get("epoch", 0),
                kwargs.get("loss"),
                kwargs.get("accuracy"),
                kwargs.get("gradient_norm"),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["step", "epoch", "loss", "accuracy", "gradient_norm"]
                },
            )
        else:
            logger.warning(f"Unknown message type: {message_type}")
