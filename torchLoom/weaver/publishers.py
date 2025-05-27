"""
Publishers for the torchLoom Weaver.

This module contains publishers for sending messages FROM the weaver to other components:
- UI publishers: Publish updates and responses to the UI
- Threadlet publishers: Publish commands and notifications to threadlets
- Status tracking: Manages UI-related state and device/replica mappings
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from torchLoom.common.constants import torchLoomConstants
from torchLoom.common.publishers import BasePublisher, EventPublisher
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import (
    EventEnvelope,
    TrainingStatus,
    UIStatusUpdate,
    deviceStatus,
)

logger = setup_logger(name="publishers")


# Re-export the base publisher for compatibility
Publisher = BasePublisher


# ===========================================
# STATUS TRACKING (Moved from status_tracker.py)
# ===========================================


@dataclass
class StatusTracker:
    """
    Manages and encapsulates UI-related information for the Weaver.

    This class maintains two primary states:
    1. Device State: Physical device information, status, and configurations
    2. Replica State: Training replica progress, status, and device-replica mappings

    The device-replica mappings that were previously handled by DeviceReplicaMapper
    are now integrated into the replica state management.

    All UI publishing logic is now integrated within this module for better organization.
    """

    # UI State (protobuf message for UI updates)
    _ui_state_proto: UIStatusUpdate = field(default_factory=UIStatusUpdate)
    communication_status: str = field(default="stable")

    # === REPLICA STATE ===
    # Device-replica mapping functionality (moved from DeviceReplicaMapper)
    device_to_replicas: Dict[str, Set[str]] = field(default_factory=dict)
    replica_to_devices: Dict[str, Set[str]] = field(default_factory=dict)

    # UI status publisher will be injected from weaver
    _ui_publisher: Optional[Any] = field(init=False, default=None)

    def set_ui_publisher(self, ui_publisher):
        """Set the UI publisher (injected from weaver)."""
        self._ui_publisher = ui_publisher

    # ========================================
    # DEVICE STATE MANAGEMENT
    # ========================================

    def update_device_status_from_proto(self, dev_proto: deviceStatus):
        """
        Updates an existing device or adds a new one in the UI state
        based on an incoming deviceStatus protobuf message.

        The dev_proto is assumed to be a fully populated deviceStatus message
        representing the latest state of that device.
        """
        now_ts = int(time.time())

        found_idx = -1
        for idx, d in enumerate(self._ui_state_proto.devices):
            if d.device_id == dev_proto.device_id:
                found_idx = idx
                break

        if found_idx != -1:
            # Device exists, update its state.
            self._ui_state_proto.devices[found_idx].CopyFrom(dev_proto)
            logger.debug(f"Updated existing device in UI state: {dev_proto.device_id}")
        else:
            # New device, add it to the list.
            new_device_entry = deviceStatus()
            new_device_entry.CopyFrom(dev_proto)
            self._ui_state_proto.devices.append(new_device_entry)
            logger.debug(f"Added new device to UI state: {dev_proto.device_id}")

        self._ui_state_proto.timestamp = now_ts

    def update_device_status(
        self,
        device_id: str,
        replica_id: str = None,
        server_id: str = None,
        status: str = None,
        utilization: float = None,
        temperature: float = None,
        memory_used: float = None,
        memory_total: float = None,
        config: Dict[str, Any] = None,
    ):
        """Update device status with individual parameters."""
        # Create a deviceStatus proto
        device_status = deviceStatus()
        device_status.device_id = device_id
        if replica_id is not None:
            device_status.replica_id = replica_id
        if server_id is not None:
            device_status.server_id = server_id
        if status is not None:
            device_status.status = status
        if utilization is not None:
            device_status.utilization = utilization
        if temperature is not None:
            device_status.temperature = temperature
        if memory_used is not None:
            device_status.memory_used = memory_used
        if memory_total is not None:
            device_status.memory_total = memory_total
        if config is not None:
            device_status.config.update({k: str(v) for k, v in config.items()})

        self.update_device_status_from_proto(device_status)

    def update_device_config(self, device_id: str, config_params: Dict[str, str]):
        """Update device configuration parameters."""
        for device in self._ui_state_proto.devices:
            if device.device_id == device_id:
                device.config.update(config_params)
                self._ui_state_proto.timestamp = int(time.time())
                logger.debug(f"Updated config for device {device_id}: {config_params}")
                return
        logger.warning(f"Device {device_id} not found for config update")

    def deactivate_device(self, device_id: str):
        """Deactivate a device by setting its status to inactive."""
        for device in self._ui_state_proto.devices:
            if device.device_id == device_id:
                device.status = "inactive"
                self._ui_state_proto.timestamp = int(time.time())
                logger.info(f"Deactivated device: {device_id}")
                return
        logger.warning(f"Device {device_id} not found for deactivation")

    def get_active_devices(self) -> List[deviceStatus]:
        """Returns a list of all devices currently marked as "active" in the UI state."""
        return [d for d in self._ui_state_proto.devices if d.status == "active"]

    @property
    def devices(self) -> Dict[str, deviceStatus]:
        """Get a dictionary of devices keyed by device_id for compatibility with existing code."""
        return {device.device_id: device for device in self._ui_state_proto.devices}

    # ========================================
    # REPLICA STATE MANAGEMENT
    # ========================================

    def update_training_progress_from_proto(self, ts_proto: TrainingStatus):
        """
        Updates an existing replica's training status or adds a new one
        in the UI state using an incoming TrainingStatus protobuf message.

        The ts_proto is assumed to be a fully populated TrainingStatus message
        representing the latest state of that training replica.
        """
        now_ts = int(time.time())

        found_idx = -1
        for idx, r in enumerate(self._ui_state_proto.training_status):
            if r.replica_id == ts_proto.replica_id:
                found_idx = idx
                break

        if found_idx != -1:
            # Replica's training status exists, update it.
            self._ui_state_proto.training_status[found_idx].CopyFrom(ts_proto)
            logger.debug(
                f"Updated existing replica training status in UI state: {ts_proto.replica_id}"
            )
        else:
            # New replica's training status, add it to the list.
            new_training_entry = TrainingStatus()
            new_training_entry.CopyFrom(ts_proto)
            self._ui_state_proto.training_status.append(new_training_entry)
            logger.debug(
                f"Added new replica training status to UI state: {ts_proto.replica_id}"
            )

        self._ui_state_proto.timestamp = now_ts

    def update_training_progress(self, replica_id: str, status: str, **kwargs):
        """Update training progress for a replica."""
        # Create a simple TrainingStatus proto
        training_status = TrainingStatus()
        training_status.replica_id = replica_id
        training_status.status = status
        training_status.status_type = kwargs.get("status_type", "update")
        training_status.current_step = kwargs.get("current_step", 0)
        training_status.epoch = kwargs.get("epoch", 0)
        training_status.training_time = kwargs.get("training_time", 0.0)
        training_status.max_step = kwargs.get("max_step", 0)
        training_status.max_epoch = kwargs.get("max_epoch", 0)

        # Handle step_progress and other parameters that might be passed but not directly in proto
        metrics = kwargs.get("metrics", {}).copy()
        if "step_progress" in kwargs:
            metrics["step_progress"] = str(kwargs["step_progress"])
        if "last_active_step" in kwargs:
            metrics["last_active_step"] = str(kwargs["last_active_step"])

        # Update metrics
        training_status.metrics.update({k: str(v) for k, v in metrics.items()})

        self.update_training_progress_from_proto(training_status)

    # Device-replica mapping methods (moved from DeviceReplicaMapper)
    def add_device_replica_mapping(self, device_uuid: str, replica_id: str) -> bool:
        """Add a mapping from device to replica. Returns True if this is a new mapping."""
        is_new = replica_id not in self.device_to_replicas.setdefault(
            device_uuid, set()
        )
        if is_new:
            self.device_to_replicas[device_uuid].add(replica_id)
            logger.debug(
                f"Added device->replica mapping: {device_uuid} -> {replica_id}"
            )
        return is_new

    def add_replica_device_mapping(self, replica_id: str, device_uuid: str) -> bool:
        """Add a mapping from replica to device. Returns True if this is a new mapping."""
        is_new = device_uuid not in self.replica_to_devices.setdefault(
            replica_id, set()
        )
        if is_new:
            self.replica_to_devices[replica_id].add(device_uuid)
            logger.debug(
                f"Added replica->device mapping: {replica_id} -> {device_uuid}"
            )
        return is_new

    def get_replicas_for_device(self, device_uuid: str) -> Set[str]:
        """Get all replicas associated with a device."""
        return self.device_to_replicas.get(device_uuid, set())

    def get_devices_for_replica(self, replica_id: str) -> Set[str]:
        """Get all devices associated with a replica."""
        return self.replica_to_devices.get(replica_id, set())

    # ========================================
    # UI STATE MANAGEMENT
    # ========================================

    def get_ui_status_snapshot(self) -> UIStatusUpdate:
        """
        Returns the current UIStatusUpdate protobuf message.
        Consider returning a copy if external modification is a concern:
        snapshot = UIStatusUpdate()
        snapshot.CopyFrom(self._ui_state_proto)
        return snapshot
        """
        return self._ui_state_proto

    async def create_ui_status_update(self):
        """Create a UI status update using the injected publisher."""
        if self._ui_publisher:
            return await self._ui_publisher.create_ui_status_update()
        else:
            logger.warning("No UI publisher available for creating UI status update")
            return None

    def set_communication_status(self, status: str):
        """Updates the overall communication status string."""
        self.communication_status = status
        self._ui_state_proto.timestamp = int(time.time())  # Update timestamp on change
        logger.info(f"Communication status set to: {status}")

    # Add compatibility property for replicas (used by websocket_server)
    @property
    def replicas(self) -> Dict[str, Any]:
        """Get replica information for compatibility with existing code."""
        replica_dict = {}
        for training_status in self._ui_state_proto.training_status:
            replica_dict[training_status.replica_id] = type(
                "Replica",
                (),
                {
                    "status": training_status.status,
                    "step_progress": float(
                        training_status.metrics.get("step_progress", 0)
                    ),
                    "last_active_step": int(
                        training_status.metrics.get("last_active_step", 0)
                    ),
                    "fixed_step": None,  # This would need to be added to protobuf if needed
                },
            )()
        return replica_dict

    def get_system_summary(self) -> Dict[str, Any]:
        """Get system summary for UI display."""
        active_devices = len(self.get_active_devices())
        total_devices = len(self._ui_state_proto.devices)
        active_replicas = len(
            [r for r in self._ui_state_proto.training_status if r.status == "training"]
        )
        total_replicas = len(self._ui_state_proto.training_status)

        return {
            "active_devices": active_devices,
            "total_devices": total_devices,
            "active_replicas": active_replicas,
            "total_replicas": total_replicas,
            "communication_status": self.communication_status,
        }


# ===========================================
# UI PUBLISHERS (Weaver -> UI)
# ===========================================


class WeaverUIStatusPublisher(BasePublisher):
    """Publisher for UI status updates specific to the Weaver.

    This class contains all the UI publishing logic moved from StatusTracker
    to centralize UI publishing functionality in the publishers module.
    """

    def __init__(self, status_tracker):
        self.status_tracker = status_tracker

    async def create_ui_status_update(self) -> Optional[EventEnvelope]:
        """Create a UIStatusUpdate envelope using the status tracker's data."""
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


class UIUpdatePublisher(WeaverUIStatusPublisher):
    """Publisher for sending consolidated UI updates FROM the weaver TO the UI.

    This now inherits from WeaverUIStatusPublisher for maximum code reuse.
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
