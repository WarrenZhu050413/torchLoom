"""
Status tracking for the torchLoom Weaver.

This module manages and encapsulates UI-related information for the Weaver,
including device state, replica state, and device-replica mappings.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import TrainingStatus, UIStatusUpdate, deviceStatus

logger = setup_logger(name="status_tracker")


@dataclass
class StatusTracker:
    """
    Manages and encapsulates UI-related information for the Weaver.

    This class maintains the core protobuf state for UI updates.
    """

    # UI State (protobuf message for UI updates)
    _ui_state_proto: UIStatusUpdate = field(default_factory=UIStatusUpdate)

    # Device-replica mapping functionality
    device_to_replicas: Dict[str, Set[str]] = field(default_factory=dict)
    replica_to_devices: Dict[str, Set[str]] = field(default_factory=dict)

    # UI notification callback - will be set by UI interface
    _ui_notification_callback: Optional[Any] = field(init=False, default=None)

    def set_ui_notification_callback(self, callback):
        """Set a callback function to notify UI of changes."""
        try:
            self._ui_notification_callback = callback
            logger.debug("UI notification callback set successfully")
        except Exception as e:
            logger.error(f"Failed to set UI notification callback: {e}")

    def _notify_change(self):
        """Notify UI of status changes if callback is set."""
        try:
            if self._ui_notification_callback:
                self._ui_notification_callback()
        except Exception as e:
            logger.warning(f"Failed to notify UI of change: {e}")

    # ========================================
    # DEVICE STATE MANAGEMENT
    # ========================================

    def update_device_status_from_proto(self, dev_proto: deviceStatus):
        """Update an existing device or add a new one based on deviceStatus protobuf message."""
        try:
            now_ts = int(time.time())

            found_idx = -1
            for idx, d in enumerate(self._ui_state_proto.devices):
                if d.device_uuid == dev_proto.device_uuid:
                    found_idx = idx
                    break

            if found_idx != -1:
                # Device exists, update its state
                self._ui_state_proto.devices[found_idx].CopyFrom(dev_proto)
                logger.debug(f"Updated existing device: {dev_proto.device_uuid}")
            else:
                # New device, add it to the list
                new_device_entry = deviceStatus()
                new_device_entry.CopyFrom(dev_proto)
                self._ui_state_proto.devices.append(new_device_entry)
                logger.info(f"Added new device: {dev_proto.device_uuid}")

            self._ui_state_proto.timestamp = now_ts
            self._notify_change()

        except Exception as e:
            logger.error(f"Failed to update device status from proto: {e}")

    def get_active_devices(self) -> List[deviceStatus]:
        """Returns a list of all devices currently marked as active."""
        try:
            # Status is removed from deviceStatus, so this method needs to be re-evaluated.
            # For now, returning all devices. Consider how to determine 'active' status.
            return [d for d in self._ui_state_proto.devices]
        except Exception as e:
            logger.error(f"Failed to get active devices: {e}")
            return []

    @property
    def devices(self) -> Dict[str, deviceStatus]:
        """Get a dictionary of devices keyed by device_uuid."""
        try:
            return {device.device_uuid: device for device in self._ui_state_proto.devices}
        except Exception as e:
            logger.error(f"Failed to get devices dict: {e}")
            return {}

    # ========================================
    # REPLICA STATE MANAGEMENT
    # ========================================

    def update_training_progress_from_proto(self, ts_proto: TrainingStatus):
        """Update training status using TrainingStatus protobuf message."""
        try:
            now_ts = int(time.time())

            found_idx = -1
            for idx, r in enumerate(self._ui_state_proto.training_status):
                if r.process_id == ts_proto.process_id:
                    found_idx = idx
                    break

            if found_idx != -1:
                # Replica exists, update it
                self._ui_state_proto.training_status[found_idx].CopyFrom(ts_proto)
                logger.debug(
                    f"Updated training status for replica: {ts_proto.process_id}"
                )
            else:
                # New replica, add it
                new_training_entry = TrainingStatus()
                new_training_entry.CopyFrom(ts_proto)
                self._ui_state_proto.training_status.append(new_training_entry)
                logger.info(
                    f"Added training status for new replica: {ts_proto.process_id}"
                )

            self._ui_state_proto.timestamp = now_ts
            self._notify_change()

        except Exception as e:
            logger.error(f"Failed to update training progress from proto: {e}")

    # ========================================
    # DEVICE-REPLICA MAPPING
    # ========================================

    def add_device_replica_mapping(self, device_uuid: str, process_id: str) -> bool:
        """Add a mapping from device to replica. Returns True if this is a new mapping."""
        try:
            is_new = process_id not in self.device_to_replicas.setdefault(
                device_uuid, set()
            )
            if is_new:
                self.device_to_replicas[device_uuid].add(process_id)
                logger.debug(
                    f"Added device->replica mapping: {device_uuid} -> {process_id}"
                )
            return is_new
        except Exception as e:
            logger.error(f"Failed to add device-replica mapping: {e}")
            return False

    def add_replica_device_mapping(self, process_id: str, device_uuid: str) -> bool:
        """Add a mapping from replica to device. Returns True if this is a new mapping."""
        try:
            is_new = device_uuid not in self.replica_to_devices.setdefault(
                process_id, set()
            )
            if is_new:
                self.replica_to_devices[process_id].add(device_uuid)
                logger.debug(
                    f"Added replica->device mapping: {process_id} -> {device_uuid}"
                )
            return is_new
        except Exception as e:
            logger.error(f"Failed to add replica-device mapping: {e}")
            return False

    def get_replicas_for_device(self, device_uuid: str) -> Set[str]:
        """Get all replicas associated with a device."""
        try:
            return self.device_to_replicas.get(device_uuid, set())
        except Exception as e:
            logger.error(f"Failed to get replicas for device {device_uuid}: {e}")
            return set()

    def get_devices_for_replica(self, process_id: str) -> Set[str]:
        """Get all devices associated with a replica."""
        try:
            return self.replica_to_devices.get(process_id, set())
        except Exception as e:
            logger.error(f"Failed to get devices for replica {process_id}: {e}")
            return set()

    # ========================================
    # UI STATE MANAGEMENT
    # ========================================

    def get_ui_status_snapshot(self) -> UIStatusUpdate:
        """Returns the current UIStatusUpdate protobuf message."""
        return self._ui_state_proto

    # ========================================
    # CONVENIENCE METHODS FOR HANDLERS
    # ========================================

    def update_training_progress(self, process_id: str, **kwargs):
        """Convenience method to update training progress with keyword arguments."""
        try:
            # Find or create training status
            training_status = None
            for ts in self._ui_state_proto.training_status:
                if ts.process_id == process_id:
                    training_status = ts
                    break

            if training_status is None:
                training_status = TrainingStatus()
                training_status.process_id = process_id
                self._ui_state_proto.training_status.append(training_status)

            # Update fields from kwargs
            for key, value in kwargs.items():
                if hasattr(training_status, key):
                    # Handle protobuf map fields (like config and metrics)
                    if key == "config" and isinstance(value, dict):
                        training_status.config.clear()
                        training_status.config.update({k: str(v) for k, v in value.items()})
                    elif key == "metrics" and isinstance(value, dict):
                        training_status.metrics.clear()
                        training_status.metrics.update({k: str(v) for k, v in value.items()})
                    else:
                        setattr(training_status, key, value)
                # Move config handling to TrainingStatus
                # This specific 'if' block for config outside hasattr is now covered by the loop above.
                # Consider removing if redundant, but it's fine as is if it serves a specific purpose for new configs.
                # if key == "config" and isinstance(value, dict):
                #     training_status.config.clear()
                #     training_status.config.update({k: str(v) for k, v in value.items()})

            self._ui_state_proto.timestamp = int(time.time())
            self._notify_change()

        except Exception as e:
            logger.error(f"Failed to update training progress: {e}")

    def update_device_status(self, device_uuid: str, **kwargs):
        """Convenience method to update device status with keyword arguments."""
        try:
            # Find or create device status
            device_status = None
            for ds in self._ui_state_proto.devices:
                if ds.device_uuid == device_uuid:
                    device_status = ds
                    break

            if device_status is None:
                device_status = deviceStatus()
                device_status.device_uuid = device_uuid
                self._ui_state_proto.devices.append(device_status)

            # Update fields from kwargs
            for key, value in kwargs.items():
                if hasattr(device_status, key):
                    setattr(device_status, key, value)

            self._ui_state_proto.timestamp = int(time.time())
            self._notify_change()

        except Exception as e:
            logger.error(f"Failed to update device status: {e}")

    def update_device_config(self, device_uuid: str, config_params: Dict[str, Any]):
        """Update configuration for a specific device."""
        # Config is moved to TrainingStatus. This method should target process_id and update TrainingStatus.
        logger.warning(f"update_device_config called for {device_uuid}. Config is now part of TrainingStatus. This method needs to be updated to target a process_id.")
