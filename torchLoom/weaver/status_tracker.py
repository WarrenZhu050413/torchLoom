"""
Status tracking for the torchLoom Weaver.

This module manages and encapsulates UI-related information for the Weaver,
including device state, replica state, and device-replica mappings.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import TrainingStatus, UIStatusUpdate, deviceStatus

logger = setup_logger(name="status_tracker")


@dataclass
class StatusTracker:
    """
    Manages and encapsulates UI-related information for the Weaver.

    This class maintains the core protobuf state for UI updates.
    """

    _ui_state_proto: UIStatusUpdate = field(default_factory=UIStatusUpdate)
    device_to_pid: Dict[str, Set[str]] = field(default_factory=dict)
    pid_to_devices: Dict[str, Set[str]] = field(default_factory=dict)
    _ui_notification_callback: Optional[Callable[[], None]] = None

    def set_ui_notification_callback(self, callback: Callable[[], None]) -> None:
        """Set the UI notification callback."""
        self._ui_notification_callback = callback

    def connect_ui_manager(self, ui_manager) -> None:
        """Automatically connect to a UI notification manager."""
        if hasattr(ui_manager, "notify_status_change"):
            self.set_ui_notification_callback(ui_manager.notify_status_change)
            logger.info(
                "StatusTracker automatically connected to UI notification manager"
            )
        else:
            logger.warning("UI manager does not have notify_status_change method")

    def _notify_change(self):
        """Notify UI of status changes if callback is set."""
        try:
            if self._ui_notification_callback:
                self._ui_notification_callback()
        except Exception as e:
            logger.warning(f"Failed to notify UI of change: {e}")

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
                self._ui_state_proto.devices[found_idx].CopyFrom(dev_proto)
                logger.debug(f"Updated existing device: {dev_proto.device_uuid}")
            else:
                new_device_entry = deviceStatus()
                new_device_entry.CopyFrom(dev_proto)
                self._ui_state_proto.devices.append(new_device_entry)
                logger.info(f"Added new device: {dev_proto.device_uuid}")

            self._ui_state_proto.timestamp = now_ts
            self._notify_change()

        except Exception as e:
            logger.error(f"Failed to update device status from proto: {e}")

    def get_active_devices(self) -> List[deviceStatus]:
        try:
            return [d for d in self._ui_state_proto.devices]
        except Exception as e:
            logger.error(f"Failed to get active devices: {e}")
            return []

    @property
    def devices(self) -> Dict[str, deviceStatus]:
        try:
            return {
                device.device_uuid: device for device in self._ui_state_proto.devices
            }
        except Exception as e:
            logger.error(f"Failed to get devices dict: {e}")
            return {}

    def update_training_progress_from_proto(self, ts_proto: TrainingStatus):
        try:
            now_ts = int(time.time())

            found_idx = -1
            for idx, r in enumerate(self._ui_state_proto.training_status):
                if r.process_id == ts_proto.process_id:
                    found_idx = idx
                    break

            if found_idx != -1:
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

    def add_device_pid_mapping(self, device_uuid: str, process_id: str) -> bool:
        try:
            is_new = process_id not in self.device_to_pid.setdefault(device_uuid, set())
            if is_new:
                self.device_to_pid[device_uuid].add(process_id)
                logger.debug(
                    f"Added device->replica mapping: {device_uuid} -> {process_id}"
                )
            return is_new
        except Exception as e:
            logger.error(f"Failed to add device-replica mapping: {e}")
            return False

    def add_pid_device_mapping(self, process_id: str, device_uuid: str) -> bool:
        try:
            is_new = device_uuid not in self.pid_to_devices.setdefault(
                process_id, set()
            )
            if is_new:
                self.pid_to_devices[process_id].add(device_uuid)
                logger.debug(
                    f"Added replica->device mapping: {process_id} -> {device_uuid}"
                )
            return is_new
        except Exception as e:
            logger.error(f"Failed to add replica-device mapping: {e}")
            return False

    def get_pid_for_device(self, device_uuid: str) -> Set[str]:
        try:
            return self.device_to_pid.get(device_uuid, set())
        except Exception as e:
            logger.error(f"Failed to get replicas for device {device_uuid}: {e}")
            return set()

    def get_devices_for_pid(self, process_id: str) -> Set[str]:
        try:
            return self.pid_to_devices.get(process_id, set())
        except Exception as e:
            logger.error(f"Failed to get devices for replica {process_id}: {e}")
            return set()

    def has_process_id(self, process_id: str) -> bool:
        try:
            return process_id in self.pid_to_devices
        except Exception as e:
            logger.error(f"Failed to check if process_id {process_id} exists: {e}")
            return False

    def get_ui_status_snapshot(self) -> UIStatusUpdate:
        return self._ui_state_proto

    def update_training_progress(self, process_id: str, **kwargs):
        try:
            training_status = None
            for ts in self._ui_state_proto.training_status:
                if ts.process_id == process_id:
                    training_status = ts
                    break

            if training_status is None:
                new_training_status = TrainingStatus()
                new_training_status.process_id = process_id
                self._ui_state_proto.training_status.append(new_training_status)
                # Get reference to the actual object in the list
                training_status = self._ui_state_proto.training_status[-1]

            for key, value in kwargs.items():
                if key == "config" and isinstance(value, dict):
                    training_status.config.clear()
                    training_status.config.update(
                        {k: str(v) for k, v in value.items()}
                    )
                elif key == "metrics" and isinstance(value, dict):
                    training_status.metrics.clear()
                    training_status.metrics.update(
                        {k: str(v) for k, v in value.items()}
                    )
                elif hasattr(training_status, key):
                    setattr(training_status, key, value)
                else:
                    logger.warning(f"TrainingStatus does not have attribute: {key}")

            self._ui_state_proto.timestamp = int(time.time())
            self._notify_change()

        except Exception as e:
            logger.error(f"Failed to update training progress: {e}")

    def update_device_status(self, device_uuid: str, **kwargs):
        try:
            device_status = None
            for ds in self._ui_state_proto.devices:
                if ds.device_uuid == device_uuid:
                    device_status = ds
                    break

            if device_status is None:
                device_status = deviceStatus()
                device_status.device_uuid = device_uuid
                self._ui_state_proto.devices.append(device_status)

            for key, value in kwargs.items():
                if hasattr(device_status, key):
                    setattr(device_status, key, value)

            self._ui_state_proto.timestamp = int(time.time())
            self._notify_change()

        except Exception as e:
            logger.error(f"Failed to update device status: {e}")