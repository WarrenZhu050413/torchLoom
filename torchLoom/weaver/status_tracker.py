"""
Manages the UI-related state for torchLoom Weaver using protobuf messages.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import TrainingStatus, UIStatusUpdate, deviceStatus

logger = setup_logger(name="weaver_ui_state")


@dataclass
class StatusTracker:
    """
    Manages and encapsulates UI-related information for the Weaver,
    primarily holding an instance of the UIStatusUpdate protobuf message.
    """

    _ui_state_proto: UIStatusUpdate = field(default_factory=UIStatusUpdate)
    communication_status: str = (
        "stable"  # Overall communication status with other components
    )

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

    def get_ui_status_snapshot(self) -> UIStatusUpdate:
        """
        Returns the current UIStatusUpdate protobuf message.
        Consider returning a copy if external modification is a concern:
        snapshot = UIStatusUpdate()
        snapshot.CopyFrom(self._ui_state_proto)
        return snapshot
        """
        return self._ui_state_proto

    def get_active_devices(self) -> List[deviceStatus]:
        """
        Returns a list of all devices currently marked as "active" in the UI state.
        """
        return [d for d in self._ui_state_proto.devices if d.status == "active"]

    def set_communication_status(self, status: str):
        """Updates the overall communication status string."""
        self.communication_status = status
        self._ui_state_proto.timestamp = int(time.time())  # Update timestamp on change
        logger.info(f"Communication status set to: {status}")

    def update_training_progress(self, replica_id: str, status: str, **kwargs):
        """Update training progress for a replica."""
        # Create a simple TrainingStatus proto
        training_status = TrainingStatus()
        training_status.replica_id = replica_id
        training_status.status = status
        training_status.status_type = kwargs.get("status_type", "update")
        training_status.current_step = kwargs.get("current_step", 0)
        training_status.epoch = kwargs.get("epoch", 0)

        # Update metrics if provided
        if "metrics" in kwargs:
            training_status.metrics.update(
                {k: str(v) for k, v in kwargs["metrics"].items()}
            )

        self.update_training_progress_from_proto(training_status)

    def deactivate_device(self, device_id: str):
        """Deactivate a device by setting its status to inactive."""
        for device in self._ui_state_proto.devices:
            if device.device_id == device_id:
                device.status = "inactive"
                self._ui_state_proto.timestamp = int(time.time())
                logger.info(f"Deactivated device: {device_id}")
                return
        logger.warning(f"Device {device_id} not found for deactivation")


# Example usage (optional, for demonstration or testing within this file):
if __name__ == "__main__":
    # This example assumes torchLoom_pb2.py is generated and in PYTHONPATH
    logger.info("Starting StatusTracker example...")
    ui_state_manager = StatusTracker()

    # Simulate receiving a device status update
    sample_device_proto = deviceStatus(
        device_id="gpu-001",
        replica_id="rep-a",
        server_id="server-alpha",
        status="active",
        utilization=75.5,
        temperature=65.2,
        memory_used=4.5,
        memory_total=16.0,
        config={"batch_size": "32", "lr": "0.001"},
    )
    ui_state_manager.update_device_status_from_proto(sample_device_proto)
    logger.info(f"Device gpu-001 added/updated.")

    # Simulate receiving a training progress update
    sample_training_proto = TrainingStatus(
        replica_id="rep-a",
        status_type="batch_update",
        current_step=100,
        epoch=1,
        status="training",
        metrics={"loss": "0.123", "accuracy": "0.95"},
        training_time=3600.0,
        max_step=10000,
        max_epoch=10,
    )
    ui_state_manager.update_training_progress_from_proto(sample_training_proto)
    logger.info(f"Training status for rep-a added/updated.")

    current_snapshot = ui_state_manager.get_ui_status_snapshot()
    logger.info(f"Current UI Snapshot Timestamp: {current_snapshot.timestamp}")
    for dev in current_snapshot.devices:
        logger.info(
            f"Device: {dev.device_id}, Status: {dev.status}, Util: {dev.utilization}%"
        )
    for ts in current_snapshot.training_status:
        logger.info(
            f"Replica: {ts.replica_id}, Step: {ts.current_step}, Metrics: {ts.metrics}"
        )

    time.sleep(1)
    sample_device_proto_updated = deviceStatus(
        device_id="gpu-001",
        replica_id="rep-a",
        server_id="server-alpha",
        status="active",
        utilization=80.0,
        temperature=68.0,
        memory_used=5.0,
        memory_total=16.0,
        config={"batch_size": "32", "lr": "0.001"},
    )
    ui_state_manager.update_device_status_from_proto(sample_device_proto_updated)
    logger.info(f"Device gpu-001 updated again.")

    updated_snapshot = ui_state_manager.get_ui_status_snapshot()
    logger.info(f"Updated UI Snapshot Timestamp: {updated_snapshot.timestamp}")
    for dev in updated_snapshot.devices:
        if dev.device_id == "gpu-001":
            logger.info(f"Updated Device: {dev.device_id}, Util: {dev.utilization}%")

    active_devs = ui_state_manager.get_active_devices()
    logger.info(f"Number of active devices: {len(active_devs)}")
    logger.info("StatusTracker example finished.")
