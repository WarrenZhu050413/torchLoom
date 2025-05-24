"""
Message handlers for the torchLoom Weaver.

This module contains consolidated handlers for processing messages sent TO the weaver from different sources:
- WeaveletHandler: Process messages from weavelets/training processes
- ExternalHandler: Process messages from external monitoring systems  
- UIHandler: Process commands from the UI
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Set

from torchLoom.common.constants import torchLoomConstants
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope, MonitoredFailEvent

logger = setup_logger(name="handlers")


class MessageHandler(ABC):
    """Abstract base class for message handlers."""

    @abstractmethod
    async def handle(self, env: EventEnvelope) -> None:
        """Handle a specific type of message."""
        pass


# ===========================================
# WEAVELET HANDLER (Training Process -> Weaver)
# ===========================================


class WeaveletHandler(MessageHandler):
    """Consolidated handler for all messages from weavelets/training processes to weaver."""

    def __init__(
        self,
        device_mapper: "DeviceReplicaMapper",
        status_tracker,
        nats_client=None,
        heartbeat_timeout: float = 90.0,
    ):
        self.device_mapper = device_mapper
        self.status_tracker = status_tracker
        self.nats_client = nats_client
        self.heartbeat_timeout = heartbeat_timeout

        # Heartbeat tracking
        self._last_heartbeats: Dict[str, float] = (
            {}
        )  # replica_id -> last_heartbeat_timestamp
        self._dead_replicas: Set[str] = set()  # Track replicas that are considered dead

    async def handle(self, env: EventEnvelope) -> None:
        """Handle messages from weavelets."""
        try:
            if env.HasField("register_device"):
                await self._handle_device_registration(env)
            elif env.HasField("heartbeat"):
                await self._handle_heartbeat(env)
            elif env.HasField("training_status"):
                await self._handle_training_status(env)
            elif env.HasField("device_status"):
                await self._handle_device_status(env)
            elif env.HasField("drain"):
                await self._handle_drain_event(env)
        except Exception as e:
            logger.exception(f"Error in WeaveletHandler: {e}")

    async def _handle_device_registration(self, env: EventEnvelope) -> None:
        """Handle device registration events from weavelets."""
        device_uuid: str = env.register_device.device_uuid
        replica_id: str = env.register_device.replica_id

        logger.info("\n" + "-" * 100)
        logger.info(
            f"[WEAVELET->WEAVER] Device registration: {device_uuid} -> {replica_id}"
        )

        # Update mappings using the device mapper
        device_added = self.device_mapper.add_device_replica_mapping(
            device_uuid, replica_id
        )
        replica_added = self.device_mapper.add_replica_device_mapping(
            replica_id, device_uuid
        )

        if device_added:
            logger.info(
                f"New device mapping: {device_uuid} -> {self.device_mapper.get_replicas_for_device(device_uuid)}"
            )

        if replica_added:
            logger.info(
                f"New replica mapping: {replica_id} -> {self.device_mapper.get_devices_for_replica(replica_id)}"
            )

        # Update status tracker to make the device/replica visible in the UI
        if self.status_tracker:
            # Add the replica to status tracker if it's new
            if replica_added:
                self.status_tracker.update_training_progress(
                    replica_id=replica_id,
                    current_step=0,
                    step_progress=0.0,
                    status="registered",
                )
                logger.info(
                    f"Added replica {replica_id} to status tracker with status 'registered'"
                )

            # Create a default device entry for this device if it doesn't exist
            default_device_id = f"{device_uuid}"
            self.status_tracker.update_device_status(
                device_id=device_uuid,
                replica_id=replica_id,
                server_id=device_uuid,
                status="active",
                utilization=0.0,
                temperature=40.0,
                memory_used=0.0,
                memory_total=8.0,
            )
            logger.info(
                f"Added default device {default_device_id} for device {device_uuid} to status tracker"
            )

    async def _handle_heartbeat(self, env: EventEnvelope) -> None:
        """Handle heartbeat events from weavelets."""
        heartbeat = env.heartbeat
        replica_id = heartbeat.replica_id
        current_time = time.time()

        # Update last heartbeat time
        self._last_heartbeats[replica_id] = current_time

        # If this replica was considered dead, mark it as alive again
        if replica_id in self._dead_replicas:
            self._dead_replicas.remove(replica_id)
            logger.info(
                f"[WEAVELET->WEAVER] Replica {replica_id} is alive again (received heartbeat)"
            )
            self.status_tracker.update_training_progress(
                replica_id=replica_id, status="active"
            )

        logger.debug(
            f"[WEAVELET->WEAVER] Heartbeat from {replica_id}, status: {heartbeat.status}"
        )

    async def _handle_training_status(self, env: EventEnvelope) -> None:
        """Handle training status events from weavelets."""
        training_status = env.training_status

        # Update training progress in status tracker
        self.status_tracker.update_training_progress(
            replica_id=training_status.replica_id,
            current_step=training_status.current_step,
            step_progress=training_status.step_progress,
            status=training_status.status,
            last_active_step=training_status.batch_idx,
            fixed_step=None,
        )

        logger.debug(
            f"[WEAVELET->WEAVER] Training status: {training_status.replica_id} - {training_status.status_type}"
        )

    async def _handle_device_status(self, env: EventEnvelope) -> None:
        """Handle device status events from weavelets."""
        device_status = env.device_status

        # Convert protobuf config map to dict
        config_dict = dict(device_status.config) if device_status.config else {}

        # Update device status in status tracker
        self.status_tracker.update_device_status(
            device_id=device_status.device_id,
            replica_id=device_status.replica_id,
            server_id=device_status.server_id,
            status=device_status.status,
            utilization=device_status.utilization,
            temperature=device_status.temperature,
            memory_used=device_status.memory_used,
            memory_total=device_status.memory_total,
            config=config_dict,
        )

        logger.debug(
            f"[WEAVELET->WEAVER] device status: {device_status.device_id} - {device_status.status}"
        )

    async def _handle_drain_event(self, env: EventEnvelope) -> None:
        """Handle drain events from weavelets."""
        device_uuid = env.drain.device_uuid

        logger.info("\n" + "-" * 100)
        logger.info(f"[WEAVELET->WEAVER] Drain event for device: {device_uuid}")

        # Get all replicas associated with this device
        replicas = self.device_mapper.get_replicas_for_device(device_uuid)

        # Update status for all affected replicas
        for replica_id in replicas:
            self.status_tracker.update_training_progress(
                replica_id=replica_id, status="draining"
            )
            logger.info(f"Set replica {replica_id} status to 'draining'")

        logger.info(
            f"Processed drain event for device {device_uuid} affecting {len(replicas)} replicas"
        )

    def check_dead_replicas(self) -> Set[str]:
        """Check for newly dead replicas based on heartbeat timeout."""
        current_time = time.time()
        newly_dead = set()

        for replica_id, last_heartbeat in self._last_heartbeats.items():
            time_since_heartbeat = current_time - last_heartbeat

            # If replica hasn't sent heartbeat within timeout and wasn't already considered dead
            if (
                time_since_heartbeat > self.heartbeat_timeout
                and replica_id not in self._dead_replicas
            ):
                newly_dead.add(replica_id)
                self._dead_replicas.add(replica_id)
                logger.warning(
                    f"[WEAVER] Replica {replica_id} is considered dead (no heartbeat for {time_since_heartbeat:.1f}s)"
                )
                self.status_tracker.update_training_progress(
                    replica_id=replica_id, status="dead"
                )

        return newly_dead


# ===========================================
# EXTERNAL HANDLER (External Systems -> Weaver)
# ===========================================


class ExternalHandler(MessageHandler):
    """Consolidated handler for all messages from external monitoring systems to weaver."""

    def __init__(
        self, device_mapper: "DeviceReplicaMapper", nats_client, status_tracker=None
    ):
        self.device_mapper = device_mapper
        self.nats_client = nats_client
        self.status_tracker = status_tracker

    async def handle(self, env: EventEnvelope) -> None:
        """Handle messages from external systems."""
        try:
            if env.HasField("monitored_fail"):
                await self._handle_failure_event(env)
            elif env.HasField("config_info"):
                await self._handle_configuration_change(env)
        except Exception as e:
            logger.exception(f"Error in ExternalHandler: {e}")

    async def _handle_failure_event(self, env: EventEnvelope) -> None:
        """Handle device failure events from external monitoring systems."""
        fail_event: MonitoredFailEvent = env.monitored_fail
        device_uuid: str = fail_event.device_uuid

        replica_ids: Set[str] = self.device_mapper.get_replicas_for_device(device_uuid)
        if replica_ids:
            logger.info(f"[EXTERNAL->WEAVER] device failure detected: {device_uuid}")
            logger.info(f"[EXTERNAL->WEAVER] Associated replicas: {replica_ids}")

            # Update status tracker to reflect the failure
            if self.status_tracker:
                # Mark any devices on this device as failed
                failed_devices = [
                    device
                    for device in self.status_tracker.devices.values()
                    if device.server_id == device_uuid
                ]
                for device in failed_devices:
                    self.status_tracker.update_device_status(
                        device_id=device.device_id,
                        replica_id=device.replica_id,
                        server_id=device.server_id,
                        status="failed",
                        utilization=0.0,
                        temperature=0.0,
                    )
                    logger.info(
                        f"Marked device {device.device_id} as failed due to device failure"
                    )

                # Mark associated replicas as failed
                for replica_id in replica_ids:
                    self.status_tracker.update_training_progress(
                        replica_id=replica_id, status="failed"
                    )
                    logger.info(
                        f"Marked replica {replica_id} as failed due to device failure"
                    )

            for replica_id in replica_ids:
                await self._send_replica_fail_event(replica_id)
        else:
            logger.warning(
                f"[EXTERNAL->WEAVER] Device {device_uuid} not found in device-to-replicas map"
            )

    async def _handle_configuration_change(self, env: EventEnvelope) -> None:
        """Handle config_info change events."""
        config_params: Dict[str, str] = dict(env.config_info.config_params)

        logger.info("\n" + "-" * 100)
        logger.info(f"[CONFIG] Received config change with parameters: {config_params}")

        # Update status tracker with new configuration
        if self.status_tracker:
            # Update all devices with the new configuration
            for device in self.status_tracker.devices.values():
                device.config.update(config_params)
            logger.info(
                f"Updated configuration for {len(self.status_tracker.devices)} devices in status tracker"
            )

        try:
            if not self.nats_client:
                raise RuntimeError("NATS connection is not initialized.")

            js = self.nats_client.jetstream()

            # Publish the entire config change to a general subject
            logger.debug(
                f"Publishing config change to {torchLoomConstants.subjects.CONFIG_INFO}"
            )
            await js.publish(
                torchLoomConstants.subjects.CONFIG_INFO, env.SerializeToString()
            )
            logger.info(
                f"Published config changes to {torchLoomConstants.subjects.CONFIG_INFO}"
            )
        except Exception as e:
            logger.exception(f"Failed to publish config changes: {e}")
            raise

    async def _send_replica_fail_event(self, replica_id: str) -> None:
        """Send a replica failure event to training processes."""
        if not self.nats_client:
            raise RuntimeError("NATS connection is not initialized")

        env: EventEnvelope = EventEnvelope()
        env.replica_fail.replica_id = replica_id
        await self.nats_client.publish(
            torchLoomConstants.subjects.REPLICA_FAIL, env.SerializeToString()
        )
        logger.info(f"[WEAVER->WEAVELET] Published replica fail event for {replica_id}")


# ===========================================
# UI HANDLER (UI -> Weaver)
# ===========================================


class UIHandler(MessageHandler):
    """Consolidated handler for all commands from the UI to weaver."""

    def __init__(self, status_tracker, nats_client=None):
        self.status_tracker = status_tracker
        self.nats_client = nats_client

    async def handle(self, env: EventEnvelope) -> None:
        """Handle UI command events and execute corresponding weaver actions."""
        try:
            if env.HasField("ui_command"):
                await self._handle_ui_command(env)
        except Exception as e:
            logger.exception(f"Error in UIHandler: {e}")

    async def _handle_ui_command(self, env: EventEnvelope) -> None:
        """Handle UI command events."""
        ui_command = env.ui_command
        command_type = ui_command.command_type
        target_id = ui_command.target_id
        params = dict(ui_command.params) if ui_command.params else {}

        logger.info(f"[UI->WEAVER] Processing command: {command_type} for {target_id}")

        if command_type == "deactivate_device":
            await self._handle_deactivate_device(target_id)
        elif command_type == "reactivate_group":
            await self._handle_reactivate_group(target_id)
        elif command_type == "update_config":
            await self._handle_update_config(target_id, params)
        elif command_type == "pause_training":
            await self._handle_pause_training(target_id)
        elif command_type == "resume_training":
            await self._handle_resume_training(target_id)
        else:
            logger.warning(f"[UI->WEAVER] Unknown command type: {command_type}")

    async def _publish_weaver_command(
        self,
        command_type: str,
        target_replica_id: str,
        params: Optional[Dict[str, str]] = None,
    ):
        """Publish a weaver command to training processes."""
        try:
            if not self.nats_client:
                logger.warning("Cannot publish weaver command - no NATS client")
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
            logger.exception(f"Failed to publish weaver command: {e}")

    async def _handle_deactivate_device(self, device_id: str):
        """Handle device deactivation command from UI."""
        if device_id in self.status_tracker.devices:
            self.status_tracker.set_communication_status("rebuilding")

            # Get replica ID for the device
            replica_id = self.status_tracker.devices[device_id].replica_id
            self.status_tracker.update_training_progress(
                replica_id, status="deactivating"
            )

            # Deactivate the device
            self.status_tracker.deactivate_device(device_id)

            # Send pause command to the replica
            await self._publish_weaver_command("pause", replica_id)

            logger.info(f"[UI->WEAVER] Deactivated device: {device_id}")
        else:
            logger.warning(f"[UI->WEAVER] device not found for deactivation: {device_id}")

    async def _handle_reactivate_group(self, replica_id: str):
        """Handle replica group reactivation command from UI."""
        self.status_tracker.set_communication_status("rebuilding")
        self.status_tracker.update_training_progress(replica_id, status="activating")

        # Reactivate the replica group
        self.status_tracker.reactivate_replica_group(replica_id)

        # Send resume command to the replica
        await self._publish_weaver_command("resume", replica_id)

        logger.info(f"[UI->WEAVER] Reactivated replica group: {replica_id}")

    async def _handle_update_config(self, replica_id: str, params: Dict[str, str]):
        """Handle configuration update command from UI."""
        # Update configuration for all devices in the replica
        replica_devices = [
            g for g in self.status_tracker.devices.values() if g.replica_id == replica_id
        ]

        for device in replica_devices:
            device.config.update(params)

        # Send config update command to the replica
        await self._publish_weaver_command("update_config", replica_id, params)

        logger.info(f"[UI->WEAVER] Updated config for replica {replica_id}: {params}")

    async def _handle_pause_training(self, replica_id: str):
        """Handle pause training command from UI."""
        self.status_tracker.update_training_progress(replica_id, status="paused")
        await self._publish_weaver_command("pause", replica_id)
        logger.info(f"[UI->WEAVER] Paused training for replica: {replica_id}")

    async def _handle_resume_training(self, replica_id: str):
        """Handle resume training command from UI."""
        self.status_tracker.update_training_progress(replica_id, status="training")
        await self._publish_weaver_command("resume", replica_id)
        logger.info(f"[UI->WEAVER] Resumed training for replica: {replica_id}")


# ===========================================
# UTILITY CLASSES
# ===========================================


class DeviceReplicaMapper:
    """Manages mapping between devices and replicas."""

    def __init__(self):
        # Many-to-many mapping between devices and replicas
        self.device_to_replicas: Dict[str, Set[str]] = (
            {}
        )  # device_uuid -> set of replica_ids
        self.replica_to_devices: Dict[str, Set[str]] = (
            {}
        )  # replica_id -> set of device_uuids

    def add_device_replica_mapping(self, device_uuid: str, replica_id: str) -> bool:
        """Add a device-to-replica mapping. Returns True if this is a new association."""
        if device_uuid not in self.device_to_replicas:
            self.device_to_replicas[device_uuid] = set()

        is_new_association = replica_id not in self.device_to_replicas[device_uuid]
        if is_new_association:
            self.device_to_replicas[device_uuid].add(replica_id)

        return is_new_association

    def add_replica_device_mapping(self, replica_id: str, device_uuid: str) -> bool:
        """Add a replica-to-device mapping. Returns True if this is a new association."""
        if replica_id not in self.replica_to_devices:
            self.replica_to_devices[replica_id] = set()

        is_new_association = device_uuid not in self.replica_to_devices[replica_id]
        if is_new_association:
            self.replica_to_devices[replica_id].add(device_uuid)

        return is_new_association

    def get_replicas_for_device(self, device_uuid: str) -> Set[str]:
        """Get all replicas associated with a device."""
        return self.device_to_replicas.get(device_uuid, set())

    def get_devices_for_replica(self, replica_id: str) -> Set[str]:
        """Get all devices associated with a replica."""
        return self.replica_to_devices.get(replica_id, set())


# ===========================================
# LEGACY COMPATIBILITY (Backward compatibility for existing imports)
# ===========================================

# Aliases for backward compatibility - users can still import individual handler names
DeviceRegistrationHandler = WeaveletHandler
HeartbeatHandler = WeaveletHandler
TrainingStatusHandler = WeaveletHandler
deviceStatusHandler = WeaveletHandler
DrainEventHandler = WeaveletHandler
FailureHandler = ExternalHandler
ConfigurationHandler = ExternalHandler
UICommandHandler = UIHandler
