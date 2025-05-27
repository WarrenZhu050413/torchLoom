"""
Simplified message handlers for the torchLoom Weaver.

This module contains consolidated handlers for processing messages sent TO the weaver from different sources:
- ThreadletHandler: Process messages from threadlets/training processes
- ExternalHandler: Process messages from external monitoring systems  
- UIHandler: Process commands from the UI
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Set

from torchLoom.common.constants import torchLoomConstants
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

logger = setup_logger(name="handlers")


class MessageHandler(ABC):
    """Abstract base class for message handlers."""

    @abstractmethod
    async def handle(self, env: EventEnvelope) -> None:
        """Handle a specific type of message."""
        pass


# ===========================================
# THREADLET HANDLER (Training Process -> Weaver)
# ===========================================


class ThreadletHandler(MessageHandler):
    """Simplified handler for messages from threadlets."""

    def __init__(
        self,
        status_tracker,
        heartbeat_timeout: float = 90.0,
    ):
        self.status_tracker = status_tracker
        self.heartbeat_timeout = heartbeat_timeout
        self._last_heartbeats: Dict[str, float] = {}
        self._dead_replicas: Set[str] = set()
        logger.info("ThreadletHandler initialized (simplified)")

    async def handle(self, env: EventEnvelope) -> None:
        """Handle messages from threadlets by dispatching to specific methods."""
        logger.debug(f"ThreadletHandler received message: {env.WhichOneof('payload')}")
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
            else:
                logger.warning(
                    f"ThreadletHandler: Unknown event type in envelope: {env.WhichOneof('payload')}"
                )
        except Exception as e:
            logger.exception(f"Error in ThreadletHandler.handle: {e}")

    async def _handle_device_registration(self, env: EventEnvelope) -> None:
        logger.info(
            f"[ThreadletHandler] Handling device registration for {env.register_device.device_uuid}"
        )
        # Minimal logic:
        # Update mappings using the status tracker
        self.status_tracker.add_device_replica_mapping(
            env.register_device.device_uuid, env.register_device.replica_id
        )
        self.status_tracker.add_replica_device_mapping(
            env.register_device.replica_id, env.register_device.device_uuid
        )
        # Update status tracker (basic registration)
        self.status_tracker.update_training_progress(
            replica_id=env.register_device.replica_id,
            status="registered",
        )
        self.status_tracker.update_device_status(
            device_id=env.register_device.device_uuid,
            replica_id=env.register_device.replica_id,
            server_id=env.register_device.device_uuid,  # Assuming server_id is device_uuid for simplicity here
            status="active",
        )
        pass

    async def _handle_heartbeat(self, env: EventEnvelope) -> None:
        logger.info(
            f"[ThreadletHandler] Handling heartbeat for {env.heartbeat.replica_id}"
        )
        # Minimal logic:
        replica_id = env.heartbeat.replica_id
        self._last_heartbeats[replica_id] = time.time()
        if replica_id in self._dead_replicas:
            self._dead_replicas.remove(replica_id)
            self.status_tracker.update_training_progress(
                replica_id=replica_id,
                status="active",  # Or "training" if that's the typical post-heartbeat state
            )
        pass

    async def _handle_training_status(self, env: EventEnvelope) -> None:
        logger.info(
            f"[ThreadletHandler] Handling training status for {env.training_status.replica_id}"
        )
        # Minimal logic:
        ts = env.training_status
        self.status_tracker.update_training_progress(
            replica_id=ts.replica_id,
            current_step=ts.current_step,
            step_progress=ts.step_progress,
            status=ts.status,
            last_active_step=ts.batch_idx,
        )
        pass

    async def _handle_device_status(self, env: EventEnvelope) -> None:
        logger.info(
            f"[ThreadletHandler] Handling device status for {env.device_status.device_id}"
        )
        # Minimal logic:
        ds = env.device_status
        self.status_tracker.update_device_status(
            device_id=ds.device_id,
            replica_id=ds.replica_id,
            server_id=ds.server_id,
            status=ds.status,
            utilization=ds.utilization,
            temperature=ds.temperature,
            memory_used=ds.memory_used,
            memory_total=ds.memory_total,
            config=dict(ds.config),
        )
        pass

    async def _handle_drain_event(self, env: EventEnvelope) -> None:
        logger.info(
            f"[ThreadletHandler] Handling drain event for {env.drain.device_uuid}"
        )
        # Minimal logic:
        replicas = self.status_tracker.get_replicas_for_device(env.drain.device_uuid)
        for replica_id in replicas:
            self.status_tracker.update_training_progress(
                replica_id=replica_id, status="draining"
            )
        pass

    def check_dead_replicas(self) -> Set[str]:
        """Simplified check for dead replicas. Actual logic moved to Weaver's heartbeat monitor task."""
        # This method is called by Weaver's heartbeat monitor.
        # It should primarily rely on its internal state (_last_heartbeats, _dead_replicas)
        # The status_tracker update should happen in the caller (Weaver) if a replica is newly dead.
        current_time = time.time()
        newly_dead = set()
        for replica_id, last_heartbeat in list(
            self._last_heartbeats.items()
        ):  # list() for safe iteration if modifying
            time_since_heartbeat = current_time - last_heartbeat
            if (
                time_since_heartbeat > self.heartbeat_timeout
                and replica_id not in self._dead_replicas
            ):
                newly_dead.add(replica_id)
                self._dead_replicas.add(replica_id)  # Mark as dead internally
                logger.warning(
                    f"[ThreadletHandler] Replica {replica_id} detected as dead (no heartbeat for {time_since_heartbeat:.1f}s)"
                )
                # StatusTracker update is now handled by the caller (Weaver's heartbeat monitor)
        return newly_dead


# ===========================================
# EXTERNAL HANDLER (External Systems -> Weaver)
# ===========================================


class ExternalHandler(MessageHandler):
    """Simplified handler for messages from external systems."""

    def __init__(
        self,
        status_tracker,
    ):
        self.status_tracker = status_tracker
        logger.info("ExternalHandler initialized (simplified)")

    async def handle(self, env: EventEnvelope) -> None:
        """Handle messages from external systems by dispatching to specific methods."""
        logger.debug(f"ExternalHandler received message: {env.WhichOneof('payload')}")
        try:
            if env.HasField("monitored_fail"):
                await self._handle_failure_event(env)
            else:
                logger.warning(
                    f"ExternalHandler: Unknown event type in envelope: {env.WhichOneof('payload')}"
                )
        except Exception as e:
            logger.exception(f"Error in ExternalHandler.handle: {e}")

    async def _handle_failure_event(self, env: EventEnvelope) -> None:
        logger.info(
            f"[ExternalHandler] Handling failure event for {env.monitored_fail.device_uuid}"
        )
        # Minimal logic:
        device_uuid = env.monitored_fail.device_uuid
        replica_ids = self.status_tracker.get_replicas_for_device(device_uuid)
        if replica_ids:
            for device_status in list(
                self.status_tracker.devices.values()
            ):  # Iterate over a copy
                if device_status.server_id == device_uuid:
                    self.status_tracker.update_device_status(
                        device_id=device_status.device_id, status="failed"
                    )
            for replica_id in replica_ids:
                self.status_tracker.update_training_progress(
                    replica_id=replica_id, status="failed"
                )
        # Publishing REPLICA_FAIL is removed from here. Weaver would do it if necessary.
        pass


# ===========================================
# UI HANDLER (UI -> Weaver)
# ===========================================


class UIHandler(MessageHandler):
    """Simplified handler for commands from the UI."""

    def __init__(
        self, status_tracker, weaver_publish_command_func
    ):  # Removed nats_client, added callback for publishing
        self.status_tracker = status_tracker
        self.publish_weaver_command = (
            weaver_publish_command_func  # Callback to Weaver's publisher
        )
        logger.info("UIHandler initialized (simplified)")

    async def handle(self, env: EventEnvelope) -> None:
        """Handle UI command events by dispatching to specific methods."""
        logger.debug(f"UIHandler received message: {env.WhichOneof('payload')}")
        try:
            if env.HasField("ui_command"):
                await self._handle_ui_command(env)
            elif env.HasField("config_info"):
                await self._handle_configuration_change(env)
            else:
                logger.warning(
                    f"UIHandler: Unknown event type in envelope: {env.WhichOneof('payload')}"
                )
        except Exception as e:
            logger.exception(f"Error in UIHandler.handle: {e}")

    async def _handle_ui_command(self, env: EventEnvelope) -> None:
        """Handle UI command events. Publishing commands is now done via callback to Weaver."""
        ui_command = env.ui_command
        command_type = ui_command.command_type
        target_id = ui_command.target_id
        params = dict(ui_command.params)

        logger.info(
            f"[UIHandler] Processing command: {command_type} for {target_id} with params: {params}"
        )

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
            logger.warning(f"[UIHandler] Unknown command type: {command_type}")
        pass

    # _publish_weaver_command is removed and replaced by a callback

    async def _handle_deactivate_device(self, device_id: str):
        logger.info(f"[UIHandler] Handling deactivate_device for {device_id}")
        # Minimal status update
        # Actual command publishing done via callback
        if device_id in self.status_tracker.devices:
            replica_id = self.status_tracker.devices[device_id].replica_id
            self.status_tracker.update_training_progress(
                replica_id, status="deactivating"
            )
            self.status_tracker.deactivate_device(
                device_id
            )  # This method should exist in StatusTracker
            await self.publish_weaver_command("pause", replica_id)
        pass

    async def _handle_reactivate_group(self, replica_id: str):
        logger.info(f"[UIHandler] Handling reactivate_group for {replica_id}")
        # Minimal status update
        self.status_tracker.update_training_progress(replica_id, status="activating")
        # self.status_tracker.reactivate_replica_group(replica_id) # This method should exist
        await self.publish_weaver_command("resume", replica_id)
        pass

    async def _handle_update_config(self, replica_id: str, params: Dict[str, str]):
        logger.info(
            f"[UIHandler] Handling update_config for {replica_id} with {params}"
        )
        # Minimal status update (config is usually part of device status or a specific config store)
        # For devices in this replica_id, update their config in status_tracker
        for device_id, device_status in self.status_tracker.devices.items():
            if device_status.replica_id == replica_id:
                # Assuming device_status.config is a dict and can be updated
                if hasattr(device_status, "config") and isinstance(
                    device_status.config, dict
                ):
                    device_status.config.update(params)
                else:  # Or set it if it doesn't exist or not a dict
                    setattr(device_status, "config", params)

        await self.publish_weaver_command("update_config", replica_id, params)
        pass

    async def _handle_pause_training(self, replica_id: str):
        logger.info(f"[UIHandler] Handling pause_training for {replica_id}")
        self.status_tracker.update_training_progress(replica_id, status="paused")
        await self.publish_weaver_command("pause", replica_id)
        pass

    async def _handle_resume_training(self, replica_id: str):
        logger.info(f"[UIHandler] Handling resume_training for {replica_id}")
        self.status_tracker.update_training_progress(
            replica_id, status="training"
        )  # Or "active"
        await self.publish_weaver_command("resume", replica_id)
        pass

    async def _handle_configuration_change(self, env: EventEnvelope) -> None:
        """Handle configuration change events from UI."""
        logger.info(
            f"[UIHandler] Handling configuration change: {env.config_info.config_params}"
        )
        # Apply configuration changes to all devices
        config_params = dict(env.config_info.config_params)
        for device_id in list(self.status_tracker.devices.keys()):
            self.status_tracker.update_device_config(device_id, config_params)
        
        # Optionally publish configuration updates to threadlets
        # This could be done via the weaver command publisher if needed
        logger.info(f"Applied configuration changes to all devices: {config_params}")
        pass


# ===========================================
# UTILITY CLASSES
# ===========================================

# DeviceReplicaMapper functionality has been moved to StatusTracker
