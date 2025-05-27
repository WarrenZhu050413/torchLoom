"""
Simplified message handlers for the torchLoom Weaver.

This module contains consolidated handlers for processing messages sent TO the weaver from different sources:
- ThreadletHandler: Process messages from threadlets/training processes
- ExternalHandler: Process messages from external monitoring systems  
- UIHandler: Process commands from the UI
"""

import logging
import time
from typing import Dict, Optional, Set

from torchLoom.common.constants import (
    HandlerConstants,
    TimeConstants,
    torchLoomConstants,
)
from torchLoom.common.handlers import *
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

logger = setup_logger(name="handlers")


# ===========================================
# THREADLET HANDLER (Training Process -> Weaver)
# ===========================================


class ThreadletHandler(BaseHandler):
    """Simplified handler for messages from threadlets."""

    def __init__(
        self,
        status_tracker,
        heartbeat_timeout: float = TimeConstants.HEARTBEAT_TIMEOUT,
    ):
        self.status_tracker = status_tracker
        self.heartbeat_timeout = heartbeat_timeout
        self._last_heartbeats: Dict[str, float] = {}
        self._dead_replicas: Set[str] = set()

        # Initialize handler registry for internal event dispatching
        self._event_registry = HandlerRegistry("threadlet_events")
        self._register_event_handlers()

        logger.info("ThreadletHandler initialized")

    def _register_event_handlers(self) -> None:
        """Register internal event handlers using the registry system."""
        self._event_registry.register_handler(
            "register_device", self._handle_device_registration
        )
        self._event_registry.register_handler("heartbeat", self._handle_heartbeat)
        self._event_registry.register_handler(
            "training_status", self._handle_training_status
        )
        self._event_registry.register_handler(
            "device_status", self._handle_device_status
        )
        self._event_registry.register_handler("drain", self._handle_drain_event)

    async def handle(self, env: EventEnvelope) -> None:
        """Handle messages from threadlets by dispatching to specific methods."""
        payload_type = env.WhichOneof("payload")
        logger.debug(f"ThreadletHandler received message: {payload_type}")

        try:
            if self._event_registry.has_handler(payload_type):
                handler_func = self._event_registry.get_handler(payload_type)
                await handler_func(env)
            else:
                logger.warning(
                    f"ThreadletHandler: Unknown event type in envelope: {payload_type}"
                )
        except Exception as e:
            logger.exception(f"Error in ThreadletHandler.handle: {e}")

    async def _handle_device_registration(self, env: EventEnvelope) -> None:
        logger.info(
            f"[ThreadletHandler] Handling device registration for {env.register_device.device_uuid}"
        )
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
            server_id=env.register_device.device_uuid,
            status="active",
        )

    async def _handle_heartbeat(self, env: EventEnvelope) -> None:
        logger.info(
            f"[ThreadletHandler] Handling heartbeat for {env.heartbeat.replica_id}"
        )
        replica_id = env.heartbeat.replica_id
        self._last_heartbeats[replica_id] = time.time()
        if replica_id in self._dead_replicas:
            self._dead_replicas.remove(replica_id)
            self.status_tracker.update_training_progress(
                replica_id=replica_id,
                status="active",
            )

    async def _handle_training_status(self, env: EventEnvelope) -> None:
        logger.info(
            f"[ThreadletHandler] Handling training status for {env.training_status.replica_id}"
        )
        ts = env.training_status
        self.status_tracker.update_training_progress(
            replica_id=ts.replica_id,
            current_step=ts.current_step,
            step_progress=ts.step_progress,
            status=ts.status,
            last_active_step=ts.batch_idx,
        )

    async def _handle_device_status(self, env: EventEnvelope) -> None:
        logger.info(
            f"[ThreadletHandler] Handling device status for {env.device_status.device_id}"
        )
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

    async def _handle_drain_event(self, env: EventEnvelope) -> None:
        logger.info(
            f"[ThreadletHandler] Handling drain event for {env.drain.device_uuid}"
        )
        replicas = self.status_tracker.get_replicas_for_device(env.drain.device_uuid)
        for replica_id in replicas:
            self.status_tracker.update_training_progress(
                replica_id=replica_id, status="draining"
            )

    def check_dead_replicas(self) -> Set[str]:
        """Check for dead replicas based on heartbeat timeout."""
        current_time = time.time()
        newly_dead = set()
        for replica_id, last_heartbeat in list(self._last_heartbeats.items()):
            time_since_heartbeat = current_time - last_heartbeat
            if (
                time_since_heartbeat > self.heartbeat_timeout
                and replica_id not in self._dead_replicas
            ):
                newly_dead.add(replica_id)
                self._dead_replicas.add(replica_id)
                logger.warning(
                    f"[ThreadletHandler] Replica {replica_id} detected as dead (no heartbeat for {time_since_heartbeat:.1f}s)"
                )
        return newly_dead


# ===========================================
# EXTERNAL HANDLER (External Systems -> Weaver)
# ===========================================


class ExternalHandler(BaseHandler):
    """Simplified handler for messages from external systems."""

    def __init__(self, status_tracker):
        self.status_tracker = status_tracker

        # Initialize handler registry for internal event dispatching
        self._event_registry = HandlerRegistry("external_events")
        self._register_event_handlers()

        logger.info("ExternalHandler initialized")

    def _register_event_handlers(self) -> None:
        """Register internal event handlers using the registry system."""
        self._event_registry.register_handler(
            "monitored_fail", self._handle_failure_event
        )

    async def handle(self, env: EventEnvelope) -> None:
        """Handle messages from external systems by dispatching to specific methods."""
        payload_type = env.WhichOneof("payload")
        logger.debug(f"ExternalHandler received message: {payload_type}")

        try:
            if self._event_registry.has_handler(payload_type):
                handler_func = self._event_registry.get_handler(payload_type)
                await handler_func(env)
            else:
                logger.warning(
                    f"ExternalHandler: Unknown event type in envelope: {payload_type}"
                )
        except Exception as e:
            logger.exception(f"Error in ExternalHandler.handle: {e}")

    async def _handle_failure_event(self, env: EventEnvelope) -> None:
        logger.info(
            f"[ExternalHandler] Handling failure event for {env.monitored_fail.device_uuid}"
        )
        device_uuid = env.monitored_fail.device_uuid
        replica_ids = self.status_tracker.get_replicas_for_device(device_uuid)
        if replica_ids:
            for device_status in list(self.status_tracker.devices.values()):
                if device_status.server_id == device_uuid:
                    self.status_tracker.update_device_status(
                        device_id=device_status.device_id, status="failed"
                    )
            for replica_id in replica_ids:
                self.status_tracker.update_training_progress(
                    replica_id=replica_id, status="failed"
                )


# ===========================================
# UI HANDLER (UI -> Weaver)
# ===========================================


class UIHandler(BaseHandler):
    """Simplified handler for commands from the UI."""

    def __init__(self, status_tracker, weaver_publish_command_func):
        self.status_tracker = status_tracker
        self.publish_weaver_command = weaver_publish_command_func

        # Initialize handler registry for internal event dispatching
        self._event_registry = HandlerRegistry("ui_events")
        self._command_registry = HandlerRegistry("ui_commands")
        self._register_event_handlers()
        self._register_command_handlers()

        logger.info("UIHandler initialized")

    def _register_event_handlers(self) -> None:
        """Register internal event handlers using the registry system."""
        self._event_registry.register_handler("ui_command", self._handle_ui_command)
        self._event_registry.register_handler(
            "config_info", self._handle_configuration_change
        )

    def _register_command_handlers(self) -> None:
        """Register UI command handlers using the registry system."""
        for command_type in HandlerConstants.UI_COMMAND_TYPES:
            method_name = f"_handle_{command_type}"
            if hasattr(self, method_name):
                self._command_registry.register_handler(
                    command_type, getattr(self, method_name)
                )

    async def handle(self, env: EventEnvelope) -> None:
        """Handle UI command events by dispatching to specific methods."""
        payload_type = env.WhichOneof("payload")
        logger.debug(f"UIHandler received message: {payload_type}")

        try:
            if self._event_registry.has_handler(payload_type):
                handler_func = self._event_registry.get_handler(payload_type)
                await handler_func(env)
            else:
                logger.warning(
                    f"UIHandler: Unknown event type in envelope: {payload_type}"
                )
        except Exception as e:
            logger.exception(f"Error in UIHandler.handle: {e}")

    async def _handle_ui_command(self, env: EventEnvelope) -> None:
        """Handle UI command events."""
        ui_command = env.ui_command
        command_type = ui_command.command_type
        target_id = ui_command.target_id
        params = dict(ui_command.params)

        logger.info(
            f"[UIHandler] Processing command: {command_type} for {target_id} with params: {params}"
        )

        if self._command_registry.has_handler(command_type):
            handler_func = self._command_registry.get_handler(command_type)
            await handler_func(target_id, params)
        else:
            logger.warning(f"[UIHandler] Unknown command type: {command_type}")

    async def _handle_deactivate_device(self, device_id: str, params: Dict = None):
        logger.info(f"[UIHandler] Handling deactivate_device for {device_id}")
        if device_id in self.status_tracker.devices:
            replica_id = self.status_tracker.devices[device_id].replica_id
            self.status_tracker.update_training_progress(
                replica_id, status="deactivating"
            )
            self.status_tracker.deactivate_device(device_id)
            await self.publish_weaver_command("pause", replica_id)

    async def _handle_reactivate_group(self, replica_id: str, params: Dict = None):
        logger.info(f"[UIHandler] Handling reactivate_group for {replica_id}")
        self.status_tracker.update_training_progress(replica_id, status="activating")
        await self.publish_weaver_command("resume", replica_id)

    async def _handle_update_config(self, replica_id: str, params: Dict):
        logger.info(
            f"[UIHandler] Handling update_config for {replica_id} with {params}"
        )
        for device_id, device_status in self.status_tracker.devices.items():
            if device_status.replica_id == replica_id:
                if hasattr(device_status, "config") and isinstance(
                    device_status.config, dict
                ):
                    device_status.config.update(params)
                else:
                    setattr(device_status, "config", params)

        await self.publish_weaver_command("update_config", replica_id, params)

    async def _handle_pause_training(self, replica_id: str, params: Dict = None):
        logger.info(f"[UIHandler] Handling pause_training for {replica_id}")
        self.status_tracker.update_training_progress(replica_id, status="paused")
        await self.publish_weaver_command("pause", replica_id)

    async def _handle_resume_training(self, replica_id: str, params: Dict = None):
        logger.info(f"[UIHandler] Handling resume_training for {replica_id}")
        self.status_tracker.update_training_progress(replica_id, status="training")
        await self.publish_weaver_command("resume", replica_id)

    async def _handle_configuration_change(self, env: EventEnvelope) -> None:
        """Handle configuration change events from UI."""
        logger.info(
            f"[UIHandler] Handling configuration change: {env.config_info.config_params}"
        )
        config_params = dict(env.config_info.config_params)
        for device_id in list(self.status_tracker.devices.keys()):
            self.status_tracker.update_device_config(device_id, config_params)

        logger.info(f"Applied configuration changes to all devices: {config_params}")


# ===========================================
# UTILITY FUNCTIONS
# ===========================================


def create_weaver_message_registry():
    """Create and configure the main message handler registry for Weaver."""
    registry = HandlerRegistry("weaver_main")

    # Register event type mappings
    registry._event_type_mapping.update(
        {
            # Threadlet events
            **{event: "threadlet" for event in HandlerConstants.THREADLET_EVENTS},
            # External events
            **{event: "external" for event in HandlerConstants.EXTERNAL_EVENTS},
            # UI events
            **{event: "ui" for event in HandlerConstants.UI_EVENTS},
        }
    )

    return registry
