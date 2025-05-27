"""
Individual handler functions for the torchLoom Weaver.

This module contains individual handler functions that can be registered
with the HandlerRegistry in weaver.py for processing different types of events.
"""

import logging
import time
from typing import Dict, Optional, Set

from torchLoom.common.constants import TimeConstants
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

logger = setup_logger(name="handlers")


# ===========================================
# THREADLET EVENT HANDLERS
# ===========================================


async def handle_device_registration(
    env: EventEnvelope, status_tracker, **kwargs
) -> None:
    """Handle device registration from threadlets."""
    logger.info(f"Handling device registration for {env.register_device.device_uuid}")
    # Update mappings using the status tracker
    status_tracker.add_device_replica_mapping(
        env.register_device.device_uuid, env.register_device.replica_id
    )
    status_tracker.add_replica_device_mapping(
        env.register_device.replica_id, env.register_device.device_uuid
    )
    # Update status tracker (basic registration)
    status_tracker.update_training_progress(
        replica_id=env.register_device.replica_id,
        status="registered",
    )
    status_tracker.update_device_status(
        device_id=env.register_device.device_uuid,
        replica_id=env.register_device.replica_id,
        server_id=env.register_device.device_uuid,
        status="active",
    )


async def handle_heartbeat(
    env: EventEnvelope, status_tracker, heartbeat_tracker, **kwargs
) -> None:
    """Handle heartbeat from threadlets."""
    logger.info(f"Handling heartbeat for {env.heartbeat.replica_id}")
    replica_id = env.heartbeat.replica_id
    heartbeat_tracker["last_heartbeats"][replica_id] = time.time()

    if replica_id in heartbeat_tracker["dead_replicas"]:
        heartbeat_tracker["dead_replicas"].remove(replica_id)
        status_tracker.update_training_progress(
            replica_id=replica_id,
            status="active",
        )


async def handle_training_status(env: EventEnvelope, status_tracker, **kwargs) -> None:
    """Handle training status updates from threadlets."""
    logger.info(f"Handling training status for {env.training_status.replica_id}")
    ts = env.training_status
    status_tracker.update_training_progress(
        replica_id=ts.replica_id,
        current_step=ts.current_step,
        step_progress=ts.step_progress,
        status=ts.status,
        last_active_step=ts.batch_idx,
    )


async def handle_device_status(env: EventEnvelope, status_tracker, **kwargs) -> None:
    """Handle device status updates from threadlets."""
    logger.info(f"Handling device status for {env.device_status.device_id}")
    ds = env.device_status
    status_tracker.update_device_status(
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


# ===========================================
# EXTERNAL EVENT HANDLERS
# ===========================================


async def handle_monitored_fail(env: EventEnvelope, status_tracker, **kwargs) -> None:
    """Handle failure events from external monitoring systems."""
    logger.info(f"Handling failure event for {env.monitored_fail.device_uuid}")
    device_uuid = env.monitored_fail.device_uuid
    replica_ids = status_tracker.get_replicas_for_device(device_uuid)

    if replica_ids:
        for device_status in list(status_tracker.devices.values()):
            if device_status.server_id == device_uuid:
                status_tracker.update_device_status(
                    device_id=device_status.device_id, status="failed"
                )
        for replica_id in replica_ids:
            status_tracker.update_training_progress(
                replica_id=replica_id, status="failed"
            )


# ===========================================
# UI EVENT HANDLERS
# ===========================================

# UI Command dispatch table
UI_COMMAND_HANDLERS = {
    "deactivate_device": "handle_deactivate_device",
    "reactivate_group": "handle_reactivate_group", 
    "update_config": "handle_update_config",
    "global_config": "handle_global_config",
    "pause_training": "handle_pause_training",
    "resume_training": "handle_resume_training",
    "drain": "handle_drain_device",
}


async def handle_ui_command(
    env: EventEnvelope, status_tracker, weaver_publish_command_func, **kwargs
) -> None:
    """Handle all UI commands - everything comes through ui_command now."""
    
    # Handle ui_command events only
    if env.HasField("ui_command"):
        ui_command = env.ui_command
        command_type = ui_command.command_type
        target_id = ui_command.target_id
        params = dict(ui_command.params)

        logger.info(
            f"Processing UI command: {command_type} for {target_id} with params: {params}"
        )

        # Use dispatch table to find handler
        handler_name = UI_COMMAND_HANDLERS.get(command_type)
        if handler_name:
            # Get the handler function from the current module
            handler_func = globals().get(handler_name)
            if handler_func:
                await handler_func(
                    target_id, params, status_tracker, weaver_publish_command_func
                )
            else:
                logger.error(f"Handler function {handler_name} not found in module")
        else:
            logger.warning(f"Unknown UI command type: {command_type}")
    
    else:
        logger.warning("UI handler received event with no ui_command payload")


# ===========================================
# UI COMMAND SUB-HANDLERS
# ===========================================


async def handle_deactivate_device(
    device_id: str, params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle deactivate device UI command."""
    logger.info(f"Handling deactivate_device for {device_id}")
    if device_id in status_tracker.devices:
        replica_id = status_tracker.devices[device_id].replica_id
        status_tracker.update_training_progress(replica_id, status="deactivating")
        status_tracker.deactivate_device(device_id)
        await weaver_publish_command_func("pause", replica_id)


async def handle_reactivate_group(
    replica_id: str, params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle reactivate group UI command."""
    logger.info(f"Handling reactivate_group for {replica_id}")
    status_tracker.update_training_progress(replica_id, status="activating")
    await weaver_publish_command_func("resume", replica_id)


async def handle_update_config(
    replica_id: str, params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle update config UI command."""
    logger.info(f"Handling update_config for {replica_id} with {params}")
    for device_id, device_status in status_tracker.devices.items():
        if device_status.replica_id == replica_id:
            if hasattr(device_status, "config") and isinstance(
                device_status.config, dict
            ):
                device_status.config.update(params)
            else:
                setattr(device_status, "config", params)

    await weaver_publish_command_func("update_config", replica_id, params)


async def handle_pause_training(
    replica_id: str, params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle pause training UI command."""
    logger.info(f"Handling pause_training for {replica_id}")
    status_tracker.update_training_progress(replica_id, status="paused")
    await weaver_publish_command_func("pause", replica_id)


async def handle_resume_training(
    replica_id: str, params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle resume training UI command."""
    logger.info(f"Handling resume_training for {replica_id}")
    status_tracker.update_training_progress(replica_id, status="training")
    await weaver_publish_command_func("resume", replica_id)


async def handle_drain_device(
    device_id: str, params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle drain device UI command."""
    logger.info(f"Handling drain_device for {device_id}")
    reason = params.get("reason", "User initiated")

    # Mark device as draining
    status_tracker.update_device_status(device_id=device_id, status="draining")

    # Get replica associated with this device and pause training
    replica_ids = status_tracker.get_replicas_for_device(device_id)
    for replica_id in replica_ids:
        status_tracker.update_training_progress(
            replica_id=replica_id, status="draining"
        )
        await weaver_publish_command_func("pause", replica_id)

    logger.info(f"Device {device_id} marked for draining. Reason: {reason}")


async def handle_global_config(
    params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle global configuration changes that affect all devices."""
    logger.info(f"Handling global configuration change with params: {params}")

    # Apply configuration changes to all devices
    for device_id in list(status_tracker.devices.keys()):
        status_tracker.update_device_config(device_id, params)

    logger.info(f"Applied global configuration changes to all devices: {params}")


# ===========================================
# UTILITY FUNCTIONS
# ===========================================


def check_dead_replicas(
    heartbeat_tracker, heartbeat_timeout: float = TimeConstants.HEARTBEAT_TIMEOUT
) -> Set[str]:
    """Check for dead replicas based on heartbeat timeout."""
    current_time = time.time()
    newly_dead = set()

    for replica_id, last_heartbeat in list(
        heartbeat_tracker["last_heartbeats"].items()
    ):
        time_since_heartbeat = current_time - last_heartbeat
        if (
            time_since_heartbeat > heartbeat_timeout
            and replica_id not in heartbeat_tracker["dead_replicas"]
        ):
            newly_dead.add(replica_id)
            heartbeat_tracker["dead_replicas"].add(replica_id)
            logger.warning(
                f"Replica {replica_id} detected as dead (no heartbeat for {time_since_heartbeat:.1f}s)"
            )

    return newly_dead
