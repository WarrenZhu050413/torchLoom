"""
Individual handler functions for the torchLoom Weaver.

This module contains individual handler functions that can be registered
with the HandlerRegistry in weaver.py for processing different types of events.
"""

import logging
import time
from typing import Dict, Optional, Set

from torchLoom.common.constants import Config, TimeConstants
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
        env.register_device.device_uuid, env.register_device.process_id
    )
    status_tracker.add_replica_device_mapping(
        env.register_device.process_id, env.register_device.device_uuid
    )
    # Update training progress
    training_update_kwargs = {
        "process_id": env.register_device.process_id,
        "status": "registered",  # Conceptual status
    }
    status_tracker.update_training_progress(**training_update_kwargs)

    # Update device status
    device_update_kwargs = {
        "device_id": env.register_device.device_uuid,
        "process_id": env.register_device.process_id, # process_id is part of deviceStatus proto
        "server_id": env.register_device.device_uuid, # Assuming device_uuid from RegisterDevice is the server_id
        "status": "active",  # Conceptual status
    }
    status_tracker.update_device_status(**device_update_kwargs)


async def handle_heartbeat(
    env: EventEnvelope, status_tracker, heartbeat_tracker, **kwargs
) -> None:
    """Handle heartbeat from threadlets."""
    process_id = env.heartbeat.process_id
    heartbeat_status = env.heartbeat.status # Get status from heartbeat proto
    logger.info(f"Handling heartbeat for {process_id} with reported status: '{heartbeat_status}'")
    
    heartbeat_tracker["last_heartbeats"][process_id] = time.time()

    # Default to the heartbeat status, or "active" if heartbeat status is empty
    # This ensures we always have a meaningful status to set.
    current_replica_status = heartbeat_status if heartbeat_status else "active"

    if process_id in heartbeat_tracker["dead_replicas"]:
        heartbeat_tracker["dead_replicas"].remove(process_id)
        logger.info(f"Replica {process_id} revived by heartbeat. Setting status to '{current_replica_status}'.")
    # else, replica is already considered alive, just update its status if reported by heartbeat
    # No, always update, because the status from heartbeat (e.g. training, idle) is the source of truth.

    training_update_kwargs = {
        "process_id": process_id,
        "status": current_replica_status, # Use status from heartbeat or default to "active"
    }
    status_tracker.update_training_progress(**training_update_kwargs)


async def handle_training_status(env: EventEnvelope, status_tracker, **kwargs) -> None:
    """Handle training status updates from threadlets."""
    logger.info(f"Handling training status for {env.training_status.process_id}")
    ts = env.training_status
    
    # Prepare kwargs for status_tracker.update_training_progress
    update_kwargs = {
        "process_id": ts.process_id,
        "current_step": ts.current_step,
        "epoch": ts.epoch,
        "training_time": ts.training_time,
        "max_step": ts.max_step,
        "max_epoch": ts.max_epoch,
        "metrics": dict(ts.metrics), # Convert protobuf map to dict
        "config": dict(ts.config),   # Convert protobuf map to dict
        # Conceptual fields, not directly on TrainingStatus proto
        "last_active_step": ts.current_step, 
    }
    
    # Calculate step progress if we have max_step
    if ts.max_step > 0:
        update_kwargs["step_progress"] = float(ts.current_step) / float(ts.max_step)
    
    # Use a default status since it's no longer in the protobuf
    update_kwargs["status"] = "training" 
            
    status_tracker.update_training_progress(**update_kwargs)


async def handle_device_status(env: EventEnvelope, status_tracker, **kwargs) -> None:
    """Handle device status updates from threadlets."""
    logger.info(f"Handling device status for {env.device_status.device_id}")
    ds = env.device_status
    
    update_kwargs = {
        "device_id": ds.device_id,
        "process_id": ds.process_id,
        "server_id": ds.server_id,
        "utilization": ds.utilization,
        "temperature": ds.temperature,
        "memory_used": ds.memory_used,
        "memory_total": ds.memory_total,
        "status": "active" # Conceptual status for a device reporting metrics
    }
    status_tracker.update_device_status(**update_kwargs)


# ===========================================
# EXTERNAL EVENT HANDLERS
# ===========================================


async def handle_monitored_fail(env: EventEnvelope, status_tracker, **kwargs) -> None:
    """Handle failure events from external monitoring systems."""
    logger.info(f"Handling failure event for {env.monitored_fail.device_uuid}")
    device_uuid = env.monitored_fail.device_uuid # This is likely the server_id that failed
    process_ids_on_failed_server = set()

    # Update device status for all devices associated with the failed server_id
    for dev_id, device_entry in status_tracker.devices.items():
        if device_entry.server_id == device_uuid:
            device_update_kwargs = {
                "device_id": dev_id,
                "status": "failed" # Conceptual status
            }
            status_tracker.update_device_status(**device_update_kwargs)
            # Collect process_ids associated with these devices
            # Assuming get_replicas_for_device uses device_id
            process_ids_on_failed_server.update(status_tracker.get_replicas_for_device(dev_id))


    # Update training progress for affected replicas
    # This can be broader than just replicas on the device_uuid if it's a server_id
    # The previous logic used get_replicas_for_device(device_uuid) which might be too narrow if device_uuid is a server.
    # Using process_ids_on_failed_server collected above is more accurate.
    if not process_ids_on_failed_server: # Fallback or broaden scope if needed
        logger.warning(f"No specific replicas found on server {device_uuid}, checking all replicas associated with it if it was a device ID.")
        process_ids_on_failed_server = status_tracker.get_replicas_for_device(device_uuid)


    for process_id in process_ids_on_failed_server:
        training_update_kwargs = {
            "process_id": process_id,
            "status": "failed" # Conceptual status
        }
        status_tracker.update_training_progress(**training_update_kwargs)


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
    # The deviceStatus proto doesn't have process_id directly.
    # We need to find the process_id(s) associated with this device_id.
    process_ids_for_device = status_tracker.get_replicas_for_device(device_id)
    if process_ids_for_device:
        for process_id in process_ids_for_device: # Should typically be one for this logic stream
            training_update_kwargs = {
                "process_id": process_id,
                "status": "deactivating" # Conceptual status
            }
            status_tracker.update_training_progress(**training_update_kwargs)
            await weaver_publish_command_func("pause", process_id)
    else:
        logger.warning(f"No process_id found for device_id {device_id} during deactivation.")
    
    # The deactivate_device method in status_tracker is conceptual as 'status' is not on deviceStatus proto
    # It currently logs a warning. If specific fields on deviceStatus proto needed update, use update_device_status.
    status_tracker.deactivate_device(device_id)


async def handle_reactivate_group(
    process_id: str, params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle reactivate group UI command."""
    logger.info(f"Handling reactivate_group for {process_id}")
    training_update_kwargs = {
        "process_id": process_id,
        "status": "activating" # Conceptual status
    }
    status_tracker.update_training_progress(**training_update_kwargs)
    await weaver_publish_command_func("resume", process_id)


async def handle_update_config(
    process_id: str, params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle update config UI command."""
    logger.info(f"Handling update_config for {process_id} with {params}")
    
    # Update the config within the TrainingStatus for the given process_id
    # The status_tracker.update_training_progress method handles merging dicts for 'config'
    status_tracker.update_training_progress(process_id=process_id, config=params)
    
    await weaver_publish_command_func("update_config", process_id, params)


async def handle_pause_training(
    process_id: str, params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle pause training UI command."""
    logger.info(f"Handling pause_training for {process_id}")
    training_update_kwargs = {
        "process_id": process_id,
        "status": "paused" # Conceptual status
    }
    status_tracker.update_training_progress(**training_update_kwargs)
    await weaver_publish_command_func("pause", process_id)


async def handle_resume_training(
    process_id: str, params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle resume training UI command."""
    logger.info(f"Handling resume_training for {process_id}")
    training_update_kwargs = {
        "process_id": process_id,
        "status": "training" # Conceptual status
    }
    status_tracker.update_training_progress(**training_update_kwargs)
    await weaver_publish_command_func("resume", process_id)


async def handle_drain_device(
    device_id: str, params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle drain device UI command."""
    logger.info(f"Handling drain_device for {device_id}")
    reason = params.get("reason", "User initiated")

    # Mark device as draining (conceptual status)
    device_update_kwargs = {
        "device_id": device_id,
        "status": "draining" # Conceptual status
    }
    status_tracker.update_device_status(**device_update_kwargs)

    # Get replica associated with this device and pause training
    process_ids = status_tracker.get_replicas_for_device(device_id)
    for process_id in process_ids:
        training_update_kwargs = {
            "process_id": process_id,
            "status": "draining" # Conceptual status
        }
        status_tracker.update_training_progress(**training_update_kwargs)
        await weaver_publish_command_func("pause", process_id)

    logger.info(f"Device {device_id} marked for draining. Reason: {reason}")


async def handle_global_config(
    params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle global configuration changes that affect all devices."""
    logger.info(f"Handling global configuration change with params: {params}")

    # Apply configuration changes to the 'config' field of all TrainingStatus entries
    # This assumes a global config change implies updating the config for all active replicas.
    all_process_ids = set()
    for ts_entry in status_tracker.get_ui_status_snapshot().training_status:
        all_process_ids.add(ts_entry.process_id)

    for process_id in all_process_ids:
        status_tracker.update_training_progress(process_id=process_id, config=params)
        # Optionally, if this should also trigger a command to threadlets:
        # await weaver_publish_command_func("update_config", process_id, params)

    logger.info(f"Applied global configuration changes to all replicas: {params}")


# ===========================================
# UTILITY FUNCTIONS
# ===========================================

def check_dead_replicas(
    heartbeat_tracker, heartbeat_timeout: float = TimeConstants.HEARTBEAT_TIMEOUT
) -> Set[str]:
    """Check for dead replicas based on heartbeat timeout."""
    current_time = time.time()
    newly_dead = set()

    for process_id, last_heartbeat in list(
        heartbeat_tracker["last_heartbeats"].items()
    ):
        time_since_heartbeat = current_time - last_heartbeat
        if (
            time_since_heartbeat > heartbeat_timeout
            and process_id not in heartbeat_tracker["dead_replicas"]
        ):
            newly_dead.add(process_id)
            heartbeat_tracker["dead_replicas"].add(process_id)
            logger.warning(
                f"Replica {process_id} detected as dead (no heartbeat for {time_since_heartbeat:.1f}s)"
            )

    return newly_dead
