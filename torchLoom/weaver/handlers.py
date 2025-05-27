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
from torchLoom.common.utils import create_training_status_dict, create_device_status_dict

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
        "device_uuid": env.register_device.device_uuid,
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
    
    update_kwargs = create_training_status_dict(ts)
            
    status_tracker.update_training_progress(**update_kwargs)


async def handle_device_status(env: EventEnvelope, status_tracker, **kwargs) -> None:
    """Handle device status updates from threadlets."""
    logger.info(f"Handling device status for {env.device_status.device_uuid}")
    ds = env.device_status
    
    update_kwargs = create_device_status_dict(ds)
    status_tracker.update_device_status(**update_kwargs)


# ===========================================
# EXTERNAL EVENT HANDLERS
# ===========================================

# TODO: Implement this
async def handle_monitored_fail(env: EventEnvelope, status_tracker, **kwargs) -> None:
    pass

# ===========================================
# UI EVENT HANDLERS
# ===========================================

# UI Command dispatch table
UI_COMMAND_HANDLERS = {
    "deactivate_device": "handle_deactivate_device",
    "update_config": "handle_update_config",
}

async def handle_ui_command(
    env: EventEnvelope, status_tracker, weaver_publish_command_func, **kwargs
) -> None:
    """Handle all UI commands - everything comes through ui_command now."""
    
    # Handle ui_command events only
    if env.HasField("ui_command"):
        ui_command = env.ui_command
        command_type = ui_command.command_type
        device_uuid = ui_command.device_uuid
        params = dict(ui_command.params)

        logger.info(
            f"Processing UI command: {command_type} for {device_uuid} with params: {params}"
        )

        # Use dispatch table to find handler
        handler_name = UI_COMMAND_HANDLERS.get(command_type)
        if handler_name:
            # Get the handler function from the current module
            handler_func = globals().get(handler_name)
            if handler_func:
                await handler_func(
                    device_uuid, params, status_tracker, weaver_publish_command_func
                )
            else:
                logger.error(f"Handler function {handler_name} not found in module")
        else:
            logger.warning(f"Unknown UI command type: {command_type}")
    
    else:
        logger.warning("UI handler received event with no ui_command payload")

async def handle_update_config(
    device_uuid: str, params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle update config UI command."""
    logger.info(f"Handling update_config for {device_uuid} with params: {params}")
    await weaver_publish_command_func("update_config", device_uuid, params)

async def handle_deactivate_device(
    device_uuid: str, params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle deactivate device UI command."""
    logger.info(f"Handling deactivate_device for {device_uuid}")
    # The deviceStatus proto doesn't have process_id directly.
    # We need to find the process_id(s) associated with this device_uuid.
    process_ids_for_device = status_tracker.get_replicas_for_device(device_uuid)
    if process_ids_for_device:
        for process_id in process_ids_for_device: # Should typically be one for this logic stream
            training_update_kwargs = {
                "process_id": process_id,
                "status": "deactivating" # Conceptual status
            }
            status_tracker.update_training_progress(**training_update_kwargs)
            await weaver_publish_command_func("pause", process_id)
    else:
        logger.warning(f"No process_id found for device_uuid {device_uuid} during deactivation.")