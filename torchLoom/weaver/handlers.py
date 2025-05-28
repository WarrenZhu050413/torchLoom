"""
Individual handler functions for the torchLoom Weaver.

This module contains individual handler functions that can be registered
with the HandlerRegistry in weaver.py for processing different types of events.
"""

import logging
import time
from typing import Dict, Optional, Set

from torchLoom.common.constants import TimeConstants
from torchLoom.common.utils import (
    create_device_status_dict,
    create_training_status_dict,
)
from torchLoom.log_utils.logger import setup_logger
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
    status_tracker.add_device_pid_mapping(
        env.register_device.device_uuid, env.register_device.process_id
    )
    status_tracker.add_pid_device_mapping(
        env.register_device.process_id, env.register_device.device_uuid
    )
    # Update training progress
    training_update_kwargs = {
        "process_id": env.register_device.process_id,
    }
    status_tracker.update_training_progress(**training_update_kwargs)

    # Update device status
    device_update_kwargs = {
        "device_uuid": env.register_device.device_uuid,
        "process_id": env.register_device.process_id,
        # "server_id": TODO
    }
    status_tracker.update_device_status(**device_update_kwargs)


async def handle_heartbeat(
    env: EventEnvelope, status_tracker, heartbeat_tracker, **kwargs
) -> None:
    """Handle heartbeat from threadlets."""
    process_id = env.heartbeat.process_id
    heartbeat_status = env.heartbeat.status  # Get status from heartbeat proto
    logger.info(
        f"Handling heartbeat for {process_id} with reported status: '{heartbeat_status}'"
    )

    heartbeat_tracker["last_heartbeats"][process_id] = time.time()

    # Default to the heartbeat status, or "active" if heartbeat status is empty
    # This ensures we always have a meaningful status to set.
    current_replica_status = heartbeat_status if heartbeat_status else "active"

    if process_id in heartbeat_tracker["dead_processes"]:
        heartbeat_tracker["dead_processes"].remove(process_id)
        logger.info(
            f"Replica {process_id} revived by heartbeat. Status reported: '{current_replica_status}'."
        )
    # else, replica is already considered alive, just update its status if reported by heartbeat
    # No, always update, because the status from heartbeat (e.g. training, idle) is the source of truth.

    # Note: TrainingStatus protobuf doesn't have a 'status' field, so we store it in metrics
    training_update_kwargs = {
        "process_id": process_id,
        "metrics": {
            "heartbeat_status": current_replica_status
        },  # Store status in metrics
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
    "pause_training": "handle_pause_training",
    "resume_training": "handle_resume_training",
}


async def handle_ui_command(
    env: EventEnvelope, status_tracker, weaver_publish_command_func, **kwargs
) -> None:
    """Handle all UI commands - everything comes through ui_command now."""

    # Handle ui_command events only
    if env.HasField("ui_command"):
        ui_command = env.ui_command
        command_type = ui_command.command_type
        process_id = ui_command.process_id
        params = dict(ui_command.params)

        logger.info(
            f"Processing UI command: {command_type} for process_id {process_id} with params: {params}"
        )

        # Use dispatch table to find handler
        handler_name = UI_COMMAND_HANDLERS.get(command_type)
        logger.info(f"Handler name: {handler_name}")
        if handler_name:
            # Get the handler function from the current module
            handler_func = globals().get(handler_name)
            if handler_func:
                await handler_func(
                    process_id, params, status_tracker, weaver_publish_command_func
                )
            else:
                logger.error(f"Handler function {handler_name} not found in module")
        else:
            logger.warning(f"Unknown UI command type: {command_type}")

    else:
        logger.warning("UI handler received event with no ui_command payload")


async def handle_update_config(
    process_id: str, params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle update config UI command."""
    logger.info(
        f"Handling update_config for process_id {process_id} with params: {params}"
    )

    # Verify the process_id is registered
    if status_tracker.has_process_id(process_id):
        logger.info(f"Sending update_config command to process_id: {process_id}")
        await weaver_publish_command_func("update_config", process_id, params)
    else:
        # Debug: Show what process_ids are actually registered
        available_process_ids = list(status_tracker.pid_to_devices.keys())
        logger.warning(f"Process_id {process_id} not found during config update.")
        logger.warning(f"Available registered process_ids: {available_process_ids}")


async def handle_pause_training(
    process_id: str, params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle pause training UI command."""
    logger.info(f"Handling pause_training for process_id {process_id}")

    # Verify the process_id is registered
    if status_tracker.has_process_id(process_id):
        logger.info(f"Sending pause command to process_id: {process_id}")
        # Store status in metrics since TrainingStatus doesn't have a status field
        training_update_kwargs = {
            "process_id": process_id,
            "metrics": {"training_status": "pausing"},
        }
        status_tracker.update_training_progress(**training_update_kwargs)
        await weaver_publish_command_func("pause", process_id, params)
    else:
        # Debug: Show what process_ids are actually registered
        available_process_ids = list(status_tracker.pid_to_devices.keys())
        logger.warning(f"Process_id {process_id} not found during pause.")
        logger.warning(f"Available registered process_ids: {available_process_ids}")


async def handle_resume_training(
    process_id: str, params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle resume training UI command."""
    logger.info(f"Handling resume_training for process_id {process_id}")

    # Verify the process_id is registered
    if status_tracker.has_process_id(process_id):
        logger.info(f"Sending resume command to process_id: {process_id}")
        # Store status in metrics since TrainingStatus doesn't have a status field
        training_update_kwargs = {
            "process_id": process_id,
            "metrics": {"training_status": "resuming"},
        }
        status_tracker.update_training_progress(**training_update_kwargs)
        await weaver_publish_command_func("resume", process_id, params)
    else:
        # Debug: Show what process_ids are actually registered
        available_process_ids = list(status_tracker.pid_to_devices.keys())
        logger.warning(f"Process_id {process_id} not found during resume.")
        logger.warning(f"Available registered process_ids: {available_process_ids}")


async def handle_deactivate_device(
    process_id: str, params: Dict, status_tracker, weaver_publish_command_func
):
    """Handle deactivate device UI command."""
    logger.info(f"Handling deactivate_device for process_id {process_id}")

    # Verify the process_id is registered
    if status_tracker.has_process_id(process_id):
        logger.info(f"Sending deactivate command to process_id: {process_id}")
        # Store status in metrics since TrainingStatus doesn't have a status field
        training_update_kwargs = {
            "process_id": process_id,
            "metrics": {"training_status": "deactivating"},
        }
        status_tracker.update_training_progress(**training_update_kwargs)
        await weaver_publish_command_func("pause", process_id, params)
    else:
        # Debug: Show what process_ids are actually registered
        available_process_ids = list(status_tracker.pid_to_devices.keys())
        logger.warning(f"Process_id {process_id} not found during deactivation.")
        logger.warning(f"Available registered process_ids: {available_process_ids}")
