"""
Individual handler functions for the torchLoom Threadlet.

This module contains individual handler functions that can be registered
with the HandlerRegistry in threadlet.py for processing different types of commands.
"""

import logging
from typing import Any, Callable, Dict, Optional

from torchLoom.common.handlers import HandlerRegistry
from torchLoom.log_utils.logger import setup_logger
from torchLoom.proto import torchLoom_pb2
from torchLoom.proto.torchLoom_pb2 import PipeCommandType as CommandType

logger = setup_logger(name="threadlet_handlers")


# Command dispatch table
COMMAND_HANDLERS = {
    CommandType.KILL: "handle_kill_command",
    CommandType.PAUSE: "handle_pause_command",
    CommandType.RESUME: "handle_resume_command",
    CommandType.UPDATE_CONFIG: "handle_update_config",
}


def handle_command_message(
    message: torchLoom_pb2.PipeCommandMessage,
    handler_registry: HandlerRegistry,
    auto_dispatch: bool,
    stop_callback: Optional[Callable] = None,
    **kwargs,
) -> None:
    """Main command handler that dispatches to specific command handlers."""
    try:
        command_type = (
            message.command_type.value
            if hasattr(message.command_type, "value")
            else message.command_type
        )
        params = dict(message.params) if message.params else {}

        # Check for custom command type in params
        actual_command_type = params.pop("_command_type", None) or command_type

        logger.info(f"Processing command: {actual_command_type} with params: {params}")

        # Use dispatch table to find handler
        handler_name = COMMAND_HANDLERS.get(actual_command_type)
        if not handler_name:
            # Try with command_type enum if actual_command_type didn't match
            handler_name = COMMAND_HANDLERS.get(command_type)

        if handler_name:
            # Get the handler function from the current module
            handler_func = globals().get(handler_name)
            if handler_func:
                handler_func(
                    params=params,
                    handler_registry=handler_registry,
                    auto_dispatch=auto_dispatch,
                    stop_callback=stop_callback,
                    **kwargs,
                )
            else:
                logger.error(f"Handler function {handler_name} not found in module")
        else:
            logger.warning(f"Unknown command type: {actual_command_type}")

    except Exception as e:
        logger.exception(f"Error handling command message: {e}")


def handle_kill_command(
    params: Dict[str, Any], stop_callback: Optional[Callable] = None, **kwargs
) -> None:
    """Handle KILL command from weaver."""
    logger.warning("Received KILL command from weaver")
    if stop_callback:
        stop_callback()
    else:
        logger.error("No stop callback provided for KILL command")


def handle_pause_command(
    params: Dict[str, Any],
    handler_registry: HandlerRegistry,
    auto_dispatch: bool,
    **kwargs,
) -> None:
    """Handle PAUSE command from weaver."""
    logger.info("Received PAUSE command from weaver")
    if auto_dispatch and handler_registry:
        handler_registry.dispatch_handlers({"pause_training": True})


def handle_resume_command(
    params: Dict[str, Any],
    handler_registry: HandlerRegistry,
    auto_dispatch: bool,
    **kwargs,
) -> None:
    """Handle RESUME command from weaver."""
    logger.info("Received RESUME command from weaver")
    if auto_dispatch and handler_registry:
        handler_registry.dispatch_handlers({"resume_training": True})


def handle_update_config(
    params: Dict[str, Any],
    handler_registry: HandlerRegistry,
    auto_dispatch: bool,
    **kwargs,
) -> None:
    """Handle UPDATE_CONFIG command from weaver."""
    logger.info(f"Received config update command with params: {params}")
    if auto_dispatch and params and handler_registry:
        handler_registry.dispatch_handlers(params)


def handle_status_command(params: Dict[str, Any], **kwargs) -> None:
    """Handle STATUS command from weaver."""
    logger.info(f"Received status command: {params}")
    # Status updates can be handled here if needed
    # For now, just log them


def create_threadlet_command_registry() -> HandlerRegistry:
    """Create and configure the command handler registry for Threadlet."""
    registry = HandlerRegistry("threadlet_commands")

    # Register the main command handler
    registry.register_handler("command", handle_command_message)

    return registry
