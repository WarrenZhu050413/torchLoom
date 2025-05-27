"""
Simplified message types for pipe communication between Threadlet and ThreadletListener processes.

This module defines basic message types for inter-process communication.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union

from torchLoom.proto import torchLoom_pb2

# Use only command messages - config is now a command type
PipeMessage = Union[
    torchLoom_pb2.PipeCommandMessage,
    torchLoom_pb2.PipeTrainingStatusMessage,
    torchLoom_pb2.PipeDeviceStatusMessage,
]
ThreadletToListenerMessage = Union[
    torchLoom_pb2.PipeTrainingStatusMessage,
    torchLoom_pb2.PipeDeviceStatusMessage,
]
ListenerToThreadletMessage = (
    torchLoom_pb2.PipeCommandMessage
)  # Only command messages now


class MessageType(Enum):
    """Message types for pipe communication."""

    TRAINING_STATUS = torchLoom_pb2.PIPE_TRAINING_STATUS
    DEVICE_STATUS = torchLoom_pb2.PIPE_DEVICE_STATUS
    COMMAND = torchLoom_pb2.PIPE_COMMAND


class CommandType(Enum):
    """Command types for command messages."""

    KILL = torchLoom_pb2.KILL
    PAUSE = torchLoom_pb2.PAUSE
    RESUME = torchLoom_pb2.RESUME
    UPDATE_CONFIG = torchLoom_pb2.UPDATE_CONFIG
    CONFIG = "CONFIG"

class MessageFactory:
    """Simple factory for creating messages using Protobuf."""

    @staticmethod
    def create_training_status(
        process_id: str,
        current_step: int = 0,
        epoch: int = 0,
        message: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        training_time: float = 0.0,
        max_step: int = 0,
        max_epoch: int = 0,
        config: Optional[Dict[str, Any]] = None,
    ) -> torchLoom_pb2.PipeTrainingStatusMessage:
        """Create a protobuf training status message using proper TrainingStatus structure."""
        
        # Convert metrics to string map
        metrics_map = {}
        if metrics:
            metrics_map = {k: str(v) for k, v in metrics.items()}
        
        # Add the message to metrics if provided
        if message:
            metrics_map["message"] = message

        # Convert config to string map
        config_map = {}
        if config:
            config_map = {k: str(v) for k, v in config.items()}

        # Create the TrainingStatus message
        training_status = torchLoom_pb2.TrainingStatus(
            process_id=process_id,
            current_step=current_step,
            epoch=epoch,
            metrics=metrics_map,
            training_time=training_time,
            max_step=max_step,
            max_epoch=max_epoch,
            config=config_map,
        )

        # Create the pipe message wrapper
        pipe_training_status = torchLoom_pb2.PipeTrainingStatusMessage(
            message_type=torchLoom_pb2.PIPE_TRAINING_STATUS,
            timestamp=int(time.time()),
            training_status=training_status,
        )
            
        return pipe_training_status

    @staticmethod
    def create_device_status(
        device_uuid: str,
        process_id: str,
        server_id: str = "",
        utilization: float = 0.0,
        temperature: float = 0.0,
        memory_used: float = 0.0,
        memory_total: float = 0.0,
    ) -> torchLoom_pb2.PipeDeviceStatusMessage:
        """Create a protobuf device status message using proper deviceStatus structure."""
        
        # Create the deviceStatus message
        device_status = torchLoom_pb2.deviceStatus(
            device_uuid=device_uuid,
            process_id=process_id,
            server_id=server_id,
            utilization=utilization,
            temperature=temperature,
            memory_used=memory_used,
            memory_total=memory_total,
        )

        # Create the pipe message wrapper
        pipe_device_status = torchLoom_pb2.PipeDeviceStatusMessage(
            message_type=torchLoom_pb2.PIPE_DEVICE_STATUS,
            timestamp=int(time.time()),
            device_status=device_status,
        )
            
        return pipe_device_status

    @staticmethod
    def create_config(
        process_id: str, config_params: Dict[str, Any]
    ) -> torchLoom_pb2.PipeCommandMessage:
        """Create a command message for config updates (replaces separate config message)."""
        # Convert all config_params values to string
        string_config_params = {k: str(v) for k, v in config_params.items()}
        return torchLoom_pb2.PipeCommandMessage(
            message_type=torchLoom_pb2.PIPE_COMMAND,
            timestamp=int(time.time()),
            process_id=process_id,
            command_type=torchLoom_pb2.UPDATE_CONFIG,  # Use UPDATE_CONFIG for config updates
            params=string_config_params,
        )

    @staticmethod
    def create_command(
        process_id: str,
        command_type: Union[
            torchLoom_pb2.PipeCommandType, str
        ],  # Allow string for custom commands
        params: Optional[Dict[str, Any]] = None,
    ) -> torchLoom_pb2.PipeCommandMessage:
        """Create a protobuf command message."""
        string_params = {k: str(v) for k, v in (params or {}).items()}

        if isinstance(command_type, str):
            if command_type == "CONFIG":
                pb_command_type = torchLoom_pb2.UPDATE_CONFIG
                string_params["_command_type"] = command_type
            else:
                pb_command_type = torchLoom_pb2.UPDATE_CONFIG
                string_params["_command_type"] = command_type
        else:
            pb_command_type = command_type

        return torchLoom_pb2.PipeCommandMessage(
            message_type=torchLoom_pb2.PIPE_COMMAND,
            timestamp=int(time.time()),
            process_id=process_id,
            command_type=pb_command_type,
            params=string_params,
        )


# Helper functions for protobuf serialization (using native protobuf methods)
def serialize_message(message: PipeMessage) -> bytes:
    """Serialize a protobuf message to bytes using native protobuf serialization."""
    return message.SerializeToString()


def deserialize_message(data: bytes, message_type: str) -> Optional[PipeMessage]:
    """Deserialize bytes back to a protobuf message using native protobuf deserialization."""
    try:
        if message_type == "TRAINING_STATUS":
            msg = torchLoom_pb2.PipeTrainingStatusMessage()
            msg.ParseFromString(data)
            return msg
        elif message_type == "DEVICE_STATUS":
            msg = torchLoom_pb2.PipeDeviceStatusMessage()
            msg.ParseFromString(data)
            return msg
        elif message_type == "COMMAND":
            msg = torchLoom_pb2.PipeCommandMessage()
            msg.ParseFromString(data)
            return msg
        else:
            return None
    except Exception:
        return None
