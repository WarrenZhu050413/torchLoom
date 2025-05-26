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
    torchLoom_pb2.PipeHeartbeatMessage,
    torchLoom_pb2.PipeMetricsMessage,
    torchLoom_pb2.PipeCommandMessage,
]
ThreadletToListenerMessage = Union[torchLoom_pb2.PipeHeartbeatMessage, torchLoom_pb2.PipeMetricsMessage]
ListenerToThreadletMessage = torchLoom_pb2.PipeCommandMessage  # Only command messages now


class MessageType(Enum):
    """Message types for pipe communication (simplified to only use COMMAND)."""
    HEARTBEAT = torchLoom_pb2.PIPE_HEARTBEAT
    METRICS = torchLoom_pb2.PIPE_METRICS
    COMMAND = torchLoom_pb2.PIPE_COMMAND
    # CONFIG removed - now handled as a command type


class CommandType(Enum):
    """Command types for command messages."""
    KILL = torchLoom_pb2.KILL
    PAUSE = torchLoom_pb2.PAUSE
    RESUME = torchLoom_pb2.RESUME
    UPDATE_CONFIG = torchLoom_pb2.UPDATE_CONFIG
    # Add CONFIG as a command type to replace the separate config message
    CONFIG = "CONFIG"  # Custom command type for config updates


class MessageFactory:
    """Simple factory for creating messages using Protobuf."""

    @staticmethod
    def create_heartbeat(replica_id: str, status: str = "active") -> torchLoom_pb2.PipeHeartbeatMessage:
        """Create a protobuf heartbeat message."""
        return torchLoom_pb2.PipeHeartbeatMessage(
            message_type=torchLoom_pb2.PIPE_HEARTBEAT,
            timestamp=int(time.time()), # Protobuf uses int64 for timestamp
            replica_id=replica_id,
            status=status,
        )

    @staticmethod
    def create_metrics(
        replica_id: str,
        step: int = 0,
        epoch: int = 0,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        **kwargs: Any, # Protobuf map<string, string> for metrics
    ) -> torchLoom_pb2.PipeMetricsMessage:
        """Create a protobuf metrics message."""
        # Convert all kwargs to string for the protobuf map
        string_metrics = {k: str(v) for k, v in kwargs.items()}
        
        metrics_msg = torchLoom_pb2.PipeMetricsMessage(
            message_type=torchLoom_pb2.PIPE_METRICS,
            timestamp=int(time.time()),
            replica_id=replica_id,
            step=step,
            epoch=epoch,
            metrics=string_metrics,
        )
        if loss is not None:
            metrics_msg.loss = loss
        if accuracy is not None:
            metrics_msg.accuracy = accuracy
        if gradient_norm is not None:
            metrics_msg.gradient_norm = gradient_norm
        return metrics_msg

    @staticmethod
    def create_config(replica_id: str, config_params: Dict[str, Any]) -> torchLoom_pb2.PipeCommandMessage:
        """Create a command message for config updates (replaces separate config message)."""
        # Convert all config_params values to string
        string_config_params = {k: str(v) for k, v in config_params.items()}
        return torchLoom_pb2.PipeCommandMessage(
            message_type=torchLoom_pb2.PIPE_COMMAND,
            timestamp=int(time.time()),
            replica_id=replica_id,
            command_type=torchLoom_pb2.UPDATE_CONFIG,  # Use UPDATE_CONFIG for config updates
            params=string_config_params,
        )

    @staticmethod
    def create_command(
        replica_id: str,
        command_type: Union[torchLoom_pb2.PipeCommandType, str], # Allow string for custom commands
        params: Optional[Dict[str, Any]] = None,
    ) -> torchLoom_pb2.PipeCommandMessage:
        """Create a protobuf command message."""
        # Convert all params values to string
        string_params = {k: str(v) for k, v in (params or {}).items()}
        
        # Handle custom command types (like CONFIG)
        if isinstance(command_type, str):
            # For custom commands, we'll use UPDATE_CONFIG as the protobuf enum
            # and store the actual command type in params
            if command_type == "CONFIG":
                pb_command_type = torchLoom_pb2.UPDATE_CONFIG
                string_params["_command_type"] = command_type
            else:
                pb_command_type = torchLoom_pb2.UPDATE_CONFIG  # Default fallback
                string_params["_command_type"] = command_type
        else:
            pb_command_type = command_type
        
        return torchLoom_pb2.PipeCommandMessage(
            message_type=torchLoom_pb2.PIPE_COMMAND,
            timestamp=int(time.time()),
            replica_id=replica_id,
            command_type=pb_command_type,
            params=string_params,
        )

    @staticmethod
    def create_status(
        replica_id: str,
        status: str = "active",
        current_step: int = 0,
        epoch: int = 0,
        message: str = ""
    ) -> torchLoom_pb2.PipeCommandMessage:
        """Create a command message for status updates."""
        params = {
            "status": status,
            "current_step": str(current_step),
            "epoch": str(epoch),
            "message": message
        }
        return MessageFactory.create_command(
            replica_id=replica_id,
            command_type="STATUS",
            params=params
        )


# Helper functions for protobuf serialization (using native protobuf methods)
def serialize_message(message: PipeMessage) -> bytes:
    """Serialize a protobuf message to bytes using native protobuf serialization."""
    return message.SerializeToString()


def deserialize_message(data: bytes, message_type: str) -> Optional[PipeMessage]:
    """Deserialize bytes back to a protobuf message using native protobuf deserialization."""
    try:
        if message_type == "HEARTBEAT":
            msg = torchLoom_pb2.PipeHeartbeatMessage()
            msg.ParseFromString(data)
            return msg
        elif message_type == "METRICS":
            msg = torchLoom_pb2.PipeMetricsMessage()
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