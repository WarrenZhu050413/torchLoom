"""
Simplified message types for pipe communication between Threadlet and ThreadletListener processes.

This module defines basic message types for inter-process communication.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union
import time


class MessageType(Enum):
    """Basic message types for pipe communication."""
    
    # Threadlet -> ThreadletListener messages
    HEARTBEAT = "heartbeat"
    METRICS = "metrics"
    STATUS = "status"
    
    # ThreadletListener -> Threadlet messages
    CONFIG = "config"
    COMMAND = "command"


class CommandType(Enum):
    """Simple command types."""
    KILL = "kill"
    PAUSE = "pause"
    RESUME = "resume"
    UPDATE_CONFIG = "update_config"


@dataclass
class BaseMessage:
    """Base message class."""
    message_type: MessageType
    timestamp: float = field(default_factory=time.time)
    replica_id: Optional[str] = None


@dataclass
class HeartbeatMessage(BaseMessage):
    """Simple heartbeat message."""
    message_type: MessageType = MessageType.HEARTBEAT
    status: str = "active"


@dataclass
class MetricsMessage(BaseMessage):
    """Basic metrics message."""
    message_type: MessageType = MessageType.METRICS
    step: int = 0
    epoch: int = 0
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    gradient_norm: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatusMessage(BaseMessage):
    """Basic status message."""
    message_type: MessageType = MessageType.STATUS
    status: str = "active"  # active, paused, error, complete
    current_step: int = 0
    epoch: int = 0
    message: str = ""


@dataclass
class ConfigMessage(BaseMessage):
    """Configuration update message."""
    message_type: MessageType = MessageType.CONFIG
    config_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandMessage(BaseMessage):
    """Command message."""
    message_type: MessageType = MessageType.COMMAND
    command_type: CommandType = CommandType.UPDATE_CONFIG
    params: Dict[str, Any] = field(default_factory=dict)


# Type aliases
ThreadletToListenerMessage = Union[HeartbeatMessage, MetricsMessage, StatusMessage]
ListenerToThreadletMessage = Union[ConfigMessage, CommandMessage]
PipeMessage = Union[ThreadletToListenerMessage, ListenerToThreadletMessage]


class MessageFactory:
    """Simple factory for creating messages."""
    
    @staticmethod
    def create_heartbeat(replica_id: str, status: str = "active") -> HeartbeatMessage:
        """Create a heartbeat message."""
        return HeartbeatMessage(replica_id=replica_id, status=status)
    
    @staticmethod
    def create_metrics(
        replica_id: str,
        step: int = 0,
        epoch: int = 0,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        **kwargs
    ) -> MetricsMessage:
        """Create a metrics message."""
        return MetricsMessage(
            replica_id=replica_id,
            step=step,
            epoch=epoch,
            loss=loss,
            accuracy=accuracy,
            gradient_norm=gradient_norm,
            metrics=kwargs
        )
    
    @staticmethod
    def create_status(
        replica_id: str,
        status: str = "active",
        current_step: int = 0,
        epoch: int = 0,
        message: str = ""
    ) -> StatusMessage:
        """Create a status message."""
        return StatusMessage(
            replica_id=replica_id,
            status=status,
            current_step=current_step,
            epoch=epoch,
            message=message
        )
    
    @staticmethod
    def create_config(
        replica_id: str,
        config_params: Dict[str, Any]
    ) -> ConfigMessage:
        """Create a config message."""
        return ConfigMessage(
            replica_id=replica_id,
            config_params=config_params
        )
    
    @staticmethod
    def create_command(
        replica_id: str,
        command_type: CommandType,
        params: Optional[Dict[str, Any]] = None
    ) -> CommandMessage:
        """Create a command message."""
        return CommandMessage(
            replica_id=replica_id,
            command_type=command_type,
            params=params or {}
        )


def serialize_message(message: PipeMessage) -> Dict[str, Any]:
    """Serialize a message to a dictionary for pipe transmission."""
    if hasattr(message, '__dict__'):
        result = message.__dict__.copy()
        # Convert enums to their values
        for key, value in result.items():
            if isinstance(value, Enum):
                result[key] = value.value
        return result
    else:
        return {"message_type": "unknown", "data": str(message)}


def deserialize_message(data: Dict[str, Any]) -> Optional[PipeMessage]:
    """Deserialize a dictionary back to a message object."""
    try:
        message_type_str = data.get("message_type")
        if not message_type_str:
            return None
        
        message_type = MessageType(message_type_str)
        
        # Create the appropriate message type
        if message_type == MessageType.HEARTBEAT:
            return HeartbeatMessage(**data)
        elif message_type == MessageType.METRICS:
            return MetricsMessage(**data)
        elif message_type == MessageType.STATUS:
            return StatusMessage(**data)
        elif message_type == MessageType.CONFIG:
            return ConfigMessage(**data)
        elif message_type == MessageType.COMMAND:
            return CommandMessage(**data)
        else:
            return None
    except Exception:
        return None 