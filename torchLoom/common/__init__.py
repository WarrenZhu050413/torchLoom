"""
Common utilities and types for torchLoom.
"""

# Import protobuf classes directly
from torchLoom.proto.torchLoom_pb2 import (
    TrainingStatus as _TrainingStatus,
    deviceStatus as _deviceStatus,
)


# Add from_dict methods to protobuf classes
def _training_status_from_dict(cls, data):
    """Create TrainingStatus from dictionary."""
    obj = cls()
    for key, value in data.items():
        if key == "metrics" and isinstance(value, dict):
            obj.metrics.update({k: str(v) for k, v in value.items()})
        elif key == "config" and isinstance(value, dict):
            obj.config.update({k: str(v) for k, v in value.items()})
        elif hasattr(obj, key):
            setattr(obj, key, value)
    return obj


def _device_status_from_dict(cls, data):
    """Create deviceStatus from dictionary."""
    obj = cls()
    for key, value in data.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
    return obj


# Monkey patch the from_dict methods
_TrainingStatus.from_dict = classmethod(_training_status_from_dict)
_deviceStatus.from_dict = classmethod(_device_status_from_dict)

# Export the enhanced classes
TrainingStatus = _TrainingStatus
deviceStatus = _deviceStatus

__all__ = [
    "TrainingStatus",
    "deviceStatus",
]
