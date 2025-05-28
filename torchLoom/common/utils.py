import asyncio
import os
import platform
from typing import Any, Dict, Tuple

import torch
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetUUID, nvmlInit

from torchLoom.common.constants import LoggerConstants
from torchLoom.log.log_utils import log_and_raise_exception
from torchLoom.log.logger import setup_logger
from torchLoom.proto import torchLoom_pb2

logger = setup_logger(
    name="torchLoom_utils", log_file=LoggerConstants.torchLoom_UTILS_LOG_FILE
)


def maybe_get_device_uuid():
    try:
        nvmlInit()
        index = torch.cuda.current_device()
        handle = nvmlDeviceGetHandleByIndex(index)
        device_uuid = nvmlDeviceGetUUID(handle)
        logger.info(f"Retrieved device UUID: {device_uuid} for device index {index}")
        return device_uuid
    except Exception as e:
        # Fallback for systems without NVIDIA libraries (e.g., Mac, CPU-only systems)
        platform_name = platform.node()
        logger.warning(f"Failed to get device UUID, on {platform_name}, Error: {e}")
        return platform.node()


def create_training_status_dict(ts: torchLoom_pb2.TrainingStatus) -> dict:
    """Converts a TrainingStatus protobuf message to a dictionary."""
    update_kwargs = {
        "process_id": ts.process_id,
        "current_step": ts.current_step,
        "epoch": ts.epoch,
        "metrics": dict(ts.metrics),  # Convert protobuf map to dict
        "training_time": ts.training_time,
        "max_step": ts.max_step,
        "max_epoch": ts.max_epoch,
        "config": dict(ts.config),  # Convert protobuf map to dict
    }
    return update_kwargs


def create_device_status_dict(ds: torchLoom_pb2.deviceStatus) -> dict:
    """Converts a DeviceStatus protobuf message to a dictionary."""
    return {
        "device_uuid": ds.device_uuid,
        "process_id": ds.process_id,
        "server_id": ds.server_id,
        "utilization": ds.utilization,
        "temperature": ds.temperature,
        "memory_used": ds.memory_used,
        "memory_total": ds.memory_total,
    }
