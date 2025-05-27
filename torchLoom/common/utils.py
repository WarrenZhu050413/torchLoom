import asyncio
import platform
from typing import Any, Dict, Tuple

import torch
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetUUID, nvmlInit

from torchLoom.log.log_utils import log_and_raise_exception
from torchLoom.log.logger import setup_logger
from torchLoom.common.constants import LoggerConstants

logger = setup_logger(name="torchLoom_utils", log_file=LoggerConstants.torchLoom_UTILS_LOG_FILE)

def get_device_uuid():
    try:
        nvmlInit()
        index = torch.cuda.current_device()
        handle = nvmlDeviceGetHandleByIndex(index)
        device_uuid = nvmlDeviceGetUUID(handle)
        logger.info(f"Retrieved device UUID: {device_uuid} for device index {index}")
        return device_uuid
    except Exception as e:
        # Fallback for systems without NVIDIA libraries (e.g., Mac, CPU-only systems)
        platform_name = os.hostname()
        logger.warning(f"Failed to get device UUID, on {platform_name}, Error: {e}")
        return os.hostname()
