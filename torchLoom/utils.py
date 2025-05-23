import asyncio
import platform
from typing import Any, Dict, Tuple

import torch
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetUUID, nvmlInit

from torchLoom.config import Config
from torchLoom.log.log_utils import log_and_raise_exception
from torchLoom.log.logger import setup_logger

logger = setup_logger(name="torchLoom_utils", log_file=Config.torchLoom_UTILS_LOG_FILE)


def get_device_uuid():
    try:
        nvmlInit()
        index = torch.cuda.current_device()
        handle = nvmlDeviceGetHandleByIndex(index)
        gpu_uuid = nvmlDeviceGetUUID(handle)
        logger.info(f"Retrieved GPU UUID: {gpu_uuid} for device index {index}")
        return gpu_uuid
    except Exception as e:
        # Fallback for systems without NVIDIA libraries (e.g., Mac, CPU-only systems)
        platform_name = platform.node()
        logger.warning(f"Failed to get GPU UUID, on {platform_name}, Error: {e}")
        return platform.node()


async def cancel_subscriptions(subscriptions: Dict[str, Tuple[Any, asyncio.Task]]):
    # cancel all the tasks
    for _, task in subscriptions.values():
        task.cancel()

    # wait for cancellation to finish
    await asyncio.gather(
        *(task for _, task in subscriptions.values()), return_exceptions=True
    )

    # unsubscribe from NATS
    for sub, _ in subscriptions.values():
        await sub.unsubscribe()
