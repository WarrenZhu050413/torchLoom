import asyncio
import os
import platform
import sys
from typing import Any, Dict, Tuple

import torch
from pynvml import (
    NVMLError,
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUUID,
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetMemoryInfo,
    NVML_TEMPERATURE_GPU,
)

from torchLoom.common.constants import LoggerConstants
from torchLoom.log_utils.log_utils import log_and_raise_exception
from torchLoom.log_utils.logger import setup_logger
from torchLoom.proto import torchLoom_pb2

logger = setup_logger(
    name="torchLoom_utils", log_file=LoggerConstants.torchLoom_UTILS_LOG_FILE
)

def maybe_get_device_uuid():
    """
    Attempts to get the GPU UUID associated with the current process.
    Falls back to the UUID of the current PyTorch device if the process isn't
    found explicitly, or to the platform node name if NVML fails or no GPU is found.
    """
    try:
        nvmlInit()
        pid = os.getpid()
        device_count = nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            try:
                processes = nvmlDeviceGetComputeRunningProcesses(handle)
                for p_info in processes:
                    if p_info.pid == pid:
                        device_uuid = nvmlDeviceGetUUID(handle)
                        logger.info(
                            f"Process PID {pid} found on device index {i}, UUID: {device_uuid}"
                        )
                        nvmlShutdown()
                        return device_uuid
            except NVMLError as e:
                # This specific NVMLError might happen if the process list is empty or restricted
                logger.debug(f"NVMLError while checking processes on device {i}: {e}")
                continue

        logger.warning(
            f"Process PID {pid} not found on any GPU via nvmlDeviceGetComputeRunningProcesses. "
            f"PyTorch current_device() index: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}. "
            f"Falling back to PyTorch device logic if available, or platform.node()."
        )
        if torch.cuda.is_available():
            try:
                torch_device_index = torch.cuda.current_device()
                # Handle CUDA_VISIBLE_DEVICES mapping
                visible_devices_str = os.environ.get("CUDA_VISIBLE_DEVICES")
                if visible_devices_str:
                    try:
                        visible_devices_list = [int(x.strip()) for x in visible_devices_str.split(',')]
                        if torch_device_index < len(visible_devices_list):
                            physical_idx = visible_devices_list[torch_device_index]
                            handle = nvmlDeviceGetHandleByIndex(physical_idx)
                            device_uuid = nvmlDeviceGetUUID(handle)
                            logger.info(
                                f"Fallback: Retrieved device UUID: {device_uuid} for PyTorch logical device index {torch_device_index} (physical index {physical_idx} from CUDA_VISIBLE_DEVICES)"
                            )
                            nvmlShutdown()
                            return device_uuid
                        else:
                            logger.warning(f"Fallback: PyTorch device index {torch_device_index} out of bounds for CUDA_VISIBLE_DEVICES: {visible_devices_str}. Cannot map to physical device.")
                    except (ValueError, IndexError) as e_visible:
                         logger.warning(f"Fallback: Failed to parse or use CUDA_VISIBLE_DEVICES '{visible_devices_str}': {e_visible}. Trying direct index.")
                         # Fall through to direct index attempt
                    except NVMLError as e_nvml_visible:
                         logger.warning(f"Fallback: NVMLError getting handle/UUID for physical index derived from CUDA_VISIBLE_DEVICES: {e_nvml_visible}. Trying direct index.")
                         # Fall through to direct index attempt

                # If CUDA_VISIBLE_DEVICES wasn't set, failed, or index was out of bounds
                if 'device_uuid' not in locals(): # Check if UUID was found in the block above
                    handle = nvmlDeviceGetHandleByIndex(torch_device_index)
                    device_uuid = nvmlDeviceGetUUID(handle)
                    logger.info(
                        f"Fallback: Retrieved device UUID: {device_uuid} for PyTorch device index {torch_device_index} (direct index, no CUDA_VISIBLE_DEVICES or mapping failed)"
                    )
                    nvmlShutdown()
                    return device_uuid

            except NVMLError as e_torch_nvml:
                logger.warning(f"Fallback using torch.cuda.current_device() failed NVML call: {e_torch_nvml}")
            except Exception as e_torch_fallback:
                logger.warning(f"Fallback using torch.cuda.current_device() failed unexpectedly: {e_torch_fallback}")

        platform_name = platform.node()
        logger.warning(f"PID {pid} not found on any GPU, PyTorch fallback failed, or no GPUs available. Using platform.node(): {platform_name}")
        try:
            nvmlShutdown() # Ensure shutdown even if fallback failed
        except NVMLError:
            pass # Ignore if shutdown fails after init failed
        return platform_name

    except NVMLError as e_init:
        platform_name = platform.node()
        logger.warning(f"NVML Error during initialization or first call (e.g., NVIDIA drivers not found), on {platform_name}, Error: {e_init}. Using platform.node().")
        try:
            nvmlShutdown() # Attempt shutdown even if init failed partially
        except NVMLError:
            pass # Ignore if shutdown fails after init failed
        return platform_name
    except Exception as e_global:
        platform_name = platform.node()
        logger.error(f"An unexpected error occurred in maybe_get_device_uuid: {e_global}. Using platform.node(): {platform_name}")
        try:
            nvmlShutdown() # Attempt shutdown on unexpected error
        except NVMLError:
            pass # Ignore if shutdown fails
        return platform_name


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

def get_device_status(device_handle, server_id: str, process_id: str = None) -> dict:
    """
    Collects device status information using pynvml for a given device handle.

    Args:
        device_handle: The pynvml device handle.
        server_id: Identifier for the server.
        process_id: Identifier for the process. Defaults to current process ID.

    Returns:
        A dictionary containing device status information, matching the structure
        of the deviceStatus protobuf message. Memory values are in GB.
    """
    if process_id is None:
        process_id = str(os.getpid())

    status = {
        "device_uuid": None,
        "process_id": process_id,
        "server_id": server_id,
        "utilization": None,
        "temperature": None,
        "memory_used": None, # in GB
        "memory_total": None, # in GB
    }

    try:
        status["device_uuid"] = nvmlDeviceGetUUID(device_handle)

        util = nvmlDeviceGetUtilizationRates(device_handle)
        status["utilization"] = float(util.gpu)

        status["temperature"] = float(nvmlDeviceGetTemperature(device_handle, NVML_TEMPERATURE_GPU))

        mem = nvmlDeviceGetMemoryInfo(device_handle)
        bytes_to_gb = 1024 * 1024 * 1024.0
        status["memory_used"] = round(mem.used / bytes_to_gb, 2) # needs conversion to GB
        status["memory_total"] = round(mem.total / bytes_to_gb, 2) # needs conversion to GB

    except NVMLError as err:
        print(f"Error collecting NVML data for device handle {device_handle}: {err}", file=sys.stderr)

    return status

try:
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    print("device_count:", device_count)
    if device_count > 0:
        for device_index in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(device_index) # Get handle for the first GPU
            server_id = platform.node() # Replace with actual server ID
            import uuid
            process_id = str(uuid.uuid4())
            status_data = get_device_status(handle, server_id, process_id=process_id)
            print(status_data)
    nvmlShutdown()
except NVMLError as err:
    print(f"NVML Error: {err}", file=sys.stderr)