import pynvml
import os
import sys
import platform

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
        # Corresponds to pynvml.nvmlDeviceGetUUID(handle)
        status["device_uuid"] = pynvml.nvmlDeviceGetUUID(device_handle)

        # Corresponds to pynvml.nvmlDeviceGetUtilizationRates(handle).gpu (0-100%)
        util = pynvml.nvmlDeviceGetUtilizationRates(device_handle)
        status["utilization"] = float(util.gpu)

        # Corresponds to pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU) (Celsius)
        # Fix: Changed NVML_TEMP_GPU to NVML_TEMPERATURE_GPU based on traceback
        status["temperature"] = float(pynvml.nvmlDeviceGetTemperature(device_handle, pynvml.NVML_TEMPERATURE_GPU))

        # Corresponds to pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
        bytes_to_gb = 1024 * 1024 * 1024.0
        status["memory_used"] = round(mem.used / bytes_to_gb, 2) # needs conversion to GB
        status["memory_total"] = round(mem.total / bytes_to_gb, 2) # needs conversion to GB

    except pynvml.NVMLError as err:
        # Log the error or handle it as appropriate for your application
        print(f"Error collecting NVML data for device handle {device_handle}: {err}", file=sys.stderr)
        # Keep None values for fields that failed

    return status

try:
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print("device_count:", device_count)
    if device_count > 0:
        for device_index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index) # Get handle for the first GPU
            server_id = platform.node() # Replace with actual server ID
            import uuid
            process_id = str(uuid.uuid4())
            status_data = get_device_status(handle, server_id, process_id=process_id)
            print(status_data)
    pynvml.nvmlShutdown()
except pynvml.NVMLError as err:
    print(f"NVML Error: {err}", file=sys.stderr)
