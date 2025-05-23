# Plan for Weavelet Integration

## Goal
Integrate a weavelet process into the Lightning training example so it can receive configuration updates from the weaver. The weavelet should run in a separate process and communicate optimizer type changes to the training process. Callback hooks will check the weavelet queue and update the optimizer when required.

## Steps
- ✅ Create a new module `torchLoom/weavelet.py` with a simple async loop that connects to NATS and listens for config messages. When an `optimizer_type` value is found it is put on a multiprocessing queue.
- ✅ Update `train.py`:
  - ✅ Spawn the weavelet process before starting training.
  - ✅ Extend `LightningTransformer` with dynamic optimizer creation and an `update_optimizer` method.
  - ✅ Add `WeaveletCallback` that checks the queue at the start of each epoch and applies optimizer updates.
- ✅ Add unit tests verifying that `update_optimizer` and the callback correctly change the optimizer based on queued values.
- ✅ Run `pytest` to ensure the repository remains stable.

## Completed Implementation

### Comprehensive Weavelet Class (New)
**Implemented**: A full-featured `Weavelet` class that replaces the simple multiprocessing approach with a comprehensive weaver communication system similar to the marduk pattern.

**Key Features**:
- **Integrated Communication**: Manages all communication between training processes and the weaver
- **Background Threading**: Runs async operations in a background thread using ThreadPoolExecutor
- **Device Registration**: Automatically registers device-to-replica mappings with the weaver
- **Config Handler Registry**: Allows registration of handlers for different configuration parameters
- **Status Publishing**: Can publish training status updates back to the weaver
- **Robust Error Handling**: Includes proper cleanup and error handling
- **Fallback Support**: Works on systems without NVIDIA libraries (e.g., Mac) using fallback device UUIDs

**Architecture**:
- **Event Loop Management**: Creates and manages its own asyncio event loop in a background thread
- **NATS Subscriptions**: Subscribes to both JetStream and regular NATS subjects
- **Message Handling**: Processes incoming config updates and replica failure notifications
- **Thread Safety**: Properly handles cross-thread communication for status publishing

### Updated Training Integration
**Enhanced**: The `LightningTransformer` now integrates the weavelet directly instead of using multiprocessing:

- **Direct Integration**: Weavelet is created and started within the Lightning module
- **Config Handlers**: Registers the `update_optimizer` method as a config handler
- **Status Reporting**: Publishes training status (batch index, loss, optimizer type) to the weaver
- **Cleanup**: Properly stops the weavelet when training ends

### Simplified Callback
**Modernized**: The `WeaveletCallback` is now simplified and focuses on lifecycle management:

- **Startup Verification**: Ensures weavelet is running when training starts
- **Cleanup Assistance**: Helps with proper weavelet shutdown when training ends
- **No Queue Management**: No longer needs to manage multiprocessing queues

### Backward Compatibility
**Maintained**: The old `weavelet_process` function is still available for backward compatibility, but it now uses the new `Weavelet` class internally.

## Issues Resolved

### Pyre Type Checker Error (Fixed)
**Problem**: The Pyre linter was incorrectly inferring that `opt_type` from `queue.get_nowait()` was a Tensor instead of a string, causing this error:
```
Object of type "Tensor" is not callable
Attribute "__call__" is unknown
```

**Root Cause**: The multiprocessing queue's `get_nowait()` method returns `Any` type, and Pyre couldn't infer that it should be a string in this context.

**Solution**: Added explicit type annotation in `train.py` line 69:
```python
opt_type: str = self.queue.get_nowait()
```

**Verification**: 
- Test passed showing queue returns string type correctly
- Related pytest tests continue to pass
- Code imports and runs without runtime errors

This fix ensures that Pyre understands the expected type while maintaining all existing functionality.

### Device UUID Fallback (New)
**Problem**: The system failed on machines without NVIDIA libraries (e.g., Mac, CPU-only systems).

**Solution**: Enhanced `get_device_uuid()` to provide fallback UUIDs using machine hostname and UUID generation when NVIDIA libraries are unavailable.

**Result**: The system now works across all environments, including development machines without GPUs.

## Next Steps
- Test the comprehensive weavelet with actual NATS server and weaver integration
- Add more configuration parameter handlers (learning rate, batch size, etc.)
- Enhance status reporting with more detailed training metrics
- Consider adding recovery logic for replica failure scenarios
