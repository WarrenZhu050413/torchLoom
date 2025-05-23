# Plan for Weavelet Integration

## Goal
Integrate a weavelet process into the Lightning training example so it can receive configuration updates from the weaver. The weavelet should run in a separate process and communicate optimizer type changes to the training process. Callback hooks will check the weavelet queue and update the optimizer when required.

## Current Task: Process-Based Weavelet (✅ COMPLETED)
**Objective**: Refactor the existing thread-based Weavelet to run in a separate process instead of a background thread, similar to the distributed training patterns shown in the user's example.

**Changes Required**:
- ✅ Convert ThreadPoolExecutor-based background execution to multiprocessing.Process
- ✅ Implement proper inter-process communication using multiprocessing.Queue
- ✅ Add process lifecycle management (start, stop, cleanup)
- ✅ Maintain all existing functionality (NATS communication, config handling, status publishing)
- ✅ Follow the pattern from distributed training examples with proper multiprocessing handling

**Implementation Summary**:
1. ✅ Created `WeaveletProcess` class that manages a separate process for weavelet operations
2. ✅ Implemented `AsyncWeavelet` class that runs inside the process with full async capabilities
3. ✅ Added proper inter-process communication using multiprocessing.Queue for config updates and status publishing
4. ✅ Implemented robust process lifecycle management with graceful shutdown, termination, and cleanup
5. ✅ Updated training integration to use queue-based communication instead of direct handler registration
6. ✅ Fixed multiprocessing compatibility issues on macOS by proper start method handling
7. ✅ Maintained backward compatibility with the old `Weavelet` name and `weavelet_process` function

**Key Features Implemented**:
- **Separate Process Execution**: Weavelet runs in its own process using multiprocessing.Process
- **Inter-Process Communication**: Configuration updates and status publishing via multiprocessing.Queue
- **Robust Process Management**: Graceful shutdown, forced termination, and cleanup with timeouts
- **Async NATS Integration**: Full async support for NATS communication within the process
- **macOS Compatibility**: Proper handling of multiprocessing start methods and import protection
- **Test Coverage**: All existing tests updated and passing with the new implementation

**Testing Results**:
- ✅ All unit tests pass (tests/test_weavelet_callback.py)
- ✅ Demo script runs successfully with process-based weavelet
- ✅ Process startup, communication, and shutdown work correctly
- ✅ Configuration updates flow properly through the queue system
- ✅ Status publishing works as expected
