# Instructions

## General Guidelines
 * Plan First: Before coding, create a detailed implementation plan in plan.md, explaining actions and rationale. Update this plan as you complete tasks.
 * Prioritize Simplicity: Favor simple, error-free approaches over complex ones, unless specifically requested. As simple as possible, but no simpler.
 * Review and Refactor: After implementing a feature, review your code for correctness, modularity, and potential code reuse.
 * Don't Alter Tests to Pass: If tests fail, fix the code, not the tests (unless the tests themselves are flawed).
 * Explain Design Decisions: Clearly articulate your design choices. If fixing bugs, explain their cause to prevent future occurrences.

## Design Guide
 * Simple Tests: Keep tests as simple as possible, akin to unit tests.
 * Code Consistency: Reuse existing code and maintain the established coding style.
 * Test without Changing Code: When writing tests, avoid modifying existing code unless it contains bugs.

## Style Guide
 * No Numbered Comments: Do not use numbered lists (e.g., 1., 2., 3...) in comments.

## Details
 * NVIDIA Environment: When setting up NVIDIA environments, verify compatibility using nvidia-smi and checking the installed version.
 * Conda Usage: Always initialize (conda init) and activate the correct conda environment before running commands.
 * Linter Settings: Do not modify linter configurations.
* Remember to import all the libraries that you are using. Also, don't reimport the same library twice.

## Testing
* After you implement any change, always run the tests to ensure that you didn't break anything.
* If the feature you implement is not covered by the tests, add tests to cover it.

## Continuous Learning

* Continuously add to AGENTS.md as you learn more about the codebase and its best practices.

<Environment Specific Instructions>
- Run tests through pytest
- You should run not only tests, but the code that you have changed.
- Run linters after changes, by doing 
```sh
lintrunner init
lintrunner -a
```

You may get the following error:

```sh
  Advice (pyre) command-failed
    Failed due to JSONDecodeError:
    Expecting value: line 1 column 1 (char 0)
```

but ignore this.

<Design Philosophy>

# torchLoom Weavelet Design Philosophy

## Vision

The **Weavelet** is designed as a **plug-and-play component** that can be seamlessly integrated into any PyTorch training code to enable **dynamic reconfiguration** during training. The goal is to transform static training processes into adaptive, controllable systems that can respond to real-time configuration changes from a central weaver service.

## Core Design Principles

### 1. **Minimal Integration Effort**
- **One-line Integration**: Adding dynamic reconfiguration should require minimal code changes
- **Framework Agnostic**: Works with PyTorch Lightning, native PyTorch, or any training framework
- **Drop-in Component**: Can be added to existing training code without architectural changes

```python
# Ideal integration pattern
class MyTrainer(L.LightningModule):
    def __init__(self):
        # Single line to enable dynamic reconfiguration
        self.weavelet = Weavelet(replica_id="trainer_1")
        self.weavelet.start()
```

### 2. **Declarative Control Specification**
- **User-Defined Scope**: Users explicitly specify what aspects of training can be controlled
- **Type-Safe Handlers**: Configuration changes are type-checked and validated
- **Handler Registration**: Clean, decorator-based or configuration-based handler registration

```python
# Users specify what can be controlled
@weavelet_controllable("optimizer_type")
def update_optimizer(self, new_type: str):
    # Handler automatically called when weaver changes optimizer_type
    self.switch_optimizer(new_type)

@weavelet_controllable("learning_rate") 
def update_lr(self, new_lr: float):
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = new_lr
```

### 3. **Process Isolation**
- **Non-Interference**: Weavelet operations run in a separate process to avoid blocking training
- **Fault Tolerance**: Training continues even if weavelet process fails
- **Resource Independence**: NATS communication and async operations don't affect training performance

### 4. **Reliable Communication**
- **Queue-Based IPC**: Use proven multiprocessing.Queue for inter-process communication
- **Message Ordering**: Configuration updates are processed in order
- **Graceful Degradation**: System continues working even with communication failures

### 5. **Bidirectional Flow**
- **Configuration Inbound**: Receive dynamic configuration changes from weaver
- **Status Outbound**: Publish training metrics and status back to weaver
- **Event-Driven**: React to both configuration changes and training events

## Architecture Components

### Weavelet (User Interface Layer)
**Purpose**: Primary API that users interact with in their training code

**Responsibilities**:
- Manage separate process lifecycle (start/stop/cleanup)
- Provide simple methods for config checking and status publishing
- Abstract away all multiprocessing complexity
- Maintain backward compatibility

**Design Goals**:
- **Simplicity**: Minimal API surface area
- **Reliability**: Robust process management with graceful shutdown
- **Performance**: Non-blocking operations for training loop

### AsyncWeavelet (Communication Layer)  
**Purpose**: Handle all weaver communication in isolated process

**Responsibilities**:
- NATS/JetStream connection management
- Subscribe to configuration update streams
- Device registration and replica mapping
- Status publishing to weaver
- Queue-based communication with main process

**Design Goals**:
- **Isolation**: Complete separation from training process
- **Async Efficiency**: Use asyncio for concurrent NATS operations
- **Resilience**: Automatic reconnection and error recovery

### Training Integration (Application Layer)
**Purpose**: Seamless integration patterns for different training frameworks

**Responsibilities**:
- Periodic configuration checking (non-blocking)
- Handler dispatch for configuration changes
- Status collection and publishing
- Lifecycle coordination with training process

**Design Goals**:
- **Minimal Overhead**: Configuration checks should not slow training
- **Flexibility**: Support different integration patterns
- **Extensibility**: Easy to add new controllable parameters

## Current Implementation Analysis

### Strengths
✅ **Process Isolation**: Separate process prevents interference with training  
✅ **Simple API**: Clean interface with `start()`, `get_config_update()`, `publish_training_status()`  
✅ **Reliable IPC**: multiprocessing.Queue provides robust communication  
✅ **Framework Agnostic**: Works with any PyTorch training code  
✅ **Graceful Lifecycle**: Proper startup, shutdown, and cleanup  
✅ **Decorator-Based Handlers**: Clean `@weavelet.handler("config_key")` registration
✅ **Automatic Handler Dispatch**: Config changes automatically trigger handlers
✅ **Type Validation**: Runtime type checking and conversion
✅ **Enhanced Lightning Integration**: `EnhancedWeaveletLightningModule` for automatic integration
✅ **Comprehensive Testing**: 6 test cases covering all functionality

### Phase 1 Complete: Enhanced Handler System ✅
✅ **Manual Polling → Event-Driven**: COMPLETED - Automatic handler dispatch implemented  
✅ **Manual Handlers → Declarative**: COMPLETED - Decorator-based handlers with @weavelet_handler  
🔄 **Configuration Checking → Specification**: PARTIAL - Type validation implemented, full specification in Phase 2  

### Phase 2 Opportunities: Declarative Configuration
🔄 **Enhanced Type System**: Support for complex types (lists, dicts, custom classes)
🔄 **Configuration Constraints**: Min/max validation, enum restrictions  
🔄 **Automatic Documentation**: Generate config docs from handler declarations
🔄 **Framework Templates**: One-line integration patterns for different frameworks

## Enhanced Design Vision

### Declarative Configuration Control
```