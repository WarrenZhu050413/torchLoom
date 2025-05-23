# torchLoom Onboarding Guide

Welcome to torchLoom! This guide will help you understand, set up, and start working with torchLoom - a runtime monitoring and control system for distributed AI training workloads.

## 📖 Project Overview

### What is torchLoom?

torchLoom is a runtime monitoring and control system for distributed AI training workloads, inspired by [torchFT](https://github.com/pytorch/torchft). Named after the Mesopotamian god who governed the universe, torchLoom provides comprehensive monitoring and dynamic control capabilities for PyTorch training jobs.

**Key Differences from Traditional Orchestration:**
- **Runtime Focus**: Unlike deployment/scheduling frameworks, torchLoom monitors and controls active training processes
- **Dynamic Control**: Adjust parameters without stopping training
- **Device-Replica Mapping**: Translates between hardware devices and training replicas
- **Real-time Failure Detection**: Immediate response to GPU and training replica failures

### Core Capabilities

- ✅ **Real-time failure detection** and recovery for GPUs and training replicas
- ✅ **Dynamic configuration adjustment** without stopping training  
- ✅ **Resource mapping** between hardware devices and training replicas
- ✅ **Lightweight messaging** through NATS pub/sub system
- ✅ **Web UI** for monitoring and control
- ✅ **CLI tools** for manual testing and control

## 🏗️ Architecture Overview

### Key Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitor CLI   │    │      Web UI     │    │  Training Jobs  │
│                 │    │                 │    │  (PyTorch)      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │              ┌───────▼───────┐              │
          │              │               │              │
          └──────────────►    WEAVER     ◄──────────────┘
                         │  (Central     │
                         │  Controller)  │
                         └───────┬───────┘
                                 │
                    ┌────────────▼────────────┐
                    │    NATS JetStream      │
                    │   (Message Broker)     │
                    └─────────────────────────┘
```

### Component Details

1. **Weaver** (`torchLoom/weaver/`): Central service maintaining device-to-replica mappings and coordinating failure responses
2. **Weavelet** (`torchLoom/weavelet/`): Client component that integrates with training processes
3. **NATS Messaging**: Lightweight pub/sub communication layer for system events
4. **Monitor CLI** (`torchLoom/cli.py`): Command-line interface for manual control and testing
5. **Web UI** (`torchLoom-ui/`): Vue.js frontend for monitoring and control

### Message Flow Architecture

torchLoom uses a clear message flow pattern organized around the **Weaver** as the central coordinator:

#### Inbound Messages (TO Weaver)
- **From Weavelets/Training Processes**:
  - Device registration events
  - Heartbeat messages (process liveness)
  - Training status updates
  - GPU status reports

- **From External Monitoring Systems**:
  - GPU failure notifications
  - Network status updates

- **From UI**:
  - Control commands (pause, resume, config changes)
  - GPU activation/deactivation requests

#### Outbound Messages (FROM Weaver)
- **To UI**:
  - Consolidated status updates
  - Training progress reports

- **To Weavelets/Training Processes**:
  - Failure notifications
  - Control commands
  - Configuration updates

### Device-Replica Mapping (DRMap)

The core data structure that enables torchLoom to translate between:
- **Monitoring system events** (device_uuid) 
- **Training system events** (replica_id)

Implemented as flexible many-to-many mappings:
- `device_to_replicas`: Hardware device IDs → Training replica IDs
- `replica_to_devices`: Training replica IDs → Hardware device IDs

## 🚀 Getting Started

### Prerequisites

- **Operating System**: Linux (recommended) or macOS
- **Python**: 3.11+
- **Conda**: For environment management
- **Node.js**: For UI development (optional)

### Installation

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd torchLoom
```

#### 2. Environment Setup

**For Ubuntu/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**For macOS:**
```bash
# Install conda environment
conda env create -f environment.yaml

# Install NATS tools
brew install nats-io/nats-tools/nats-tools

# Install protobuf compiler
brew install protobuf
```

#### 3. Activate Environment

```bash
conda init bash && source ~/.bashrc
conda activate nats-torch27
```

### Quick Start Guide

#### Step 1: Start NATS Server
```bash
./nats/nats-server -c ./nats/nats.conf
```

#### Step 2: Start the Weaver (Central Controller)
```bash
python -m torchLoom.weaver
```

#### Step 3: Start Monitor CLI (Optional)
```bash
python -m torchLoom.monitor_cli
```

#### Step 4: Run Training Example
```bash
python examples/pytorch/mnist.py
```

#### Step 5: Start Web UI (Optional)
```bash
# Using the integrated launcher
python start_torchloom.py

# Or start UI separately
cd torchLoom-ui
npm install
npm run dev
```

## 📁 Project Structure

```
torchLoom/
├── torchLoom/                 # Main package
│   ├── weaver/               # Central controller components
│   │   ├── core.py          # Main Weaver class
│   │   ├── inbound_handlers.py   # Handlers for incoming messages
│   │   ├── outbound_handlers.py  # Handlers for outgoing messages
│   │   └── subscription.py  # NATS subscription management
│   ├── weavelet/            # Client-side components
│   ├── proto/               # Protocol buffer definitions
│   ├── log/                 # Logging utilities
│   ├── cli.py              # Monitor CLI
│   ├── constants.py        # System constants
│   └── config.py           # Configuration
├── examples/
│   ├── pytorch/            # PyTorch integration examples
│   └── torchft/           # TorchFT integration examples
├── torchLoom-ui/          # Vue.js web interface
├── docs/                  # Documentation
│   ├── design/           # Architecture docs
│   └── tutorial/         # Usage tutorials
├── tests/                # Test suite
├── nats/                 # NATS server configuration
└── environment.yaml      # Conda environment
```

### Handler Organization

The Weaver uses a clean separation of message handling responsibilities:

#### Inbound Handlers (`inbound_handlers.py`)
Handle messages **TO** the Weaver:

**Weavelet Handlers** (Training Process → Weaver):
- `DeviceRegistrationHandler`: Device registration from training processes
- `HeartbeatHandler`: Process liveness tracking
- `TrainingStatusHandler`: Training progress updates
- `GPUStatusHandler`: GPU status and metrics

**External Handlers** (External Systems → Weaver):
- `FailureHandler`: GPU failures from monitoring systems
- `NetworkStatusHandler`: Network status from external monitors

**UI Handlers** (UI → Weaver):
- `UICommandHandler`: Control commands from web interface
- `ConfigurationHandler`: Configuration changes (legacy support)

#### Outbound Handlers (`outbound_handlers.py`)
Handle messages **FROM** the Weaver:

**UI Handlers** (Weaver → UI):
- `UIUpdateHandler`: Consolidated status updates to web interface

**Weavelet Handlers** (Weaver → Training Processes):
- `WeaveletCommandHandler`: Commands and notifications to training processes
- `HeartbeatMonitor`: Dead replica detection and failure event publishing

## 🧪 Testing the System

### Basic Functionality Test

1. **Device Registration Test**: Start a training job and verify device registration in Weaver logs
2. **Failure Simulation**: Use CLI to simulate device failures
3. **Dynamic Configuration**: Change learning rates during training

### Example Test Scenario

```bash
# In Monitor CLI
torchLoom> test GPU-307a982d-bf2b-4cc3-64e3-aae456bf6a28

# Expected Weaver output:
# [EXTERNAL->WEAVER] GPU failure detected: GPU-307a982d-bf2b-4cc3-64e3-aae456bf6a28
# [EXTERNAL->WEAVER] Associated replicas: {'train_ddp_1:b584d120-...'}
```

## 🔧 Development Guidelines

### Code Organization

The handler reorganization follows these principles:

1. **Clear Message Flow**: Separate inbound (TO weaver) and outbound (FROM weaver) handlers
2. **Source-based Grouping**: Group handlers by message source (weavelet, UI, external)
3. **Single Responsibility**: Each handler has one clear purpose
4. **Consistent Logging**: Use directional logging prefixes (`[WEAVELET->WEAVER]`, `[WEAVER->UI]`, etc.)

### Key Development Practices

1. **Plan First**: Create detailed implementation plans in `plan.md`
2. **Prioritize Simplicity**: Favor simple, error-free approaches
3. **Review and Refactor**: Review code for correctness and modularity
4. **Don't Alter Tests**: Fix code, not tests (unless tests are flawed)
5. **Explain Decisions**: Document design choices clearly

### Adding New Handlers

When adding new message types:

1. **Identify Direction**: Is this TO the weaver (inbound) or FROM the weaver (outbound)?
2. **Identify Source**: What component is sending/receiving the message?
3. **Choose Base Class**: Inherit from `MessageHandler` (inbound) or `OutboundHandler` (outbound)
4. **Add Logging**: Use consistent directional prefixes in log messages
5. **Update Exports**: Add the new handler to `__init__.py`

### Running Tests

```bash
# Activate environment first
conda activate nats-torch27

# Run test suite
pytest torchLoom/tests/
```

## 📚 Key Concepts

### NATS JetStream Integration

torchLoom uses NATS JetStream for reliable messaging:
- **Subjects**: Logical channels for different message types
- **Streams**: Persistent storage for messages
- **Consumers**: Processing units that consume messages

### Message Flow Patterns

1. **Training Registration**: Training processes register with Weaver via weavelet handlers
2. **Device Mapping**: Weaver maintains device-to-replica mappings using `DeviceReplicaMapper`
3. **Failure Detection**: External monitors report failures via external handlers
4. **Recovery Coordination**: Weaver notifies affected replicas via outbound handlers
5. **Dynamic Control**: UI sends commands via UI handlers, weaver executes via outbound handlers

### Configuration Management

- Environment variables: `torchLoom_ADDR` for NATS server address
- Default configuration in `torchLoom/config.py`
- Runtime constants in `torchLoom/constants.py`

## 🎯 Common Use Cases

### 1. Basic Training Monitoring
Monitor PyTorch training jobs with automatic device registration and failure detection.

### 2. Dynamic Parameter Adjustment
Change learning rates, batch sizes, or other parameters during training without restart.

### 3. Multi-GPU Failure Recovery
Automatically detect and recover from GPU failures in distributed training.

### 4. Resource Optimization
Monitor resource usage and adjust training configuration based on real-time metrics.

## 🐛 Troubleshooting

### Common Issues

1. **NATS Connection Failures**
   - Verify NATS server is running: `nats server check`
   - Check firewall settings and port 4222 availability

2. **Environment Issues**
   - Ensure conda environment is activated: `conda activate nats-torch27`
   - Verify Python version: `python --version` (should be 3.11+)

3. **Import Errors**
   - Check protobuf installation: `protoc --version`
   - Verify all dependencies: `pip list`

### Getting Help

- Check existing documentation in `docs/`
- Review test files in `tests/` for usage examples
- Examine example training scripts in `examples/`

## 🔗 Related Documentation

- [Design Document](docs/design/design.md): Detailed system architecture
- [Tutorial](docs/tutorial/tutorial.md): Step-by-step testing procedures
- [TODO](TODO.md): Development roadmap and upcoming features
- [Contributing Guidelines](CONTRIBUTING.md): Contribution standards and processes

## 🎉 Next Steps

1. **Follow the Quick Start**: Get the system running with the MNIST example
2. **Explore Examples**: Check out PyTorch and TorchFT integration examples
3. **Read the Design**: Understand the architecture from `docs/design/design.md`
4. **Run Tests**: Execute the test suite to verify your setup
5. **Join Development**: Check `TODO.md` for areas where you can contribute

Welcome to the torchLoom community! 🧵✨ 