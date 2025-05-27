# torchLoom Onboarding Guide

Welcome to torchLoom! This guide will help you understand, set up, and start working with torchLoom - a runtime monitoring and control system for distributed AI training workloads.

## 🤖 AI Agent Navigation Guide

### Essential Files to Know

**Core Communication Components:**
- `torchLoom/threadlet/threadlet.py` - Client-side component (Threadlet) that integrates with training processes, manages configuration, and sends status updates.
- `torchLoom/weaver/weaver.py` - Central controller (Weaver) that orchestrates all communication, message handling, and subscriptions.
- `torchLoom/weaver/websocket_server.py` - WebSocket server for real-time UI communication.
- `torchLoom/common/constants.py` - Defines all NATS subjects, stream configurations, and communication constants.
- `torchLoom/proto/torchLoom.proto` - Protocol Buffer message definitions for NATS communication (compiled to `torchLoom_pb2.py`).

**Unified Handler System:**
- `torchLoom/common/handlers.py` - Unified handler registry system used across all components for consistent event handling.
- `torchLoom/weaver/handlers.py` - Weaver-side message handlers for different communication sources (Threadlets, UI, External Systems).
- `torchLoom/threadlet/listener.py` - Async listener in the Threadlet process for NATS communication and inter-process communication with the main training code.

**State Management & UI Interface:**
- `torchLoom/weaver/status_tracker.py` - Pure state management for device and training status, with no external dependencies.
- `torchLoom/weaver/ui_interface.py` - UI-specific functionality including WebSocket broadcasting and status formatting.
- `torchLoom-ui/src/stores/training.js` - Pinia store for frontend state management and WebSocket message handling.
- `torchLoom-ui/src/services/api.js` - Frontend WebSocket-only service for real-time communication.

**Publisher Organization:**
- `torchLoom/common/publishers.py` - Common publisher functionality shared between components.
- `torchLoom/weaver/publishers.py` - Weaver-side publishers for sending commands to Threadlets.
- `torchLoom/threadlet/publishers.py` - Threadlet-side publishers for sending events to the Weaver.

**Subscription Management:**
- `torchLoom/common/subscription.py` - NATS and JetStream subscription management with connection lifecycle handling.

### Architecture Overview

The torchLoom architecture follows a clean, event-driven design:

```
┌─────────────────┐     WebSocket      ┌─────────────┐
│       UI        │◄──────────────────►│   Weaver    │
│  (Vue + Pinia)  │                    │             │
└─────────────────┘                    └──────┬──────┘
                                              │
                                          NATS│JetStream
                                              │
                                       ┌──────┴──────┐
                                       │  Threadlet  │
                                       │  (Process)  │
                                       └──────┬──────┘
                                              │
                                       ┌──────┴──────┐
                                       │  Training   │
                                       │   Process   │
                                       └─────────────┘
```

### Message Flow Architecture

#### 🔄 Training Process → UI (Status Updates)

**1. Training Process → Threadlet**
```python
# In training code, threadlet publishes status updates
threadlet.publish_metrics(
    step=150,
    epoch=5,
    loss=0.25,
    accuracy=0.95,
    learning_rate=0.001
)
```

**2. Threadlet → Weaver (via NATS)**
- **Subject:** `torchLoom.threadlet.events`
- **Handler:** `handle_training_status()` in `torchLoom/weaver/handlers.py`
- **Message Type:** `EventEnvelope.training_status`

**3. Weaver → UI (via WebSocket)**
- **StatusTracker:** Updates internal state and triggers UI notification callback
- **UINotificationManager:** Broadcasts status update to all connected WebSocket clients
- **Message Format:** `{"type": "status_update", "data": {...}}`

**4. UI Receives Update**
```javascript
// In torchLoom-ui/src/stores/training.js
function handleWebSocketMessage(message) {
  if (message.type === 'status_update') {
    updateLocalState(message.data)
  }
}
```

#### 🔄 UI → Training Process (Commands)

**1. UI → Weaver (via WebSocket)**
```javascript
// UI sends command via WebSocket
apiService.updateConfig('replica_1', { learning_rate: 0.002 })

// In api.js - sends WebSocket message
this.send({
  type: 'update_config',
  process_id: 'replica_1',
  config_params: { learning_rate: 0.002 }
})
```

**2. WebSocket Server → Weaver**
- **Handler:** `WebSocketServer.handle_ui_message()` converts to UI command format
- **Dispatch:** Routes to `handle_ui_command()` via HandlerRegistry
- **Message Type:** `EventEnvelope.ui_command`

**3. Weaver → Threadlet (via NATS)**
- **Publisher:** `ThreadletCommandPublisher.publish_weaver_command()`
- **Subject:** `torchLoom.weaver.commands`
- **Message Type:** `EventEnvelope.weaver_command`

**4. Threadlet → Training Process**
```python
# Threadlet receives command and dispatches to registered handlers
def _handle_command_message(self, message):
    if message.command_type == CommandType.UPDATE_CONFIG:
        self._dispatch_handlers(params)  # Uses HandlerRegistry
```

### Key Design Patterns

#### 1. **Unified Handler Registry**
All components use the same `HandlerRegistry` pattern for consistent message handling:
```python
registry = HandlerRegistry("weaver_main")
registry.register_handler("training_status", handle_training_status)
registry.dispatch_handler("training_status", event_data)
```

#### 2. **Event-Driven UI Updates**
UI updates are triggered by state changes, not polling:
```python
# StatusTracker notifies UI when state changes
self._notify_change()  # Triggers UINotificationManager.notify_status_change()
```

#### 3. **Clean State Management**
StatusTracker provides a clean API with both proto and convenience methods:
```python
# Proto method (for incoming messages)
status_tracker.update_device_status_from_proto(device_proto)

# Convenience method (for handlers)
status_tracker.update_device_status(device_uuid, status="active", utilization=75.0)
```

#### 4. **WebSocket-Only Communication**
All UI communication uses WebSockets for real-time updates:
- No REST endpoints
- Automatic reconnection
- Message conversion from simple formats to proto

### Configuration Update Flow Example

**Complete flow for changing learning rate:**

1. **UI sends command:**
   ```javascript
   apiService.updateConfig("replica_1", {learning_rate: "0.002"})
   ```

2. **WebSocket Server converts:**
   - From: `{type: "update_config", process_id: "replica_1", config_params: {...}}`
   - To: UI command format for protobuf

3. **Handler dispatches:**
   ```python
   UI_COMMAND_HANDLERS["update_config"] → handle_update_config()
   ```

4. **Weaver publishes:**
   ```python
   await publish_weaver_command("update_config", "replica_1", {"learning_rate": "0.002"})
   ```

5. **Threadlet receives:**
   - Message arrives via NATS
   - Converted to command message
   - Dispatched to training process

6. **Training updates:**
   ```python
   # Registered handler executes
   optimizer.param_groups[0]['lr'] = 0.002
   ```

## 🚀 Quick Start

### 1. Start NATS Server
```bash
./nats/nats-server
```

### 2. Start the Weaver
```bash
python -m torchLoom.weaver.weaver
```

### 3. Run Training with torchLoom
```bash
# Run the interactive training example
python examples/interactive/train_interactive.py
```

### 4. Open the UI
Navigate to `http://localhost:8080` to see real-time training status and control the training process.

## 🧪 Testing the System

The best way to test torchLoom is with the interactive training example:

```bash
cd examples/interactive
python train_interactive.py --epochs 10 --learning-rate 0.01
```

**Features you can test:**
- ✅ Real-time metrics updates in the UI
- ✅ Pause/resume training via UI buttons
- ✅ Update learning rate on the fly
- ✅ Device status monitoring
- ✅ Multi-replica coordination

## 🏗️ Architecture Benefits

1. **Clean Separation of Concerns**
   - Each module has a single, well-defined responsibility
   - Easy to test and maintain individual components

2. **Real-Time Communication**
   - WebSocket-only for immediate updates
   - Event-driven architecture reduces latency

3. **Flexible Configuration**
   - Register any parameter for dynamic updates
   - Type-safe handler registration

4. **Production Ready**
   - Comprehensive error handling
   - Automatic reconnection
   - Clean shutdown procedures

## 📚 Next Steps

1. **Explore Examples**
   - `examples/interactive/train_interactive.py` - Complete training integration
   - `examples/pytorch/train_fed.py` - Federated learning example

2. **Extend Functionality**
   - Add custom handlers for your training parameters
   - Create external monitoring integrations
   - Build custom UI dashboards

3. **Deploy at Scale**
   - Configure NATS clustering for high availability
   - Use JetStream for persistent message delivery
   - Monitor system health via built-in metrics

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to torchLoom.

The architecture is designed for extensibility - new handlers, publishers, and UI components can be added without modifying core infrastructure.