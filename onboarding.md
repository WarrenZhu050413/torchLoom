# torchLoom Onboarding Guide

Welcome to torchLoom! This guide will help you understand, set up, and start working with torchLoom - a runtime monitoring and control system for distributed AI training workloads.

## 🤖 AI Agent Navigation Guide

### Essential Files to Know

**Core Communication Components:**
- `torchLoom/threadlet/core.py` - Client-side component that integrates with training processes
- `torchLoom/weaver/core.py` - Central controller that orchestrates all communication
- `torchLoom/weaver/websocket_server.py` - WebSocket server for real-time UI communication
- `torchLoom/common/constants.py` - All NATS subjects and communication constants
- `torchLoom-ui/src/stores/training.js` - Frontend state management and API integration
- `torchLoom-ui/src/services/api.js` - Frontend API service for WebSocket/REST communication

**Handler Organization:**
- `torchLoom/weaver/handlers/` - Message handlers for different communication directions
- `torchLoom/weaver/publishers/` - Outbound message publishers
- `torchLoom/threadlet/handlers/` - Client-side configuration handlers

**Protocol Definitions:**
- `torchLoom/proto/torchLoom_pb2.py` - Protobuf message definitions for NATS communication

### Message Flow Architecture

For a visual representation of the overall system, please refer to the [Architecture Diagram](docs/design/architecture.md#architecture-diagram).

#### 🔄 Training Process → UI (Status Updates)

**1. Training Process → Threadlet**
```python
# In training code, threadlet publishes status updates
threadlet.publish_status({
    "replica_id": "train_ddp_1:uuid",
    "status_type": "batch_update", 
    "current_step": 150,
    "metrics": {"loss": 0.25, "learning_rate": 0.001}
})
```

**2. Threadlet → Weaver (via NATS)**
- **Subject:** `torchLoomConstants.subjects.TRAINING_STATUS` ("torchLoom.training.status")
- **Handler:** `ThreadletHandler.handle()` in `torchLoom/weaver/handlers/`
- **Message Type:** `EventEnvelope.training_status`

**3. Weaver → UI (via WebSocket)**
- **Publisher:** `UIUpdatePublisher.publish_ui_update()` 
- **WebSocket:** `WebSocketServer.broadcast_status_update()`
- **Message Type:** `{"type": "status_update", "data": {...}}`

**4. UI Receives Update**
```javascript
// In training.js
function handleWebSocketMessage(message) {
  if (message.type === 'status_update') {
    updateFromBackendData(message.data)
  }
}
```

#### 🔄 UI → Training Process (Commands)

**1. UI → Weaver (via WebSocket)**
```javascript
// In training.js via api.js
await apiService.updateConfig(replicaId, {
  learning_rate: 0.002,
  batch_size: 128
})
```

**2. WebSocket → Weaver**
- **Handler:** `WebSocketServer.handle_websocket_message()`
- **NATS Subject:** `torchLoomConstants.subjects.UI_COMMAND` ("torchLoom.ui.commands")
- **Message Type:** `EventEnvelope.ui_command`

**3. Weaver → Threadlet (via NATS)**
- **Publisher:** `ThreadletCommandPublisher.publish_command()`
- **Subject:** `torchLoomConstants.subjects.WEAVER_COMMANDS` ("torchLoom.weaver.commands")
- **Message Type:** `EventEnvelope.weaver_command`

**4. Threadlet → Training Process**
```python
# In threadlet core.py
config_update = threadlet.get_config_update()
if config_update:
    # Automatically dispatches to registered handlers
    # e.g., @threadlet.handler("learning_rate")
    def update_lr(new_lr):
        optimizer.param_groups[0]['lr'] = new_lr
```

### Key NATS Communication Subjects

**Training Process → Weaver:**
- `torchLoom.training.status` - Training progress and metrics
- `torchLoom.device.status` - device utilization and health
- `torchLoom.heartbeat` - Process liveness monitoring
- `torchLoom.weaver.events` - Device registration (via JetStream)

**UI ↔ Weaver:**
- `torchLoom.ui.commands` - Control commands from UI
- `torchLoom.ui.update` - Status broadcasts to UI (via WebSocket, not NATS)

**Weaver → Threadlet:**
- `torchLoom.weaver.commands` - Configuration updates and control commands
- `torchLoom.config.info` - Configuration change notifications

**External Systems → Weaver:**
- `torchLoom.monitored.failure` - Hardware failure notifications
- `torchLoom.replica.fail` - Replica failure events

### Message Handler Routing

**In Weaver (`torchLoom/weaver/core.py`):**
```python
async def message_handler(self, msg: Msg):
    env = EventEnvelope()
    env.ParseFromString(msg.data)
    
    # Route to consolidated handlers
    if env.HasField("training_status") or env.HasField("device_status"):
        await self._handlers["threadlet"].handle(env)  # ThreadletHandler
    
    if env.HasField("ui_command"):
        await self._handlers["ui"].handle(env)        # UIHandler
    
    if env.HasField("monitored_fail"):
        await self._handlers["external"].handle(env)  # ExternalHandler
```

**Handler Classes:**
- `ThreadletHandler` - Processes training status, device status, heartbeats, device registration
- `UIHandler` - Processes UI commands (deactivate device, reactivate group, config updates)
- `ExternalHandler` - Processes external failure notifications and config changes

### Configuration Update Flow Example

**Complete flow for changing learning rate:**

1. **UI sends command:**
```javascript
apiService.updateConfig("replica_group_1", {learning_rate: 0.002})
```

2. **WebSocket message:**
```json
{
  "type": "update_config",
  "replica_id": "replica_group_1", 
  "config_params": {"learning_rate": "0.002"}
}
```

3. **NATS UI command:**
```protobuf
EventEnvelope {
  ui_command {
    command_type: "update_config"
    target_id: "replica_group_1"
    params: {"learning_rate": "0.002"}
  }
}
```

4. **NATS weaver command:**
```protobuf
EventEnvelope {
  weaver_command {
    command_type: "config_update"
    target_replica: "replica_group_1"
    config_params: {"learning_rate": "0.002"}
  }
}
```

5. **Threadlet handler dispatch:**
```python
@threadlet.handler("learning_rate")
def update_learning_rate(new_lr: float):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    logger.info(f"Updated learning rate to {new_lr}")
```