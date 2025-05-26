# torchLoom Onboarding Guide

Welcome to torchLoom! This guide will help you understand, set up, and start working with torchLoom - a runtime monitoring and control system for distributed AI training workloads.

## 🤖 AI Agent Navigation Guide

### Essential Files to Know

**Core Communication Components:**
- `torchLoom/threadlet/core.py` - Client-side component (Threadlet) that integrates with training processes, manages configuration, and sends status updates.
- `torchLoom/weaver/core.py` - Central controller (Weaver) that orchestrates all communication, message handling, and subscriptions.
- `torchLoom/weaver/websocket_server.py` - WebSocket and REST API server for real-time UI communication.
- `torchLoom/common/constants.py` - Defines all NATS subjects, stream configurations, and communication constants.
- `torchLoom/proto/torchLoom.proto` - Protocol Buffer message definitions for NATS communication (compiled to `torchLoom_pb2.py`).
- `torchLoom-ui/src/stores/training.js` - Pinia store for frontend state management, API integration, and WebSocket message handling in the UI.
- `torchLoom-ui/src/services/api.js` - Frontend service for WebSocket and REST API communication with the backend.

**Handler Organization:**
- `torchLoom/weaver/handlers.py` - Weaver-side message handlers for different communication sources (Threadlets, UI, External Systems).
- `torchLoom/weaver/publishers.py` - Weaver-side publishers for sending messages to Threadlets and the UI.
- `torchLoom/threadlet/handlers.py` - Client-side (Threadlet) configuration handler registration and dispatch system.
- `torchLoom/threadlet/listener.py` - Async listener in the Threadlet process for NATS communication and inter-process communication with the main training code.

**Status Tracking:**
- `torchLoom/weaver/status_tracker.py` - Manages the state of replicas, devices, and training progress within the Weaver.

**Protocol Definitions:**
- `torchLoom/proto/torchLoom.proto` - Source Protocol Buffer message definitions.
- `torchLoom/proto/torchLoom_pb2.py` - Generated Python code from `torchLoom.proto` for NATS communication.

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
- **Subject:** `torchLoomConstants.subjects.TRAINING_STATUS` (defined in `torchLoom/common/constants.py`)
- **Handler:** `ThreadletHandler._handle_training_status()` in `torchLoom/weaver/handlers.py`
- **Message Type:** `EventEnvelope.training_status` (defined in `torchLoom/proto/torchLoom.proto`)

**3. Weaver → UI (via WebSocket)**
- **Publisher:** `UIUpdatePublisher.publish_ui_update()` in `torchLoom/weaver/publishers.py` (sends NATS message to `torchLoom.ui.update`)
- **WebSocket Server:** `WebSocketServer` in `torchLoom/weaver/websocket_server.py` listens to NATS subject `torchLoom.ui.update` and broadcasts to connected UI clients.
- **Message Type (WebSocket):** `{"type": "status_update", "data": {...}}` or individual types like `{"type": "training_status", "data": {...}}`

**4. UI Receives Update**
```javascript
// In torchLoom-ui/src/stores/training.js
// (Simplified from handleWebSocketMessage and other update functions)
function handleStatusUpdate(messageData) {
  // Updates replicaGroups, deviceStatuses, etc.
}
```

#### 🔄 UI → Training Process (Commands)

**1. UI → Weaver (via WebSocket/REST)**
```javascript
// In a Vue component, calls a method in torchLoom-ui/src/stores/training.js
// trainingStore.updateConfig(replicaId, { learning_rate: 0.002 })

// In torchLoom-ui/src/stores/training.js
async function updateConfig(replicaId, configParams) {
  await apiService.updateConfig(replicaId, configParams)
}

// In torchLoom-ui/src/services/api.js
async updateConfig(replicaId, configParams) {
  // Sends WebSocket message: { type: 'update_config', replica_id: replicaId, config_params: configParams }
  // AND/OR makes a POST request to /api/commands/update-config
  this.send({ type: 'update_config', replica_id: replicaId, config_params: configParams });
  // await fetch(...);
}
```

**2. WebSocket/REST → Weaver**
- **WebSocket Handler:** `WebSocketServer.handle_websocket_message()` in `torchLoom/weaver/websocket_server.py`, which may call internal handlers like `handle_config_update()`.
- **REST Handler:** FastAPI routes in `torchLoom/weaver/websocket_server.py` (e.g., `@app.post("/api/commands/update-config")`).
- These handlers in `websocket_server.py` then typically interact with the `UIHandler` in `torchLoom/weaver/handlers.py` or directly publish to NATS.
- **NATS Subject (if used for internal routing or fallback):** `torchLoomConstants.subjects.UI_COMMANDS`
- **Handler for NATS UI_COMMANDS:** `UIHandler._handle_ui_command()` in `torchLoom/weaver/handlers.py`
- **Message Type (NATS):** `EventEnvelope.ui_command`

**3. Weaver → Threadlet (via NATS)**
- **Publisher:** `ThreadletCommandPublisher.publish_weaver_command()` in `torchLoom/weaver/publishers.py` (called by `UIHandler`).
- **Subject:** `torchLoomConstants.subjects.WEAVER_COMMANDS` (defined in `torchLoom/common/constants.py`)
- **Message Type:** `EventEnvelope.weaver_command` (defined in `torchLoom/proto/torchLoom.proto`)

**4. Threadlet → Training Process**
```python
# In torchLoom/threadlet/listener.py
# _handle_weaver_command receives the NATS message
# and sends it to the main Threadlet process via an internal pipe.

# In torchLoom/threadlet/core.py
# get_config_update() or check_and_apply_updates() receives data from the pipe.
config_update_or_command = threadlet.get_config_update() 
if config_update_or_command and config_update_or_command.get("_is_weaver_command"):
    command_type = config_update_or_command.get("command_type")
    params = config_update_or_command.get("params")
    # Dispatch command, e.g., if command_type == "config_update":
    #   threadlet._dispatch_handlers(params) 
    #   which then calls registered handlers:
    #     @threadlet.handler("learning_rate")
    #     def update_lr(new_lr):
    #         optimizer.param_groups[0]['lr'] = new_lr
elif config_update_or_command:
    # This is a direct config update (not from a weaver_command)
    threadlet._dispatch_handlers(config_update_or_command)
```

### Key NATS Communication Subjects
(Refer to `torchLoom/common/constants.py` for the source of truth)

**Training Process (Threadlet) → Weaver:**
- `torchLoom.training.status` - Training progress and metrics.
- `torchLoom.device.status` - Device utilization and health.
- `torchLoom.heartbeat` - Process liveness monitoring from Threadlets.
- `torchLoom.weaver.events` / `torchLoomConstants.weaver_stream.subjects.DR_SUBJECT` - Device registration (via JetStream, typically `torchLoom.DRentry`).

**UI ↔ Weaver:**
- `torchLoom.ui.commands` - Control commands from UI to Weaver (typically via WebSocket first, then potentially NATS).
- `torchLoom.ui.update` - Status broadcasts from Weaver to UI (Weaver publishes to this NATS subject, WebSocket server subscribes and forwards to UI clients).

**Weaver → Training Process (Threadlet):**
- `torchLoom.weaver.commands` - Configuration updates and control commands (e.g., pause, resume, update_config).
- `torchLoom.config.info` - General configuration change notifications (can be published by Weaver and received by Threadlets).

**External Systems → Weaver:**
- `torchLoom.monitored.failure` - Hardware failure notifications.
- `torchLoom.replica.fail` - Replica failure events (can be published by Weaver if a Threadlet is unresponsive, and received by other Threadlets).


### Message Handler Routing

**In Weaver (`torchLoom/weaver/core.py`):**
```python
# Simplified from Weaver.message_handler
async def message_handler(self, msg: Msg):
    env = EventEnvelope()
    env.ParseFromString(msg.data)
    
    # Route to consolidated handlers based on message content
    if env.HasField("register_device") or \
       env.HasField("heartbeat") or \
       env.HasField("training_status") or \
       env.HasField("device_status") or \
       env.HasField("drain"):
        await self._handlers["threadlet"].handle(env)  # ThreadletHandler in torchLoom/weaver/handlers.py
    
    elif env.HasField("ui_command"): # Typically received if WebSocketServer forwards to NATS
        await self._handlers["ui"].handle(env)        # UIHandler in torchLoom/weaver/handlers.py
    
    elif env.HasField("monitored_fail") or \
         env.HasField("config_info"): # config_info can also be published by UI via UIHandler
        await self._handlers["external"].handle(env)  # ExternalHandler in torchLoom/weaver/handlers.py
```

**Handler Classes (in `torchLoom/weaver/handlers.py`):**
- `ThreadletHandler` - Processes device registration, heartbeats, training status, device status, and drain events from Threadlets.
- `UIHandler` - Processes UI commands (e.g., deactivate device, reactivate group, update config, pause/resume training) which usually originate from WebSocket messages. It then often publishes `WeaverCommand` messages to Threadlets.
- `ExternalHandler` - Processes external failure notifications (e.g., `monitored_fail`) and general configuration changes (`config_info`).

### Configuration Update Flow Example

**Complete flow for changing learning rate:**

1.  **UI sends command:**
    A Vue component calls `trainingStore.updateConfig("replica_group_1", {learning_rate: "0.002"})`.
    This calls `apiService.updateConfig` in `torchLoom-ui/src/services/api.js`.

2.  **`api.js` sends to WebSocket Server:**
    `apiService.send({ type: 'update_config', replica_id: 'replica_group_1', config_params: {learning_rate: '0.002'} })`
    (It might also make a REST POST request to `/api/commands/update-config`)

3.  **WebSocket Server (`torchLoom/weaver/websocket_server.py`) handles:**
    - `handle_websocket_message` receives the JSON.
    - Calls `self.handle_config_update("replica_group_1", {learning_rate: "0.002"})`.
    - `handle_config_update` may directly call the Weaver's `UIHandler` or publish a NATS message.
    If publishing to NATS (or as a fallback):
    - **NATS Subject:** `torchLoom.ui.commands`
    - **Message:**
      ```protobuf
      EventEnvelope {
        ui_command {
          command_type: "update_config"
          target_id: "replica_group_1" # This is the replica_id
          params: {"learning_rate": "0.002"}
        }
      }
      ```

4.  **Weaver's `UIHandler` (`torchLoom/weaver/handlers.py`) processes `ui_command`:**
    - `UIHandler._handle_ui_command` is called.
    - It identifies `command_type` as "update_config".
    - Calls `UIHandler._handle_update_config("replica_group_1", {"learning_rate": "0.002"})`.
    - `_handle_update_config` then calls `self._publish_weaver_command("update_config", "replica_group_1", {"learning_rate": "0.002"})`.

5.  **`UIHandler` publishes `WeaverCommand` via `ThreadletCommandPublisher` (`torchLoom/weaver/publishers.py`):**
    - **NATS Subject:** `torchLoom.weaver.commands`
    - **Message:**
      ```protobuf
      EventEnvelope {
        weaver_command {
          command_type: "update_config" # Indicates the type of command for the threadlet
          target_replica_id: "replica_group_1"
          params: {"learning_rate": "0.002"} # The actual configuration to update
        }
      }
      ```

6.  **Threadlet Listener (`torchLoom/threadlet/listener.py`) receives NATS message:**
    - `_handle_weaver_command` gets the `EventEnvelope`.
    - It sees `target_replica_id` matches its own.
    - Sends `{"command_type": "update_config", "params": {"learning_rate": "0.002"}, "_is_weaver_command": True}` to the main Threadlet process via an internal pipe.

7.  **Threadlet Core (`torchLoom/threadlet/core.py`) receives from pipe and dispatches:**
    - `check_and_apply_updates()` (or `get_config_update()`) receives the dictionary.
    - Because `_is_weaver_command` is true and `command_type` is `update_config`, it effectively calls `self._dispatch_handlers({"learning_rate": "0.002"})`.
    - `_dispatch_handlers` (in `torchLoom/threadlet/handlers.py` via `HandlerRegistry`) finds the registered handler for "learning_rate".

8.  **Specific handler in training code is executed:**
    ```python
    # In user's training script or a Threadlet-aware module
    @threadlet.handler("learning_rate") # Registered in torchLoom/threadlet/core.py or by user
    def update_learning_rate(new_lr: float): # Type conversion handled by HandlerRegistry
        # Example: Update PyTorch optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        logger.info(f"Updated learning rate to {new_lr}")
    ```