# torchLoom Weaver Default Handlers

The torchLoom Weaver comes with a comprehensive set of **default handlers** that handle all standard message types out of the box. Users don't need to define handlers manually unless they want custom behavior.

## 🎯 Zero-Configuration Usage

For most use cases, you can use the Weaver without any handler configuration:

```python
from torchLoom.weaver.core import Weaver
import asyncio

async def main():
    # All default handlers are automatically configured
    weaver = Weaver(enable_ui=True)
    await weaver.initialize()
    
    # Start the weaver - all message types are handled!
    # ... your weaver usage code ...
    
    await weaver.stop()

asyncio.run(main())
```

## 📋 Available Default Handlers

The following handlers are automatically configured:

| Handler | Event Type | Purpose |
|---------|------------|---------|
| `DeviceRegistrationHandler` | `register_device` | Handles device registration from weavelets |
| `HeartbeatHandler` | `heartbeat` | Monitors weavelet liveness via heartbeat messages |
| `TrainingStatusHandler` | `training_status` | Processes training progress updates from weavelets |
| `deviceStatusHandler` | `device_status` | Handles device status and utilization updates |
| `NetworkStatusHandler` | `network_status` | Processes network connectivity and performance data |
| `FailureHandler` | `monitored_fail` | Manages device and replica failure scenarios |
| `DrainEventHandler` | `drain` | Handles graceful device drain requests |
| `UICommandHandler` | `ui_command` | Processes commands from the UI (pause/resume/config changes) |
| `WeaverCommandHandler` | `weaver_command` | Handles command acknowledgments from weavelets |
| `ConfigurationHandler` | `config_info` | Manages configuration change events |

## 🔧 Customizing Handlers

### Override Existing Handlers

You can replace any default handler with your custom implementation:

```python
from torchLoom.weaver.core import Weaver
from torchLoom.weaver.handlers import MessageHandler
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

class CustomHeartbeatHandler(MessageHandler):
    def __init__(self, status_tracker, nats_client=None):
        self.status_tracker = status_tracker
        self.nats_client = nats_client
        self.heartbeat_count = 0
    
    async def handle(self, env: EventEnvelope) -> None:
        if not env.HasField("heartbeat"):
            return
            
        self.heartbeat_count += 1
        heartbeat = env.heartbeat
        
        # Custom logic here
        print(f"🔄 Received heartbeat #{self.heartbeat_count} from {heartbeat.replica_id}")
        
        # Call original functionality if needed
        # ... your custom logic ...

async def main():
    weaver = Weaver()
    await weaver.initialize()
    
    # Override the default heartbeat handler
    custom_handler = CustomHeartbeatHandler(weaver.status_tracker)
    weaver.override_handler("heartbeat", custom_handler)
    
    print("Heartbeat handler customized!")
```

### Add Custom Handlers

You can add handlers for custom event types:

```python
class MyCustomHandler(MessageHandler):
    async def handle(self, env: EventEnvelope) -> None:
        # Handle your custom event type
        pass

# Add the custom handler
weaver.add_custom_handler("my_custom_event", MyCustomHandler())
```

## 🔍 Inspecting Handlers

### List All Handlers

```python
handlers = weaver.list_handlers()
print("Current handlers:")
for event_type, handler_class in handlers.items():
    print(f"  {event_type}: {handler_class}")
```

### Get Specific Handler

```python
heartbeat_handler = weaver.get_handler("heartbeat")
print(f"Heartbeat handler: {heartbeat_handler.__class__.__name__}")
```

### View Supported Events

```python
events = weaver.get_supported_events()
print("Supported event types:")
for event_type, description in events.items():
    print(f"  {event_type}: {description}")
```

## 📝 Handler Development Guide

### Creating Custom Handlers

All handlers must inherit from `MessageHandler` and implement the `handle` method:

```python
from torchLoom.weaver.handlers import MessageHandler
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

class MyCustomHandler(MessageHandler):
    def __init__(self, required_deps):
        self.deps = required_deps
    
    async def handle(self, env: EventEnvelope) -> None:
        """Handle a specific message type."""
        
        # Check if this handler should process the message
        if not env.HasField("your_event_type"):
            return
        
        # Extract the event data
        event_data = env.your_event_type
        
        # Process the event
        # ... your custom logic ...
        
        # Log what happened
        logger.info(f"Processed {event_data.some_field}")
```

### Best Practices

1. **Always check `HasField()`**: Ensure your handler only processes the correct event type
2. **Use dependency injection**: Pass required dependencies through the constructor
3. **Handle exceptions**: Wrap your logic in try-catch blocks
4. **Log appropriately**: Use the logger to track what your handler is doing
5. **Keep it simple**: Handlers should focus on a single responsibility

## 🔄 Handler Lifecycle

1. **Initialization**: Handlers are created when `weaver.initialize()` is called
2. **Message Processing**: Handlers process messages as they arrive via NATS
3. **Cleanup**: Handlers are cleaned up when `weaver.stop()` is called

## 📚 Example: Complete Customization

Here's a comprehensive example showing how to customize multiple handlers:

```python
import asyncio
from torchLoom.weaver.core import Weaver
from torchLoom.weaver.handlers import MessageHandler
from torchLoom.proto.torchLoom_pb2 import EventEnvelope
from torchLoom.log.logger import setup_logger

logger = setup_logger(name="custom_handlers")

class CustomTrainingStatusHandler(MessageHandler):
    """Enhanced training status handler with metrics collection."""
    
    def __init__(self, status_tracker):
        self.status_tracker = status_tracker
        self.metrics_collector = {}  # Your custom metrics
    
    async def handle(self, env: EventEnvelope) -> None:
        if not env.HasField("training_status"):
            return
        
        training_status = env.training_status
        replica_id = training_status.replica_id
        
        # Call the default functionality
        self.status_tracker.update_training_progress(
            replica_id=replica_id,
            current_step=training_status.current_step,
            step_progress=training_status.step_progress,
            status=training_status.status,
            last_active_step=training_status.batch_idx,
            fixed_step=None,
        )
        
        # Add your custom metrics collection
        if replica_id not in self.metrics_collector:
            self.metrics_collector[replica_id] = []
        
        self.metrics_collector[replica_id].append({
            'step': training_status.current_step,
            'progress': training_status.step_progress,
            'timestamp': time.time()
        })
        
        logger.info(f"📊 Collected metrics for {replica_id} at step {training_status.current_step}")

async def main():
    weaver = Weaver(enable_ui=False)
    await weaver.initialize()
    
    # Customize specific handlers while keeping others as defaults
    custom_training_handler = CustomTrainingStatusHandler(weaver.status_tracker)
    weaver.override_handler("training_status", custom_training_handler)
    
    print("✅ Weaver initialized with custom training status handler")
    
    # Show the current handler configuration
    handlers = weaver.list_handlers()
    for event_type, handler_class in handlers.items():
        if event_type == "training_status":
            print(f"  {event_type}: {handler_class} ⭐ (custom)")
        else:
            print(f"  {event_type}: {handler_class}")
    
    await weaver.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## ✅ Benefits of Default Handlers

- **Zero Configuration**: Works out of the box for standard use cases
- **Full Coverage**: All message types are handled by default
- **Easy Customization**: Override only what you need to change
- **Backward Compatibility**: Existing code continues to work
- **Best Practices**: Default implementations follow torchLoom conventions
- **Production Ready**: Thoroughly tested and optimized

The default handlers system makes torchLoom incredibly easy to use while maintaining full flexibility for advanced customization when needed. 