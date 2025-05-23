# Heartbeat Functionality and Pipe-Based Communication

## Overview

This document describes the implementation of heartbeat functionality and the migration from multiprocessing queues to pipes in the torchLoom weavelet system.

## Changes Made

### 1. 🔄 **Migration from Queues to Pipes**

#### **Before (Queues)**
```python
# Inter-process communication
self._config_queue = multiprocessing.Queue()
self._status_queue = multiprocessing.Queue()

# Usage
config_update = self._config_queue.get_nowait()
self._status_queue.put_nowait(status)
```

#### **After (Pipes)**
```python
# Inter-process communication using pipes
self._config_receiver, self._config_sender = multiprocessing.Pipe(duplex=False)
self._status_receiver, self._status_sender = multiprocessing.Pipe(duplex=False)

# Usage
if self._config_receiver.poll(0):
    config_update = self._config_receiver.recv()
self._status_sender.send(status)
```

#### **Benefits of Pipes**
- ✅ **Better Performance**: More efficient for simple data exchange
- ✅ **Cleaner Resource Management**: No need for `join_thread()` calls
- ✅ **Simpler API**: `send()`/`recv()` vs `put()`/`get()`
- ✅ **Better Error Handling**: Cleaner detection of broken connections
- ✅ **No Asyncio Issues**: Eliminates task cancellation warnings

### 2. 💓 **Heartbeat Functionality**

#### **Architecture**

```
┌─────────────────┐    Heartbeat     ┌─────────────────┐
│   Weavelet      │ ────────────────► │     Weaver      │
│   (Process)     │                  │   (Monitor)     │
│                 │                  │                 │
│ • Sends every   │                  │ • Tracks last   │
│   30 seconds    │                  │   heartbeat     │
│ • Includes      │                  │ • Detects dead  │
│   metadata      │                  │   processes     │
└─────────────────┘                  └─────────────────┘
```

#### **Protobuf Message Definition**
```protobuf
message Heartbeat {
  string replica_id = 1;
  string device_uuid = 2;
  int64 timestamp = 3;               // Unix timestamp
  string status = 4;                 // "active", "training", "idle"
  map<string, string> metadata = 5;  // Optional metadata
}
```

#### **Implementation Details**

**1. Weavelet Side (Sender)**
```python
async def _heartbeat_loop(self) -> None:
    """Send heartbeats every 30 seconds."""
    heartbeat_interval = 30.0
    
    while not self._stop_event.is_set():
        await self._send_heartbeat()
        await asyncio.sleep(heartbeat_interval)

async def _send_heartbeat(self) -> None:
    """Send heartbeat message to weaver."""
    envelope = EventEnvelope()
    heartbeat = envelope.heartbeat
    heartbeat.replica_id = self._replica_id
    heartbeat.timestamp = int(time.time())
    heartbeat.status = "active"
    
    await self._nc.publish(
        torchLoomConstants.subjects.HEARTBEAT,
        envelope.SerializeToString()
    )
```

**2. Weaver Side (Receiver)**
```python
class HeartbeatHandler(MessageHandler):
    def __init__(self, heartbeat_timeout: float = 90.0):
        self.heartbeat_timeout = heartbeat_timeout  # 3x heartbeat interval
        self._last_heartbeats: Dict[str, float] = {}
        self._dead_replicas: Set[str] = set()
    
    def check_dead_replicas(self) -> Set[str]:
        """Detect replicas that haven't sent heartbeats."""
        current_time = time.time()
        newly_dead = set()
        
        for replica_id, last_heartbeat in self._last_heartbeats.items():
            if current_time - last_heartbeat > self.heartbeat_timeout:
                self._dead_replicas.add(replica_id)
                newly_dead.add(replica_id)
        
        return newly_dead
```

#### **Process Death Detection**

1. **Heartbeat Interval**: 30 seconds
2. **Timeout Threshold**: 90 seconds (3x interval)
3. **Detection Logic**:
   - Weavelet sends heartbeat every 30s
   - Weaver tracks last heartbeat timestamp
   - If no heartbeat for >90s → mark as dead
   - Publish `REPLICA_FAIL` event
   - Trigger recovery actions

#### **Recovery Actions**
When a process is detected as dead:
1. **Immediate**: Mark replica as failed
2. **Notification**: Publish failure event to other components
3. **Recovery**: (Future) Restart process automatically
4. **Rebalancing**: (Future) Redistribute workload

## File Changes

### Core Files Modified

1. **`torchLoom/weavelet/core.py`**
   - ✅ Replaced queues with pipes
   - ✅ Updated all communication methods
   - ✅ Improved cleanup logic

2. **`torchLoom/weavelet/listener.py`**
   - ✅ Added heartbeat loop
   - ✅ Updated to use pipes
   - ✅ Added heartbeat message publishing

3. **`torchLoom/lightning.py`**
   - ✅ Fixed linter errors
   - ✅ Added better None checks
   - ✅ Improved attribute delegation

4. **`torchLoom/proto/torchLoom.proto`**
   - ✅ Added Heartbeat message type
   - ✅ Updated EventEnvelope

5. **`torchLoom/constants.py`**
   - ✅ Added HEARTBEAT subject

6. **`torchLoom/weaver/status_handlers.py`**
   - ✅ Added HeartbeatHandler class
   - ✅ Process death detection
   - ✅ Recovery event publishing

### Test Files

1. **`test_user_hooks.py`**
   - ✅ Updated to work with pipe-based system
   - ✅ All tests pass successfully

2. **`test_heartbeat.py`** (New)
   - ✅ Comprehensive heartbeat testing
   - ✅ Interactive testing modes
   - ✅ Multiple replica scenarios

## Usage Examples

### Basic Usage
```python
# Create Lightning module
trainer = MyLightningModule()

# Wrap with weavelet (includes heartbeat)
weavelet_trainer = make_weavelet(trainer, replica_id="my_trainer")

# Use normally - heartbeats sent automatically
lightning_trainer = L.Trainer()
lightning_trainer.fit(weavelet_trainer)
```

### Monitoring Heartbeats
```python
# In weaver
heartbeat_handler = HeartbeatHandler(heartbeat_timeout=90.0)

# Check periodically
dead_replicas = heartbeat_handler.check_dead_replicas()
if dead_replicas:
    print(f"Dead replicas detected: {dead_replicas}")
    # Trigger recovery actions
```

## Testing

### Run Basic Tests
```bash
# Test existing functionality
python test_user_hooks.py

# Test heartbeat functionality
python test_heartbeat.py
```

### Interactive Testing
```bash
# Interactive heartbeat test
python test_heartbeat.py --interactive

# Monitor heartbeats only
python test_heartbeat.py --monitor
```

### Test Scenarios
1. **Normal Operation**: Replica runs with regular heartbeats
2. **Process Death**: Replica dies, heartbeats stop
3. **Multiple Replicas**: Different lifespans, staggered failures
4. **Recovery**: Dead replicas detected and marked

## Benefits Achieved

### 🚀 **Performance Improvements**
- Eliminated asyncio task warnings
- Faster inter-process communication
- Cleaner resource cleanup
- No hanging processes

### 🔍 **Reliability Enhancements**
- Automatic process death detection
- Configurable timeout thresholds
- Robust error handling
- Process lifecycle tracking

### 🛠️ **Developer Experience**
- Cleaner APIs
- Better error messages
- Comprehensive testing
- Interactive debugging tools

## Future Enhancements

1. **Automatic Recovery**: Restart dead processes
2. **Health Metrics**: Include system metrics in heartbeats
3. **Adaptive Timeouts**: Dynamic timeout based on system load
4. **Cascade Detection**: Detect and handle cascading failures
5. **Dashboard Integration**: Real-time heartbeat visualization

## Configuration

### Heartbeat Settings
```python
# Weavelet heartbeat interval (seconds)
HEARTBEAT_INTERVAL = 30.0

# Weaver timeout threshold (seconds)
HEARTBEAT_TIMEOUT = 90.0  # 3x interval

# Detection check frequency (seconds)
CHECK_FREQUENCY = 10.0
```

### NATS Subjects
```python
# Heartbeat messages
HEARTBEAT = "torchLoom.heartbeat"

# Failure notifications
REPLICA_FAIL = "torchLoom.replica.fail"
```

## Conclusion

The implementation successfully achieves:
✅ **Reliable process monitoring** through heartbeats
✅ **Efficient communication** using pipes instead of queues
✅ **Clean resource management** with proper cleanup
✅ **Backward compatibility** with existing code
✅ **Comprehensive testing** for all scenarios

This provides a solid foundation for building a robust distributed training system with automatic failure detection and recovery capabilities. 