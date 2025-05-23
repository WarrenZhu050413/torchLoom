# torchLoom Weavelet Demo Guide

This guide demonstrates the **Enhanced Handler System** for dynamic training configuration management. The weavelet system allows you to modify training parameters (optimizer type, learning rate, etc.) in real-time without stopping training.

## 🎯 What is the Weavelet System?

The **Weavelet** is a plug-and-play component that enables **dynamic reconfiguration** of PyTorch training processes. It provides:

- **🔧 Real-time Configuration Changes**: Modify optimizer, learning rate, batch size during training
- **🛡️ Type Safety**: Automatic validation and conversion of configuration values
- **⚡ Zero Overhead**: Process isolation ensures no impact on training performance
- **🧩 Framework Agnostic**: Works with PyTorch Lightning, native PyTorch, or any framework
- **🎨 Declarative API**: Clean decorator-based handler registration

## 🏗️ Architecture Overview

```
┌─────────────┐    NATS/JetStream    ┌─────────────────┐    Queue    ┌──────────────┐
│   Weaver    │ ─────────────────► │ WeaveletListener │ ──────────► │  Weavelet    │
│ (Control)   │   Config Updates    │  (Subprocess)   │    IPC      │ (Training)   │
└─────────────┘                     └─────────────────┘             └──────────────┘
                                                                              │
                                                                              ▼
                                                                    ┌─────────────────┐
                                                                    │ Handler Dispatch│
                                                                    │ Type Validation │
                                                                    │ Function Calls  │
                                                                    └─────────────────┘
```

## 🚀 Available Demos

### 1. Basic Handler System Demo (`demo_weavelet.py`)

**What it demonstrates:**
- Core handler registration and dispatch
- Type validation and conversion
- Multiple simultaneous configuration changes

**Key Features Shown:**
```python
# Register handlers with decorators
@weavelet.handler("optimizer_type")
def handle_optimizer_change(new_type: str):
    print(f"Optimizer changed to: {new_type}")

@weavelet.handler("learning_rate")
def handle_lr_change(new_lr: float):
    print(f"Learning rate changed to: {new_lr}")

# Automatic type conversion
weavelet._dispatch_handlers({"learning_rate": "0.001"})  # "0.001" → 0.001
```

### 2. Enhanced Lightning Integration Demo (`demo_weavelet.py`)

**What it demonstrates:**
- Automatic weavelet lifecycle management
- Enhanced Lightning base classes
- Real training loop integration
- Automatic status publishing

**Key Features Shown:**
```python
class DemoTrainer(EnhancedWeaveletLightningModule):
    def __init__(self, vocab_size: int):
        super().__init__(replica_id="demo_trainer")  # Automatic setup
        
    @weavelet_handler("optimizer_type")  # Auto-registered
    def update_optimizer(self, new_type: str):
        # Called automatically when config changes
        self.optimizer_type = new_type
```

### 3. Training Integration Comparison (`train.py`)

**What it demonstrates:**
- Original vs Enhanced integration patterns
- Backward compatibility
- Production-ready examples

**Integration Patterns:**

#### Original Enhanced Integration:
```python
class LightningTransformer(L.LightningModule):
    def __init__(self, vocab_size, replica_id):
        self.weavelet = Weavelet(replica_id=replica_id)
        
        # Manual handler registration
        @self.weavelet.handler("optimizer_type")
        def update_optimizer(new_type: str):
            self.update_optimizer(new_type)
        
        self.weavelet.start()
```

#### Automatic Enhanced Integration:
```python
class EnhancedLightningTransformer(EnhancedWeaveletLightningModule):
    def __init__(self, vocab_size: int, replica_id: str):
        super().__init__(replica_id=replica_id)  # Everything automatic
        
    @weavelet_handler("optimizer_type")  # Auto-registered
    def update_optimizer(self, new_type: str):
        # Handler called automatically
        pass
```

## 🏃‍♂️ How to Run the Demos

### Prerequisites

1. **Environment Setup:**
   ```bash
   conda activate nats-torch27
   cd /path/to/torchLoom
   ```

2. **NATS Server** (optional - demos work without actual NATS):
   ```bash
   # If you want to test with real NATS communication
   nats-server -js
   ```

### Running Individual Demos

#### 1. Enhanced Weavelet Demo
```bash
python demo_weavelet.py
```

**Expected Output:**
```
🔥 Basic Handler System Demo
==============================
✅ Handlers registered
📋 Initial state: {'optimizer_type': 'SGD', 'learning_rate': 0.01, 'enabled': True}

1. Applying config: {'optimizer_type': 'Adam'}
  🔧 Optimizer changed to: Adam
   📋 New state: {'optimizer_type': 'Adam', 'learning_rate': 0.01, 'enabled': True}

🌟 torchLoom Enhanced Weavelet Demo
==================================================
✅ Enhanced trainer created with replica_id: demo_enhanced_trainer
🔧 Initial optimizer: SGD
📈 Initial learning rate: 0.01

🧪 Testing Handler System:
1. Testing optimizer type change...
🔧 Handler: Updating optimizer SGD → Adam

🚀 Testing Lightning Training with Automatic Integration:
   Starting training (config changes and status publishing are automatic)...
   ✅ Training completed successfully!
```

#### 2. Training Integration Demo
```bash
python train.py
```

**Expected Output:**
```
🔥 Demo: Original Integration with Enhanced Handlers
============================================================
Registered handler for 'optimizer_type' with type <class 'str'>
Registered handler for 'learning_rate' with type <class 'float'>
Weavelet process started with PID: 12345

🚀 Demo: Enhanced Automatic Integration
============================================================
Auto-registered weavelet handler: update_optimizer for 'optimizer_type'
Auto-registered weavelet handler: update_learning_rate for 'learning_rate'

✅ All demos completed!
Key improvements demonstrated:
  ✓ Decorator-based handler registration
  ✓ Automatic handler dispatch
  ✓ Type validation for config parameters
  ✓ Enhanced Lightning integration classes
```

#### 3. Run Tests
```bash
python -m pytest tests/test_weavelet_callback.py -v
```

## 🔧 Key Features Demonstrated

### 1. **Decorator-Based Handler Registration**
```python
@weavelet.handler("optimizer_type")
def update_optimizer(new_type: str):
    # Type automatically inferred from annotation
    self.switch_optimizer(new_type)
```

### 2. **Automatic Type Validation**
```python
# String to float conversion
"0.001" → 0.001

# String to boolean conversion  
"true" → True
"false" → False
"1" → True

# Error handling
"invalid_float" → TypeError with clear message
```

### 3. **Enhanced Lightning Integration**
```python
class MyTrainer(EnhancedWeaveletLightningModule):
    def __init__(self):
        super().__init__(replica_id="trainer_1")
        # Weavelet automatically started and managed
        
    def training_step(self, batch, batch_idx):
        # Automatic config checking and status publishing
        return super().training_step(batch, batch_idx)
```

### 4. **Process Isolation**
- **Weavelet**: Main interface in training process
- **WeaveletListener**: NATS communication in separate subprocess
- **Zero Impact**: Training performance unaffected by network operations

## 🎨 How to Adapt for Your Use Case

### 1. **Basic Integration** (5 lines of code)
```python
from torchLoom.weavelet import Weavelet

class MyTrainer:
    def __init__(self):
        self.weavelet = Weavelet(replica_id="my_trainer")
        
        @self.weavelet.handler("learning_rate")
        def update_lr(new_lr: float):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
                
        self.weavelet.start()
```

### 2. **Lightning Integration** (Automatic)
```python
from torchLoom.lightning_integration import EnhancedWeaveletLightningModule, weavelet_handler

class MyLightningModule(EnhancedWeaveletLightningModule):
    def __init__(self):
        super().__init__(replica_id="lightning_trainer")
        
    @weavelet_handler("batch_size")
    def update_batch_size(self, new_size: int):
        # Called automatically when batch_size changes
        self.batch_size = new_size
```

### 3. **Custom Parameters**
```python
@weavelet.handler("custom_param")
def handle_custom(value: str):
    # Handle any configuration parameter
    self.custom_setting = value

@weavelet.handler("threshold") 
def handle_threshold(threshold: float):
    # Numeric parameters with validation
    if 0.0 <= threshold <= 1.0:
        self.threshold = threshold
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the correct conda environment
   ```bash
   conda activate nats-torch27
   ```

2. **Multiprocessing Issues**: The demos handle this automatically, but if you see warnings:
   ```python
   import multiprocessing as mp
   mp.set_start_method('spawn', force=True)
   ```

3. **NATS Connection**: Demos work without NATS server, but for real usage:
   ```bash
   # Start NATS server with JetStream
   nats-server -js
   ```

### Expected Warnings

These are normal and don't affect functionality:
- `"EventEnvelope" is unknown import symbol` - Protobuf type checking
- `Force terminating weavelet process` - Normal cleanup process
- Lightning deprecation warnings - Framework version compatibility

## 🎯 What You Should See

After running the demos, you should observe:

1. **✅ Handler Registration**: Clear confirmation of registered handlers
2. **✅ Type Conversion**: Automatic string-to-number conversions
3. **✅ Error Handling**: Graceful handling of invalid values
4. **✅ Process Management**: Clean startup and shutdown of subprocesses
5. **✅ Training Integration**: Seamless operation during actual training steps

## 🚀 Next Steps

1. **Try the demos** to understand the system
2. **Adapt the examples** for your training code
3. **Add your own handlers** for custom parameters
4. **Integrate with your weaver** for production use

The weavelet system transforms static training into **dynamic, controllable processes** with minimal code changes and maximum safety! 🎉

## 📚 Additional Resources

- **Architecture**: See `AGENTS.md` for detailed design philosophy
- **Implementation**: See `plan.md` for development roadmap  
- **Tests**: See `tests/test_weavelet_callback.py` for comprehensive examples
- **Integration**: See `torchLoom/lightning_integration.py` for Lightning helpers 