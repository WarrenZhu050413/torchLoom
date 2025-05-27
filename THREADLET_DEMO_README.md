# Threadlet Demo Scripts

This directory contains two scripts for demonstrating torchLoom threadlet functionality:

## 1. Modified WebSocket CLI (`websocket_cli.py`)

The WebSocket CLI has been updated to continue running after sending demo commands instead of terminating.

### Features:
- Connects to the weaver WebSocket server at `ws://localhost:8080/ws`
- Sends a sequence of demo commands (config updates, pause/resume training, device commands)
- **Continues running indefinitely** after demo commands complete
- Displays real-time status updates from the weaver
- Sends periodic ping messages to keep connection alive

### Usage:
```bash
# Activate the conda environment first
conda init bash && source ~/.bashrc && conda activate nats-torch27

# Run the WebSocket CLI
python websocket_cli.py
```

## 2. Standalone Threadlet Runner (`spawn_threadlet.py`)

A new script that directly spawns a threadlet process to listen to weaver commands and simulate training.

### Features:
- Creates a threadlet instance with configurable replica ID
- Registers handlers for configuration updates (learning_rate, batch_size, pause/resume)
- Simulates training progress with realistic metrics (loss, accuracy)
- Publishes heartbeats, status updates, and training metrics to the weaver
- Responds to configuration changes from the weaver in real-time

### Usage:
```bash
# Activate the conda environment first
conda init bash && source ~/.bashrc && conda activate nats-torch27

# Run with default settings
python spawn_threadlet.py

# Run with custom replica ID
python spawn_threadlet.py --replica-id my-custom-replica

# Run with custom torchLoom address
python spawn_threadlet.py --torchLoom-addr nats://localhost:4222

# Run with debug logging
python spawn_threadlet.py --log-level DEBUG
```

### Configuration Handlers:
The threadlet registers handlers for these configuration parameters:
- `learning_rate`: Updates the simulated learning rate
- `batch_size`: Updates the simulated batch size  
- `pause_training`: Pauses the training simulation
- `resume_training`: Resumes the training simulation

## Demo Workflow

To see the full system in action:

1. **Start the infrastructure:**
   ```bash
   # Terminal 1: Start NATS server
   ./nats/nats-server -c nats/nats.conf -D
   
   # Terminal 2: Start the weaver
   conda init bash && source ~/.bashrc && conda activate nats-torch27
   python -m torchLoom.weaver.core
   
   # Terminal 3: Start the UI (optional)
   cd torchLoom-ui
   npm run dev -y
   ```

2. **Start the threadlet:**
   ```bash
   # Terminal 4: Start threadlet listener
   conda init bash && source ~/.bashrc && conda activate nats-torch27
   python spawn_threadlet.py --replica-id demo-replica-1
   ```

3. **Start the WebSocket CLI:**
   ```bash
   # Terminal 5: Start WebSocket CLI
   conda init bash && source ~/.bashrc && conda activate nats-torch27
   python websocket_cli.py
   ```

4. **Observe the interaction:**
   - The WebSocket CLI will send demo commands targeting `demo-replica-1`
   - The threadlet will receive and respond to these commands
   - You'll see real-time updates in both terminals
   - The system continues running for ongoing monitoring

## Expected Output

### WebSocket CLI:
- Connection status messages
- Demo command sequence execution
- Real-time status updates from threadlets
- Training metrics display

### Threadlet Runner:
- Threadlet startup and configuration
- Configuration change notifications
- Training simulation progress
- Heartbeat and status publishing

Both scripts run indefinitely until stopped with Ctrl+C. 