#!/bin/bash

# Fault Tolerance Demo Script for torchLoom
# This script demonstrates worker preemption based on external monitoring

set -e

echo "=== torchLoom Fault Tolerance Demo ==="
echo "This demo showcases worker preemption based on external monitoring"
echo ""

# Configuration
DEMO_DURATION=180  # 3 minutes demo
NATS_CONFIG="../../nats/nats.conf"
NATS_SERVER="../../nats/nats-server"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up demo processes..."
    
    # Kill background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Kill specific processes by name
    pkill -f "nats-server" 2>/dev/null || true
    pkill -f "torchLoom.weaver" 2>/dev/null || true
    pkill -f "train_fault_tolerant" 2>/dev/null || true
    
    # Wait a moment for processes to terminate
    sleep 2
    
    echo "Cleanup completed"
}

# Set up cleanup trap
trap cleanup EXIT INT TERM

# Check dependencies
check_dependencies() {
    echo "Checking dependencies..."
    
    if [ ! -f "$NATS_SERVER" ]; then
        echo "Error: NATS server not found at $NATS_SERVER"
        echo "Please ensure NATS is installed and the path is correct"
        exit 1
    fi
    
    if [ ! -f "$NATS_CONFIG" ]; then
        echo "Error: NATS config not found at $NATS_CONFIG"
        exit 1
    fi
    
    # Check Python dependencies
    python -c "import torch, torchLoom" 2>/dev/null || {
        echo "Error: Required Python packages not found"
        echo "Please ensure torch and torchLoom are installed"
        exit 1
    }
    
    echo "Dependencies check passed"
}

# Start NATS server
start_nats() {
    echo "Starting NATS server..."
    
    # Clear any existing NATS data
    rm -rf /tmp/nats-jetstream 2>/dev/null || true
    
    $NATS_SERVER -c $NATS_CONFIG > nats.log 2>&1 &
    NATS_PID=$!
    
    # Wait for NATS to start
    sleep 3
    
    if ! kill -0 $NATS_PID 2>/dev/null; then
        echo "Error: Failed to start NATS server"
        cat nats.log
        exit 1
    fi
    
    echo "NATS server started (PID: $NATS_PID)"
}

# Start Weaver service
start_weaver() {
    echo "Starting Weaver service..."
    
    cd ../../
    python -m torchLoom.weaver.weaver > examples/fault_tolerance/weaver.log 2>&1 &
    WEAVER_PID=$!
    cd examples/fault_tolerance/
    
    # Wait for Weaver to start
    sleep 5
    
    if ! kill -0 $WEAVER_PID 2>/dev/null; then
        echo "Error: Failed to start Weaver service"
        cat weaver.log
        exit 1
    fi
    
    echo "Weaver service started (PID: $WEAVER_PID)"
}

# Start fault tolerant training
start_training() {
    echo "Starting fault tolerant training..."
    
    python train_fault_tolerant.py > training.log 2>&1 &
    TRAINING_PID=$!
    
    # Wait for training to initialize
    sleep 10
    
    if ! kill -0 $TRAINING_PID 2>/dev/null; then
        echo "Error: Failed to start training"
        cat training.log
        exit 1
    fi
    
    echo "Fault tolerant training started (PID: $TRAINING_PID)"
}

# Inject fault scenarios
inject_faults() {
    echo ""
    echo "=== Fault Injection Phase ==="
    echo "Injecting various fault scenarios to test fault tolerance..."
    
    # Wait for training to stabilize
    sleep 20
    
    echo "Injecting memory pressure fault..."
    python -c "
from fault_injector import FaultInjector, FaultType
import time

injector = FaultInjector('demo-injector', failure_probability=0.2)
injector.inject_fault(FaultType.MEMORY_PRESSURE)
time.sleep(5)
injector.inject_fault(FaultType.SLOW_TRAINING)
print('Memory pressure and slow training faults injected')
"
    
    sleep 15
    
    echo "Injecting network partition scenario..."
    python -c "
from fault_injector import FaultInjector
import time

injector = FaultInjector('demo-injector')
injector.create_fault_scenario('network_issues')
print('Network issues scenario injected')
"
    
    sleep 20
    
    echo "Injecting cascade failure scenario..."
    python -c "
from fault_injector import FaultInjector
import time

injector = FaultInjector('demo-injector')
injector.create_fault_scenario('cascade_failure')
print('Cascade failure scenario injected')
"
    
    sleep 15
    
    echo "Enabling chaos mode for intensive testing..."
    python -c "
from fault_injector import AdvancedFaultInjector
import time

injector = AdvancedFaultInjector('demo-injector')
injector.enable_chaos_mode('medium')
injector.start_burst_pattern(duration=30.0, intensity=0.2)
print('Chaos mode enabled with burst pattern')
"
    
    sleep 30
}

# Monitor training progress
monitor_training() {
    echo ""
    echo "=== Monitoring Training Progress ==="
    echo "Monitoring fault tolerance metrics..."
    
    # Monitor for specified duration
    local end_time=$((SECONDS + DEMO_DURATION))
    
    while [ $SECONDS -lt $end_time ]; do
        if kill -0 $TRAINING_PID 2>/dev/null; then
            echo "Training process alive - checking metrics..."
            
            # Extract metrics from training log
            if [ -f training.log ]; then
                tail -n 5 training.log | grep -E "(Epoch|Preemption|Recovery|Checkpoint)" || true
            fi
        else
            echo "Training process terminated"
            break
        fi
        
        sleep 20
    done
}

# Generate final report
generate_report() {
    echo ""
    echo "=== Fault Tolerance Demo Report ==="
    echo "Generating final report..."
    
    # Count events from logs
    local preemptions=0
    local recoveries=0
    local checkpoints=0
    
    if [ -f training.log ]; then
        preemptions=$(grep -c "preemption" training.log 2>/dev/null || echo "0")
        recoveries=$(grep -c "recovery\|resumed" training.log 2>/dev/null || echo "0")
        checkpoints=$(grep -c "checkpoint" training.log 2>/dev/null || echo "0")
    fi
    
    echo "Demo Results:"
    echo "  Duration: $DEMO_DURATION seconds"
    echo "  Preemptions Triggered: $preemptions"
    echo "  Successful Recoveries: $recoveries"
    echo "  Checkpoints Created: $checkpoints"
    echo ""
    
    if [ $preemptions -gt 0 ] && [ $recoveries -gt 0 ]; then
        echo "✅ Fault tolerance demonstration SUCCESSFUL"
        echo "   Workers were successfully preempted and recovered"
    else
        echo "⚠️  Fault tolerance demonstration may need adjustment"
        echo "   Check logs for details"
    fi
    
    echo ""
    echo "Log files generated:"
    echo "  - training.log: Training process output"
    echo "  - weaver.log: Weaver service output"
    echo "  - nats.log: NATS server output"
    echo ""
    echo "Example completed. Check the logs for detailed information."
}

# Main execution
main() {
    echo "Starting fault tolerance demonstration..."
    echo "This will run for approximately $DEMO_DURATION seconds"
    echo ""
    
    check_dependencies
    start_nats
    start_weaver
    start_training
    inject_faults
    monitor_training
    generate_report
}

# Run the demo
main 