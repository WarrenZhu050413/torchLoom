#!/bin/bash

# Topology Optimization Demo Script for torchLoom
# This script demonstrates allreduce topology optimization via real-time network profiling

set -e

echo "=== torchLoom Topology Optimization Demo ==="
echo "This demo showcases allreduce topology optimization via real-time network profiling"
echo ""

# Configuration
DEMO_DURATION=240  # 4 minutes demo
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
    pkill -f "train_topology_aware" 2>/dev/null || true
    
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
    python -m torchLoom.weaver.weaver > examples/topology_optimization/weaver.log 2>&1 &
    WEAVER_PID=$!
    cd examples/topology_optimization/
    
    # Wait for Weaver to start
    sleep 5
    
    if ! kill -0 $WEAVER_PID 2>/dev/null; then
        echo "Error: Failed to start Weaver service"
        cat weaver.log
        exit 1
    fi
    
    echo "Weaver service started (PID: $WEAVER_PID)"
}

# Start topology-aware training
start_training() {
    echo "Starting topology-aware training..."
    
    python train_topology_aware.py > training.log 2>&1 &
    TRAINING_PID=$!
    
    # Wait for training to initialize
    sleep 15
    
    if ! kill -0 $TRAINING_PID 2>/dev/null; then
        echo "Error: Failed to start training"
        cat training.log
        exit 1
    fi
    
    echo "Topology-aware training started (PID: $TRAINING_PID)"
}

# Simulate network conditions
simulate_network_conditions() {
    echo ""
    echo "=== Network Simulation Phase ==="
    echo "Simulating various network conditions to trigger topology optimization..."
    
    # Wait for training to stabilize
    sleep 20
    
    echo "Starting with stable network conditions..."
    python -c "
from bandwidth_simulator import BandwidthSimulator
import time

simulator = BandwidthSimulator('demo-simulator', num_nodes=4)
simulator.start('stable')
print('Stable network scenario started')
time.sleep(10)
simulator.stop()
"
    
    sleep 15
    
    echo "Switching to congested network scenario..."
    python -c "
from bandwidth_simulator import BandwidthSimulator
import time

simulator = BandwidthSimulator('demo-simulator', num_nodes=4)
simulator.start('congested')
print('Congested network scenario started')

# Inject congestion spike
simulator.inject_network_event('congestion_spike', {
    'severity': 0.6,
    'duration': 20.0
})
print('Congestion spike injected')

time.sleep(25)
simulator.stop()
"
    
    sleep 20
    
    echo "Testing hierarchical network topology..."
    python -c "
from bandwidth_simulator import BandwidthSimulator
import time

simulator = BandwidthSimulator('demo-simulator', num_nodes=4)
simulator.start('hierarchical')
print('Hierarchical network scenario started')

# Inject network partition
simulator.inject_network_event('partition', {
    'group1': [0, 1],
    'group2': [2, 3],
    'duration': 15.0
})
print('Network partition injected')

time.sleep(20)
simulator.stop()
"
    
    sleep 25
    
    echo "Testing unstable network with high variation..."
    python -c "
from bandwidth_simulator import BandwidthSimulator
import time

simulator = BandwidthSimulator('demo-simulator', num_nodes=4)
simulator.start('unstable')
print('Unstable network scenario started')

# Multiple network events
simulator.inject_network_event('link_failure', {'src': 0, 'dst': 1, 'duration': 10})
time.sleep(5)
simulator.inject_network_event('bandwidth_boost', {'boost_factor': 2.5, 'duration': 15})
print('Link failure and bandwidth boost events injected')

time.sleep(20)
simulator.stop()
"
    
    sleep 25
    
    echo "Testing asymmetric bandwidth patterns..."
    python -c "
from bandwidth_simulator import BandwidthSimulator
import time

simulator = BandwidthSimulator('demo-simulator', num_nodes=4)
simulator.start('asymmetric')
print('Asymmetric network scenario started')
time.sleep(15)
simulator.stop()
"
}

# Monitor topology optimization
monitor_topology_optimization() {
    echo ""
    echo "=== Topology Optimization Monitoring ==="
    echo "Monitoring topology switching and performance metrics..."
    
    # Monitor for remaining duration
    local remaining_time=$((DEMO_DURATION - SECONDS))
    local end_time=$((SECONDS + remaining_time))
    
    while [ $SECONDS -lt $end_time ]; do
        if kill -0 $TRAINING_PID 2>/dev/null; then
            echo "Training process alive - checking topology metrics..."
            
            # Extract topology metrics from training log
            if [ -f training.log ]; then
                tail -n 10 training.log | grep -E "(Topology|Bandwidth|Optimization|Efficiency)" | head -3 || true
            fi
        else
            echo "Training process terminated"
            break
        fi
        
        sleep 25
    done
}

# Generate topology optimization report
generate_report() {
    echo ""
    echo "=== Topology Optimization Demo Report ==="
    echo "Generating final report..."
    
    # Count events from logs
    local topology_switches=0
    local optimizations=0
    local network_events=0
    
    if [ -f training.log ]; then
        topology_switches=$(grep -c "topology.*switch\|Topology switched" training.log 2>/dev/null || echo "0")
        optimizations=$(grep -c "optimization\|Optimization" training.log 2>/dev/null || echo "0")
        network_events=$(grep -c "network.*event\|Network.*scenario" training.log 2>/dev/null || echo "0")
    fi
    
    echo "Demo Results:"
    echo "  Duration: $DEMO_DURATION seconds"
    echo "  Topology Switches: $topology_switches"
    echo "  Optimization Events: $optimizations"
    echo "  Network Scenario Changes: $network_events"
    echo ""
    
    # Extract final performance metrics
    if [ -f training.log ]; then
        echo "Final Performance Metrics:"
        tail -n 20 training.log | grep -E "(efficiency|bandwidth|communication)" | tail -5 || echo "  No performance metrics found"
    fi
    
    echo ""
    
    if [ $topology_switches -gt 0 ] && [ $optimizations -gt 0 ]; then
        echo "✅ Topology optimization demonstration SUCCESSFUL"
        echo "   Dynamic topology switching was triggered based on network conditions"
    else
        echo "⚠️  Topology optimization demonstration may need adjustment"
        echo "   Check logs for details on optimization decisions"
    fi
    
    echo ""
    echo "Log files generated:"
    echo "  - training.log: Training process and topology optimization output"
    echo "  - weaver.log: Weaver service output"
    echo "  - nats.log: NATS server output"
    echo ""
    
    # Performance summary
    echo "Topology Performance Summary:"
    echo "  Expected topologies for different conditions:"
    echo "    - Stable/High BW: Butterfly (highest bandwidth utilization)"
    echo "    - Congested: Ring or Tree (fault tolerant)"
    echo "    - Hierarchical: Hierarchical (matches infrastructure)"
    echo "    - Unstable: All-to-All (maximum redundancy)"
    echo "    - Asymmetric: Tree (adapts to bandwidth asymmetry)"
    echo ""
    echo "Example completed. Check the logs for detailed topology optimization information."
}

# Test topology optimizer independently
test_topology_optimizer() {
    echo "Testing topology optimizer independently..."
    
    python -c "
from topology_optimizer import TopologyOptimizer, TopologyType
from network_profiler import NetworkProfiler
import time

# Create mock network conditions
class MockProfiler:
    def get_current_conditions(self):
        return {
            'bandwidth_matrix': {
                0: {1: 8000, 2: 2000, 3: 1000},
                1: {0: 8000, 2: 8000, 3: 2000},
                2: {0: 2000, 1: 8000, 3: 8000},
                3: {0: 1000, 1: 2000, 2: 8000}
            },
            'latency_matrix': {
                0: {1: 1.0, 2: 5.0, 3: 10.0},
                1: {0: 1.0, 2: 1.0, 3: 5.0},
                2: {0: 5.0, 1: 1.0, 3: 1.0},
                3: {0: 10.0, 1: 5.0, 2: 1.0}
            },
            'network_health': {'score': 0.8}
        }

profiler = MockProfiler()
optimizer = TopologyOptimizer('test-optimizer', 4, profiler)

# Test topology evaluation
evaluations = optimizer.force_topology_evaluation()
print('Topology Evaluation Results:')
for topology, perf in evaluations.items():
    print(f'  {topology.value}: score={perf.overall_score:.3f}, time={perf.estimated_time_ms:.1f}ms')

optimal = optimizer.find_optimal_topology()
print(f'Optimal topology for current conditions: {optimal.value}')
" || echo "Could not test topology optimizer independently"
}

# Main execution
main() {
    echo "Starting topology optimization demonstration..."
    echo "This will run for approximately $DEMO_DURATION seconds"
    echo ""
    
    check_dependencies
    test_topology_optimizer
    start_nats
    start_weaver
    start_training
    simulate_network_conditions
    monitor_topology_optimization
    generate_report
}

# Run the demo
main 