"""
Fault Injector for torchLoom Fault Tolerance Testing

This module provides utilities to inject various types of faults
into the training process for testing fault tolerance mechanisms.
"""

import logging
import random
import time
import threading
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class FaultType(Enum):
    """Types of faults that can be injected."""
    PROCESS_CRASH = "process_crash"
    NETWORK_PARTITION = "network_partition" 
    MEMORY_PRESSURE = "memory_pressure"
    DISK_FULL = "disk_full"
    GPU_ERROR = "gpu_error"
    SLOW_TRAINING = "slow_training"
    CHECKPOINT_CORRUPTION = "checkpoint_corruption"
    COMMUNICATION_TIMEOUT = "communication_timeout"


class FaultInjector:
    """
    Fault injection utility for testing fault tolerance mechanisms.
    
    This class can inject various types of faults into the training process
    to test how well the fault tolerance system handles different scenarios.
    """
    
    def __init__(self, process_id: str, failure_probability: float = 0.01):
        self.process_id = process_id
        self.failure_probability = failure_probability
        
        # Fault injection state
        self.active_faults = set()
        self.fault_history = []
        self.injection_count = 0
        
        # Fault configuration
        self.fault_config = {
            FaultType.PROCESS_CRASH: {
                'probability': 0.001,  # 0.1% per check
                'duration': 0,  # Immediate
                'severity': 'critical'
            },
            FaultType.NETWORK_PARTITION: {
                'probability': 0.005,  # 0.5% per check
                'duration': 30,  # 30 seconds
                'severity': 'high'
            },
            FaultType.MEMORY_PRESSURE: {
                'probability': 0.01,   # 1% per check
                'duration': 60,  # 60 seconds
                'severity': 'medium'
            },
            FaultType.GPU_ERROR: {
                'probability': 0.002,  # 0.2% per check
                'duration': 15,  # 15 seconds
                'severity': 'high'
            },
            FaultType.SLOW_TRAINING: {
                'probability': 0.02,   # 2% per check
                'duration': 120, # 2 minutes
                'severity': 'low'
            }
        }
        
        # Fault injection callbacks
        self.fault_handlers = {}
        
        logger.info(f"Fault injector initialized for process: {process_id}")
    
    def register_fault_handler(self, fault_type: FaultType, handler):
        """Register a handler for a specific fault type."""
        if fault_type not in self.fault_handlers:
            self.fault_handlers[fault_type] = []
        self.fault_handlers[fault_type].append(handler)
        logger.info(f"Registered handler for fault type: {fault_type.value}")
    
    def should_inject_fault(self) -> bool:
        """Determine if a fault should be injected based on probability."""
        return random.random() < self.failure_probability
    
    def inject_random_fault(self) -> Optional[FaultType]:
        """Inject a random fault based on configured probabilities."""
        for fault_type, config in self.fault_config.items():
            if random.random() < config['probability']:
                return self.inject_fault(fault_type)
        return None
    
    def inject_fault(self, fault_type: FaultType, custom_config: Optional[Dict] = None) -> FaultType:
        """Inject a specific type of fault."""
        if fault_type in self.active_faults:
            logger.debug(f"Fault {fault_type.value} already active, skipping")
            return fault_type
        
        config = self.fault_config.get(fault_type, {})
        if custom_config:
            config.update(custom_config)
        
        self.active_faults.add(fault_type)
        self.injection_count += 1
        
        fault_event = {
            'fault_type': fault_type,
            'process_id': self.process_id,
            'timestamp': time.time(),
            'injection_id': self.injection_count,
            'config': config,
            'severity': config.get('severity', 'medium')
        }
        
        self.fault_history.append(fault_event)
        
        logger.warning(f"Injecting fault: {fault_type.value} (severity: {config.get('severity')})")
        
        # Execute fault-specific logic
        self._execute_fault(fault_type, config)
        
        # Call registered handlers
        if fault_type in self.fault_handlers:
            for handler in self.fault_handlers[fault_type]:
                try:
                    handler(fault_event)
                except Exception as e:
                    logger.error(f"Error in fault handler for {fault_type.value}: {e}")
        
        # Schedule fault recovery if it has duration
        duration = config.get('duration', 0)
        if duration > 0:
            recovery_thread = threading.Thread(
                target=self._schedule_recovery,
                args=(fault_type, duration),
                daemon=True
            )
            recovery_thread.start()
        
        return fault_type
    
    def _execute_fault(self, fault_type: FaultType, config: Dict):
        """Execute fault-specific logic."""
        if fault_type == FaultType.PROCESS_CRASH:
            self._simulate_process_crash()
        elif fault_type == FaultType.NETWORK_PARTITION:
            self._simulate_network_partition(config)
        elif fault_type == FaultType.MEMORY_PRESSURE:
            self._simulate_memory_pressure(config)
        elif fault_type == FaultType.GPU_ERROR:
            self._simulate_gpu_error(config)
        elif fault_type == FaultType.SLOW_TRAINING:
            self._simulate_slow_training(config)
        elif fault_type == FaultType.CHECKPOINT_CORRUPTION:
            self._simulate_checkpoint_corruption(config)
        elif fault_type == FaultType.COMMUNICATION_TIMEOUT:
            self._simulate_communication_timeout(config)
    
    def _simulate_process_crash(self):
        """Simulate process crash - in real scenario this would terminate the process."""
        logger.critical("FAULT INJECTION: Simulating process crash")
        # In a real scenario, this might call sys.exit() or raise SystemExit
        # For testing, we just log it
        self._notify_fault_event("process_crash", "Process crash simulated")
    
    def _simulate_network_partition(self, config: Dict):
        """Simulate network partition."""
        logger.warning("FAULT INJECTION: Simulating network partition")
        # In real scenario, this might block network calls or modify routing
        self._notify_fault_event("network_partition", f"Network partition for {config.get('duration', 30)}s")
    
    def _simulate_memory_pressure(self, config: Dict):
        """Simulate memory pressure."""
        logger.warning("FAULT INJECTION: Simulating memory pressure")
        # In real scenario, this might allocate large amounts of memory
        self._notify_fault_event("memory_pressure", f"Memory pressure for {config.get('duration', 60)}s")
    
    def _simulate_gpu_error(self, config: Dict):
        """Simulate GPU error."""
        logger.warning("FAULT INJECTION: Simulating GPU error")
        # In real scenario, this might cause CUDA operations to fail
        self._notify_fault_event("gpu_error", f"GPU error for {config.get('duration', 15)}s")
    
    def _simulate_slow_training(self, config: Dict):
        """Simulate slow training by introducing delays."""
        logger.warning("FAULT INJECTION: Simulating slow training")
        # Add artificial delay to training steps
        delay = config.get('delay', 0.5)  # Default 500ms delay
        time.sleep(delay)
        self._notify_fault_event("slow_training", f"Training slowdown for {config.get('duration', 120)}s")
    
    def _simulate_checkpoint_corruption(self, config: Dict):
        """Simulate checkpoint corruption."""
        logger.warning("FAULT INJECTION: Simulating checkpoint corruption")
        # In real scenario, this might corrupt checkpoint files
        self._notify_fault_event("checkpoint_corruption", "Checkpoint corruption detected")
    
    def _simulate_communication_timeout(self, config: Dict):
        """Simulate communication timeout."""
        logger.warning("FAULT INJECTION: Simulating communication timeout")
        # In real scenario, this might cause collective operations to timeout
        self._notify_fault_event("communication_timeout", f"Communication timeout for {config.get('duration', 30)}s")
    
    def _notify_fault_event(self, fault_type: str, message: str):
        """Send fault notification."""
        # This would typically send to monitoring systems or threadlet
        logger.info(f"FAULT NOTIFICATION: {fault_type} - {message}")
    
    def _schedule_recovery(self, fault_type: FaultType, duration: float):
        """Schedule automatic recovery from fault after duration."""
        time.sleep(duration)
        self.recover_from_fault(fault_type)
    
    def recover_from_fault(self, fault_type: FaultType):
        """Recover from a specific fault."""
        if fault_type not in self.active_faults:
            logger.debug(f"Fault {fault_type.value} not active, cannot recover")
            return
        
        self.active_faults.remove(fault_type)
        
        recovery_event = {
            'fault_type': fault_type,
            'process_id': self.process_id,
            'timestamp': time.time(),
            'event_type': 'recovery'
        }
        
        logger.info(f"Recovering from fault: {fault_type.value}")
        
        # Execute recovery-specific logic
        self._execute_recovery(fault_type)
        
        return recovery_event
    
    def _execute_recovery(self, fault_type: FaultType):
        """Execute recovery logic for specific fault types."""
        if fault_type == FaultType.NETWORK_PARTITION:
            logger.info("RECOVERY: Network partition resolved")
        elif fault_type == FaultType.MEMORY_PRESSURE:
            logger.info("RECOVERY: Memory pressure relieved")
        elif fault_type == FaultType.GPU_ERROR:
            logger.info("RECOVERY: GPU error resolved")
        elif fault_type == FaultType.SLOW_TRAINING:
            logger.info("RECOVERY: Training speed normalized")
        elif fault_type == FaultType.COMMUNICATION_TIMEOUT:
            logger.info("RECOVERY: Communication restored")
    
    def clear_all_faults(self):
        """Clear all active faults."""
        active_faults = self.active_faults.copy()
        for fault_type in active_faults:
            self.recover_from_fault(fault_type)
        
        logger.info("All faults cleared")
    
    def get_fault_statistics(self) -> Dict[str, Any]:
        """Get statistics about fault injection."""
        fault_counts = {}
        for event in self.fault_history:
            fault_type = event['fault_type'].value
            fault_counts[fault_type] = fault_counts.get(fault_type, 0) + 1
        
        return {
            'process_id': self.process_id,
            'total_injections': self.injection_count,
            'active_faults': [f.value for f in self.active_faults],
            'fault_counts': fault_counts,
            'fault_history': len(self.fault_history)
        }
    
    def configure_fault_probability(self, fault_type: FaultType, probability: float):
        """Configure probability for a specific fault type."""
        if fault_type not in self.fault_config:
            self.fault_config[fault_type] = {}
        
        self.fault_config[fault_type]['probability'] = probability
        logger.info(f"Updated {fault_type.value} probability to {probability}")
    
    def enable_chaos_mode(self, chaos_level: str = 'medium'):
        """Enable chaos mode with increased fault injection."""
        chaos_configs = {
            'low': {
                'base_probability': 0.02,
                'multiplier': 2
            },
            'medium': {
                'base_probability': 0.05,
                'multiplier': 3
            },
            'high': {
                'base_probability': 0.1,
                'multiplier': 5
            }
        }
        
        config = chaos_configs.get(chaos_level, chaos_configs['medium'])
        
        # Increase all fault probabilities
        for fault_type in self.fault_config:
            current_prob = self.fault_config[fault_type]['probability']
            new_prob = min(0.5, current_prob * config['multiplier'])
            self.fault_config[fault_type]['probability'] = new_prob
        
        self.failure_probability = config['base_probability']
        
        logger.warning(f"Chaos mode enabled: {chaos_level}")
    
    def create_fault_scenario(self, scenario_name: str) -> List[FaultType]:
        """Create predefined fault scenarios for testing."""
        scenarios = {
            'cascade_failure': [
                FaultType.MEMORY_PRESSURE,
                FaultType.SLOW_TRAINING,
                FaultType.GPU_ERROR
            ],
            'network_issues': [
                FaultType.NETWORK_PARTITION,
                FaultType.COMMUNICATION_TIMEOUT
            ],
            'resource_exhaustion': [
                FaultType.MEMORY_PRESSURE,
                FaultType.DISK_FULL
            ],
            'random_chaos': [
                random.choice(list(FaultType))
                for _ in range(random.randint(1, 3))
            ]
        }
        
        faults = scenarios.get(scenario_name, [])
        
        logger.info(f"Creating fault scenario: {scenario_name} with faults: {[f.value for f in faults]}")
        
        # Inject faults with delays
        for i, fault_type in enumerate(faults):
            if i > 0:
                time.sleep(random.uniform(5, 15))  # Random delay between faults
            self.inject_fault(fault_type)
        
        return faults


class AdvancedFaultInjector(FaultInjector):
    """
    Advanced fault injector with more sophisticated failure patterns.
    """
    
    def __init__(self, process_id: str, failure_probability: float = 0.01):
        super().__init__(process_id, failure_probability)
        
        # Advanced patterns
        self.failure_patterns = {
            'burst': False,
            'gradual': False,
            'periodic': False
        }
        
        self.pattern_threads = {}
    
    def start_burst_pattern(self, duration: float = 60.0, intensity: float = 0.1):
        """Start burst failure pattern with high intensity for short duration."""
        def burst_loop():
            start_time = time.time()
            while time.time() - start_time < duration and self.failure_patterns['burst']:
                if random.random() < intensity:
                    self.inject_random_fault()
                time.sleep(1.0)
            self.failure_patterns['burst'] = False
        
        self.failure_patterns['burst'] = True
        thread = threading.Thread(target=burst_loop, daemon=True)
        thread.start()
        self.pattern_threads['burst'] = thread
        
        logger.info(f"Started burst failure pattern: {duration}s duration, {intensity} intensity")
    
    def start_gradual_pattern(self, duration: float = 300.0):
        """Start gradual failure pattern with increasing intensity."""
        def gradual_loop():
            start_time = time.time()
            while time.time() - start_time < duration and self.failure_patterns['gradual']:
                # Increase intensity over time
                elapsed = time.time() - start_time
                intensity = min(0.2, (elapsed / duration) * 0.2)
                
                if random.random() < intensity:
                    self.inject_random_fault()
                
                time.sleep(5.0)
            self.failure_patterns['gradual'] = False
        
        self.failure_patterns['gradual'] = True
        thread = threading.Thread(target=gradual_loop, daemon=True)
        thread.start()
        self.pattern_threads['gradual'] = thread
        
        logger.info(f"Started gradual failure pattern: {duration}s duration")
    
    def start_periodic_pattern(self, period: float = 120.0, intensity: float = 0.05):
        """Start periodic failure pattern with regular intervals."""
        def periodic_loop():
            while self.failure_patterns['periodic']:
                if random.random() < intensity:
                    self.inject_random_fault()
                time.sleep(period)
            
        self.failure_patterns['periodic'] = True
        thread = threading.Thread(target=periodic_loop, daemon=True)
        thread.start()
        self.pattern_threads['periodic'] = thread
        
        logger.info(f"Started periodic failure pattern: {period}s period, {intensity} intensity")
    
    def stop_all_patterns(self):
        """Stop all active failure patterns."""
        for pattern in self.failure_patterns:
            self.failure_patterns[pattern] = False
        
        # Wait for threads to finish
        for thread in self.pattern_threads.values():
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        self.pattern_threads.clear()
        logger.info("All failure patterns stopped")


def main():
    """Test the fault injector."""
    # Create fault injector
    injector = AdvancedFaultInjector(
        process_id="test-injector",
        failure_probability=0.1
    )
    
    # Register test handlers
    def crash_handler(event):
        print(f"CRASH HANDLER: {event}")
    
    def network_handler(event):
        print(f"NETWORK HANDLER: {event}")
    
    injector.register_fault_handler(FaultType.PROCESS_CRASH, crash_handler)
    injector.register_fault_handler(FaultType.NETWORK_PARTITION, network_handler)
    
    try:
        # Test individual fault injection
        print("Testing individual fault injection...")
        injector.inject_fault(FaultType.MEMORY_PRESSURE)
        time.sleep(2)
        
        # Test fault scenario
        print("Testing fault scenario...")
        injector.create_fault_scenario('cascade_failure')
        time.sleep(5)
        
        # Test advanced patterns
        print("Testing burst pattern...")
        injector.start_burst_pattern(duration=10.0, intensity=0.3)
        time.sleep(12)
        
        # Get statistics
        stats = injector.get_fault_statistics()
        print(f"Fault statistics: {stats}")
        
    except KeyboardInterrupt:
        print("Stopping fault injector test")
    finally:
        injector.stop_all_patterns()
        injector.clear_all_faults()


if __name__ == "__main__":
    main() 