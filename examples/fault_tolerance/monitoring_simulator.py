"""
External Monitoring Simulator for torchLoom Fault Tolerance

This module simulates external monitoring systems that can trigger
worker preemption based on various conditions like:
- System resource exhaustion
- Network connectivity issues
- External scheduling decisions
- Hardware health problems
"""

import logging
import random
import threading
import time
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


class MonitoringSimulator:
    """
    Simulates external monitoring system behavior for fault tolerance testing.
    
    This class generates various monitoring events that would typically come
    from real infrastructure monitoring systems like Prometheus, Grafana,
    or cloud provider monitoring services.
    """
    
    def __init__(self, process_id: str, threadlet=None, config: Optional[Dict[str, Any]] = None):
        self.process_id = process_id
        self.threadlet = threadlet
        self.config = config or {}
        
        # Monitoring state
        self.is_running = False
        self.monitoring_thread = None
        
        # Simulation parameters
        self.check_interval = self.config.get('check_interval', 30.0)  # seconds
        self.failure_probability = self.config.get('monitoring_failure_prob', 0.02)  # 2% per check
        self.recovery_probability = self.config.get('recovery_prob', 0.8)  # 80% recovery chance
        
        # Simulated metrics
        self.system_metrics = {
            'cpu_usage': 50.0,
            'memory_usage': 60.0,
            'gpu_utilization': 75.0,
            'network_latency': 10.0,  # ms
            'disk_io': 30.0,
            'temperature': 65.0  # celsius
        }
        
        # Event callbacks
        self.event_handlers = {}
        
        logger.info(f"Monitoring simulator initialized for process: {process_id}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register handler for specific monitoring events."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type: {event_type}")
    
    def start(self):
        """Start the monitoring simulation."""
        if self.is_running:
            logger.warning("Monitoring simulator already running")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Monitoring simulator started")
    
    def stop(self):
        """Stop the monitoring simulation."""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Monitoring simulator stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop that simulates periodic checks."""
        while self.is_running:
            try:
                # Update simulated metrics
                self._update_metrics()
                
                # Check for various failure conditions
                self._check_resource_exhaustion()
                self._check_network_issues()
                self._check_hardware_health()
                self._check_external_signals()
                
                # Publish current metrics
                self._publish_metrics()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def _update_metrics(self):
        """Update simulated system metrics with realistic variations."""
        # Simulate metric fluctuations
        for metric, value in self.system_metrics.items():
            # Add random variation (-5% to +5%)
            variation = random.uniform(-0.05, 0.05)
            new_value = value * (1 + variation)
            
            # Keep within realistic bounds
            if metric == 'cpu_usage':
                new_value = max(10.0, min(100.0, new_value))
            elif metric == 'memory_usage':
                new_value = max(20.0, min(95.0, new_value))
            elif metric == 'gpu_utilization':
                new_value = max(0.0, min(100.0, new_value))
            elif metric == 'network_latency':
                new_value = max(1.0, min(1000.0, new_value))
            elif metric == 'disk_io':
                new_value = max(5.0, min(100.0, new_value))
            elif metric == 'temperature':
                new_value = max(30.0, min(90.0, new_value))
            
            self.system_metrics[metric] = new_value
    
    def _check_resource_exhaustion(self):
        """Check for resource exhaustion conditions."""
        # CPU exhaustion
        if self.system_metrics['cpu_usage'] > 95.0:
            self._trigger_event('cpu_exhaustion', {
                'reason': 'CPU usage exceeded threshold',
                'cpu_usage': self.system_metrics['cpu_usage'],
                'action': 'preempt_worker'
            })
        
        # Memory exhaustion
        if self.system_metrics['memory_usage'] > 90.0:
            self._trigger_event('memory_exhaustion', {
                'reason': 'Memory usage exceeded threshold',
                'memory_usage': self.system_metrics['memory_usage'],
                'action': 'preempt_worker'
            })
        
        # GPU overutilization
        if self.system_metrics['gpu_utilization'] > 98.0:
            self._trigger_event('gpu_exhaustion', {
                'reason': 'GPU utilization exceeded threshold',
                'gpu_utilization': self.system_metrics['gpu_utilization'],
                'action': 'preempt_worker'
            })
    
    def _check_network_issues(self):
        """Check for network-related issues."""
        # High network latency
        if self.system_metrics['network_latency'] > 500.0:
            self._trigger_event('network_degradation', {
                'reason': 'Network latency too high',
                'network_latency': self.system_metrics['network_latency'],
                'action': 'preempt_worker'
            })
        
        # Simulate intermittent network failures
        if random.random() < 0.005:  # 0.5% chance per check
            self._trigger_event('network_failure', {
                'reason': 'Network connectivity lost',
                'action': 'preempt_worker'
            })
    
    def _check_hardware_health(self):
        """Check for hardware health issues."""
        # Temperature threshold
        if self.system_metrics['temperature'] > 85.0:
            self._trigger_event('thermal_throttling', {
                'reason': 'Device temperature too high',
                'temperature': self.system_metrics['temperature'],
                'action': 'preempt_worker'
            })
        
        # Simulate hardware failures
        if random.random() < 0.001:  # 0.1% chance per check
            self._trigger_event('hardware_failure', {
                'reason': 'Hardware failure detected',
                'action': 'preempt_worker'
            })
    
    def _check_external_signals(self):
        """Check for external preemption signals."""
        # Simulate external scheduler decisions
        if random.random() < self.failure_probability:
            reasons = [
                'Higher priority job scheduled',
                'Maintenance window started',
                'Cost optimization triggered',
                'Resource rebalancing required',
                'Spot instance preemption'
            ]
            
            reason = random.choice(reasons)
            self._trigger_event('external_preemption', {
                'reason': reason,
                'action': 'preempt_worker',
                'expected_duration': random.randint(60, 300)  # seconds
            })
    
    def _trigger_event(self, event_type: str, event_data: Dict[str, Any]):
        """Trigger a monitoring event."""
        event_data.update({
            'event_type': event_type,
            'process_id': self.process_id,
            'timestamp': time.time(),
            'source': 'monitoring_simulator'
        })
        
        logger.warning(f"Monitoring event triggered: {event_type} - {event_data['reason']}")
        
        # Call registered handlers
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
        
        # Send to threadlet if available
        if self.threadlet and event_data.get('action') == 'preempt_worker':
            try:
                self.threadlet.handle_message('preempt_worker', event_data['reason'])
            except Exception as e:
                logger.error(f"Failed to send preemption signal via threadlet: {e}")
        
        # Publish event to monitoring system
        self._publish_event(event_data)
    
    def _publish_metrics(self):
        """Publish current metrics to monitoring system."""
        if not self.threadlet:
            return
        
        try:
            metrics_data = {
                'process_id': self.process_id,
                'metrics': self.system_metrics.copy(),
                'timestamp': time.time(),
                'source': 'monitoring_simulator'
            }
            
            self.threadlet.publish_status(metrics_data)
            
        except Exception as e:
            logger.error(f"Failed to publish metrics: {e}")
    
    def _publish_event(self, event_data: Dict[str, Any]):
        """Publish monitoring event."""
        if not self.threadlet:
            return
        
        try:
            self.threadlet.publish_event('monitoring_event', event_data)
        except Exception as e:
            logger.error(f"Failed to publish monitoring event: {e}")
    
    def simulate_recovery(self):
        """Simulate system recovery after issues."""
        logger.info("Simulating system recovery")
        
        # Reset metrics to healthy ranges
        self.system_metrics.update({
            'cpu_usage': random.uniform(30.0, 60.0),
            'memory_usage': random.uniform(40.0, 70.0),
            'gpu_utilization': random.uniform(50.0, 80.0),
            'network_latency': random.uniform(5.0, 20.0),
            'disk_io': random.uniform(20.0, 40.0),
            'temperature': random.uniform(55.0, 70.0)
        })
        
        # Trigger recovery event
        self._trigger_event('system_recovery', {
            'reason': 'System conditions returned to normal',
            'action': 'resume_worker'
        })
    
    def inject_specific_failure(self, failure_type: str, severity: str = 'moderate'):
        """Inject a specific type of failure for testing."""
        logger.info(f"Injecting {severity} {failure_type} failure")
        
        if failure_type == 'cpu_spike':
            self.system_metrics['cpu_usage'] = 98.0 if severity == 'severe' else 85.0
        elif failure_type == 'memory_leak':
            self.system_metrics['memory_usage'] = 95.0 if severity == 'severe' else 80.0
        elif failure_type == 'network_latency':
            self.system_metrics['network_latency'] = 800.0 if severity == 'severe' else 300.0
        elif failure_type == 'thermal_event':
            self.system_metrics['temperature'] = 90.0 if severity == 'severe' else 80.0
        
        # Force immediate check
        threading.Thread(target=self._check_all_conditions, daemon=True).start()
    
    def _check_all_conditions(self):
        """Check all monitoring conditions immediately."""
        self._check_resource_exhaustion()
        self._check_network_issues()
        self._check_hardware_health()
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return {
            'process_id': self.process_id,
            'metrics': self.system_metrics.copy(),
            'timestamp': time.time(),
            'is_healthy': self._is_system_healthy()
        }
    
    def _is_system_healthy(self) -> bool:
        """Check if system is currently healthy."""
        return (
            self.system_metrics['cpu_usage'] < 95.0 and
            self.system_metrics['memory_usage'] < 90.0 and
            self.system_metrics['network_latency'] < 500.0 and
            self.system_metrics['temperature'] < 85.0
        )


class AdvancedMonitoringSimulator(MonitoringSimulator):
    """
    Advanced monitoring simulator with more sophisticated failure patterns.
    """
    
    def __init__(self, process_id: str, threadlet=None, config: Optional[Dict[str, Any]] = None):
        super().__init__(process_id, threadlet, config)
        
        # Advanced simulation state
        self.failure_patterns = {
            'cascading': False,
            'intermittent': False,
            'gradual_degradation': False
        }
        
        self.pattern_timers = {}
    
    def start_failure_pattern(self, pattern_type: str, duration: float = 300.0):
        """Start a specific failure pattern."""
        logger.info(f"Starting failure pattern: {pattern_type}")
        
        self.failure_patterns[pattern_type] = True
        
        # Set timer to end pattern
        def end_pattern():
            time.sleep(duration)
            self.failure_patterns[pattern_type] = False
            logger.info(f"Failure pattern ended: {pattern_type}")
        
        pattern_thread = threading.Thread(target=end_pattern, daemon=True)
        pattern_thread.start()
        self.pattern_timers[pattern_type] = pattern_thread
    
    def _update_metrics(self):
        """Update metrics with advanced failure patterns."""
        super()._update_metrics()
        
        # Apply failure patterns
        if self.failure_patterns['cascading']:
            self._apply_cascading_failure()
        
        if self.failure_patterns['intermittent']:
            self._apply_intermittent_failure()
        
        if self.failure_patterns['gradual_degradation']:
            self._apply_gradual_degradation()
    
    def _apply_cascading_failure(self):
        """Simulate cascading failure where one issue leads to others."""
        if self.system_metrics['cpu_usage'] > 80.0:
            # High CPU leads to increased temperature
            self.system_metrics['temperature'] += 2.0
            
            # Which leads to memory pressure
            self.system_metrics['memory_usage'] += 1.5
            
            # And network latency
            self.system_metrics['network_latency'] += 10.0
    
    def _apply_intermittent_failure(self):
        """Simulate intermittent failures that come and go."""
        if random.random() < 0.1:  # 10% chance
            # Spike one random metric
            metric = random.choice(list(self.system_metrics.keys()))
            if metric == 'network_latency':
                self.system_metrics[metric] *= 3.0
            else:
                self.system_metrics[metric] = min(100.0, self.system_metrics[metric] * 1.5)
    
    def _apply_gradual_degradation(self):
        """Simulate gradual system degradation over time."""
        # Slowly increase all stress metrics
        degradation_rate = 0.01  # 1% per check
        
        self.system_metrics['cpu_usage'] += degradation_rate
        self.system_metrics['memory_usage'] += degradation_rate
        self.system_metrics['temperature'] += 0.1
        self.system_metrics['network_latency'] += 1.0


def main():
    """Test the monitoring simulator."""
    import time
    
    # Create simulator
    simulator = AdvancedMonitoringSimulator(
        process_id="test-monitor",
        config={
            'check_interval': 5.0,
            'monitoring_failure_prob': 0.1
        }
    )
    
    # Register test event handler
    def test_handler(event_data):
        print(f"Event received: {event_data['event_type']} - {event_data['reason']}")
    
    simulator.register_event_handler('external_preemption', test_handler)
    simulator.register_event_handler('cpu_exhaustion', test_handler)
    
    # Start monitoring
    simulator.start()
    
    try:
        # Let it run for a bit
        time.sleep(10)
        
        # Inject some failures
        simulator.inject_specific_failure('cpu_spike', 'severe')
        time.sleep(5)
        
        simulator.start_failure_pattern('cascading', 30)
        time.sleep(15)
        
        simulator.simulate_recovery()
        time.sleep(10)
        
    except KeyboardInterrupt:
        print("Stopping monitoring simulator test")
    finally:
        simulator.stop()


if __name__ == "__main__":
    main() 