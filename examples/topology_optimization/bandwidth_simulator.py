"""
Bandwidth Simulator for torchLoom Topology Optimization Demo

This module simulates various network conditions and bandwidth patterns
to demonstrate topology optimization capabilities.
"""

import logging
import random
import threading
import time
from typing import Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class NetworkScenario(Enum):
    """Available network simulation scenarios."""
    STABLE = "stable"
    CONGESTED = "congested" 
    UNSTABLE = "unstable"
    HIERARCHICAL = "hierarchical"
    ASYMMETRIC = "asymmetric"


class BandwidthSimulator:
    """
    Simulates network bandwidth conditions for topology optimization testing.
    """
    
    def __init__(self, process_id: str, num_nodes: int = 4, base_bandwidth: float = 10000.0):
        self.process_id = process_id
        self.num_nodes = num_nodes
        self.base_bandwidth = base_bandwidth  # Mbps
        
        # Simulation state
        self.is_running = False
        self.simulation_thread = None
        self.current_scenario = None
        
        # Network matrices
        self.bandwidth_matrix = {}
        self.latency_matrix = {}
        
        # Event management
        self.active_events = []
        
        # Simulation parameters
        self.update_interval = 2.0  # seconds
        self.noise_factor = 0.1  # 10% random variation
        
        # Initialize base network topology
        self._initialize_network_topology()
        
        logger.info(f"Bandwidth simulator initialized for {num_nodes} nodes")
    
    def start(self, scenario: str = "stable"):
        """Start network simulation with specified scenario."""
        if self.is_running:
            logger.warning("Bandwidth simulator already running")
            return
        
        try:
            self.current_scenario = NetworkScenario(scenario)
        except ValueError:
            logger.error(f"Unknown scenario: {scenario}")
            self.current_scenario = NetworkScenario.STABLE
        
        self.is_running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
        
        logger.info(f"Bandwidth simulation started with scenario: {self.current_scenario.value}")
    
    def stop(self):
        """Stop network simulation."""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=5.0)
        
        self.active_events.clear()
        logger.info("Bandwidth simulation stopped")
    
    def _initialize_network_topology(self):
        """Initialize base network topology with default values."""
        # Create fully connected topology
        for i in range(self.num_nodes):
            self.bandwidth_matrix[i] = {}
            self.latency_matrix[i] = {}
            
            for j in range(self.num_nodes):
                if i != j:
                    # Base bandwidth varies by distance (simulated)
                    distance_factor = 1.0 - abs(i - j) * 0.1
                    self.bandwidth_matrix[i][j] = self.base_bandwidth * distance_factor
                    
                    # Base latency increases with distance
                    self.latency_matrix[i][j] = 1.0 + abs(i - j) * 0.5  # ms
    
    def _simulation_loop(self):
        """Main simulation loop."""
        while self.is_running:
            try:
                # Update network conditions based on current scenario
                self._update_network_conditions()
                
                # Apply random noise
                self._apply_network_noise()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                time.sleep(1.0)
    
    def _update_network_conditions(self):
        """Update network conditions based on current scenario."""
        if self.current_scenario == NetworkScenario.STABLE:
            self._simulate_stable_network()
        elif self.current_scenario == NetworkScenario.CONGESTED:
            self._simulate_congested_network()
        elif self.current_scenario == NetworkScenario.UNSTABLE:
            self._simulate_unstable_network()
        elif self.current_scenario == NetworkScenario.HIERARCHICAL:
            self._simulate_hierarchical_network()
        elif self.current_scenario == NetworkScenario.ASYMMETRIC:
            self._simulate_asymmetric_network()
    
    def _simulate_stable_network(self):
        """Simulate stable network conditions with minimal variation."""
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    distance_factor = 1.0 - abs(i - j) * 0.05
                    self.bandwidth_matrix[i][j] = self.base_bandwidth * distance_factor * 0.95
                    self.latency_matrix[i][j] = 1.0 + abs(i - j) * 0.2
    
    def _simulate_congested_network(self):
        """Simulate congested network with reduced bandwidth."""
        congestion_factor = 0.4 + 0.3 * random.random()  # 40-70% of base bandwidth
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    distance_factor = 1.0 - abs(i - j) * 0.1
                    self.bandwidth_matrix[i][j] = self.base_bandwidth * distance_factor * congestion_factor
                    congestion_latency = 1.5 + random.random() * 2.0
                    self.latency_matrix[i][j] = (1.0 + abs(i - j) * 0.5) * congestion_latency
    
    def _simulate_unstable_network(self):
        """Simulate unstable network with high variation."""
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    variation = 0.5 + random.random()  # 50-150% variation
                    distance_factor = 1.0 - abs(i - j) * 0.15
                    self.bandwidth_matrix[i][j] = self.base_bandwidth * distance_factor * variation
                    latency_variation = 0.5 + random.random() * 3.0
                    self.latency_matrix[i][j] = (1.0 + abs(i - j) * 0.5) * latency_variation
    
    def _simulate_hierarchical_network(self):
        """Simulate hierarchical network topology."""
        groups = [[0, 1], [2, 3]]
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    same_group = any(i in group and j in group for group in groups)
                    
                    if same_group:
                        self.bandwidth_matrix[i][j] = self.base_bandwidth * 1.2
                        self.latency_matrix[i][j] = 0.5
                    else:
                        self.bandwidth_matrix[i][j] = self.base_bandwidth * 0.3
                        self.latency_matrix[i][j] = 5.0
    
    def _simulate_asymmetric_network(self):
        """Simulate asymmetric bandwidth patterns."""
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    if i < j:
                        self.bandwidth_matrix[i][j] = self.base_bandwidth * 1.1
                        self.latency_matrix[i][j] = 1.0
                    else:
                        self.bandwidth_matrix[i][j] = self.base_bandwidth * 0.6
                        self.latency_matrix[i][j] = 2.5
    
    def _apply_network_noise(self):
        """Apply random noise to network conditions."""
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    noise = 1.0 + (random.random() - 0.5) * self.noise_factor
                    self.bandwidth_matrix[i][j] *= noise
                    self.bandwidth_matrix[i][j] = max(1.0, self.bandwidth_matrix[i][j])
    
    def inject_network_event(self, event_type: str, parameters: Dict[str, Any]):
        """Inject a network event into the simulation."""
        logger.info(f"Injected network event: {event_type}")
        
        if event_type == "congestion_spike":
            severity = parameters.get('severity', 0.5)
            for i in self.bandwidth_matrix:
                for j in self.bandwidth_matrix[i]:
                    self.bandwidth_matrix[i][j] *= (1.0 - severity)
        
        elif event_type == "link_failure":
            src = parameters.get('src', 0)
            dst = parameters.get('dst', 1)
            if src < self.num_nodes and dst < self.num_nodes:
                self.bandwidth_matrix[src][dst] = 1.0
                self.bandwidth_matrix[dst][src] = 1.0
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status."""
        total_bandwidth = sum(sum(row.values()) for row in self.bandwidth_matrix.values())
        avg_bandwidth = total_bandwidth / (self.num_nodes * (self.num_nodes - 1))
        
        return {
            'is_running': self.is_running,
            'scenario': self.current_scenario.value if self.current_scenario else None,
            'avg_bandwidth_mbps': avg_bandwidth,
            'health_score': min(1.0, avg_bandwidth / self.base_bandwidth),
            'nodes': self.num_nodes
        }


if __name__ == "__main__":
    simulator = BandwidthSimulator("test-simulator", num_nodes=4)
    
    scenarios = ["stable", "congested", "unstable", "hierarchical", "asymmetric"]
    
    for scenario in scenarios:
        print(f"\nTesting scenario: {scenario}")
        simulator.start(scenario)
        time.sleep(3)
        
        status = simulator.get_network_status()
        print(f"Network status: {status}")
        
        simulator.stop()
        time.sleep(1)
    
    print("\nBandwidth simulator test completed") 