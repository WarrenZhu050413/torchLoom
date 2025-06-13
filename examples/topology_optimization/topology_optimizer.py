"""
Topology Optimizer for torchLoom

This module optimizes allreduce topology based on real-time network profiling
and performance metrics to maximize communication efficiency.
"""

import logging
import math
import random
import threading
import time
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class TopologyType(Enum):
    """Available allreduce topology types."""
    RING = "ring"
    TREE = "tree"
    BUTTERFLY = "butterfly"
    ALL_TO_ALL = "all_to_all"
    HIERARCHICAL = "hierarchical"


@dataclass
class TopologyPerformance:
    """Performance metrics for a specific topology."""
    topology: TopologyType
    estimated_time_ms: float
    bandwidth_utilization: float
    fault_tolerance_score: float
    scalability_score: float
    overall_score: float


class TopologyOptimizer:
    """
    Optimizes allreduce topology based on network conditions and performance metrics.
    
    Uses real-time network profiling data to select optimal communication
    topologies for distributed training workloads.
    """
    
    def __init__(self, process_id: str, world_size: int, network_profiler=None, threadlet=None):
        self.process_id = process_id
        self.world_size = world_size
        self.network_profiler = network_profiler
        self.threadlet = threadlet
        
        # Optimization state
        self.is_running = False
        self.optimization_thread = None
        
        # Performance history
        self.topology_performance_history = {topology: [] for topology in TopologyType}
        self.current_topology = TopologyType.RING
        
        # Optimization parameters
        self.optimization_interval = 30.0  # seconds
        self.performance_window = 10  # Keep last 10 measurements
        
        # Topology confidence scores
        self.topology_confidence = {topology: 0.5 for topology in TopologyType}
        
        logger.info(f"Topology optimizer initialized for world size {world_size}")
    
    def start(self):
        """Start topology optimization."""
        if self.is_running:
            logger.warning("Topology optimizer already running")
            return
        
        self.is_running = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        logger.info("Topology optimizer started")
    
    def stop(self):
        """Stop topology optimization."""
        self.is_running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5.0)
        logger.info("Topology optimizer stopped")
    
    def _optimization_loop(self):
        """Main optimization loop."""
        while self.is_running:
            try:
                # Get current network conditions
                if self.network_profiler:
                    network_conditions = self.network_profiler.get_current_conditions()
                    
                    # Evaluate all topologies
                    topology_scores = self._evaluate_all_topologies(network_conditions)
                    
                    # Update confidence scores
                    self._update_confidence_scores(topology_scores)
                    
                    # Check if topology switch is beneficial
                    optimal_topology = self._select_optimal_topology(topology_scores)
                    
                    if optimal_topology != self.current_topology:
                        self._recommend_topology_switch(optimal_topology, topology_scores)
                
                time.sleep(self.optimization_interval)
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(1.0)
    
    def _evaluate_all_topologies(self, network_conditions: Dict[str, Any]) -> Dict[TopologyType, TopologyPerformance]:
        """Evaluate performance of all available topologies."""
        bandwidth_matrix = network_conditions.get('bandwidth_matrix', {})
        latency_matrix = network_conditions.get('latency_matrix', {})
        
        topology_scores = {}
        
        for topology in TopologyType:
            performance = self._evaluate_topology(topology, bandwidth_matrix, latency_matrix)
            topology_scores[topology] = performance
        
        return topology_scores
    
    def _evaluate_topology(self, topology: TopologyType, 
                          bandwidth_matrix: Dict, 
                          latency_matrix: Dict) -> TopologyPerformance:
        """Evaluate performance of a specific topology."""
        
        # Estimate communication time
        estimated_time = self.estimate_allreduce_time(
            topology=topology,
            tensor_size_mb=100.0,  # Standard test size
            bandwidth_matrix=bandwidth_matrix,
            latency_matrix=latency_matrix
        )
        
        # Calculate bandwidth utilization
        bandwidth_util = self._calculate_bandwidth_utilization(
            topology, bandwidth_matrix
        )
        
        # Calculate fault tolerance score
        fault_tolerance = self._calculate_fault_tolerance_score(
            topology, bandwidth_matrix
        )
        
        # Calculate scalability score
        scalability = self._calculate_scalability_score(topology)
        
        # Calculate overall score (weighted combination)
        overall_score = (
            0.4 * (1.0 - min(1.0, estimated_time / 1000.0)) +  # Time penalty
            0.3 * bandwidth_util +
            0.2 * fault_tolerance +
            0.1 * scalability
        )
        
        return TopologyPerformance(
            topology=topology,
            estimated_time_ms=estimated_time,
            bandwidth_utilization=bandwidth_util,
            fault_tolerance_score=fault_tolerance,
            scalability_score=scalability,
            overall_score=overall_score
        )
    
    def estimate_allreduce_time(self, topology: TopologyType, 
                               tensor_size_mb: float,
                               bandwidth_matrix: Dict,
                               latency_matrix: Optional[Dict] = None) -> float:
        """Estimate allreduce completion time for a given topology."""
        
        if topology == TopologyType.RING:
            return self._estimate_ring_time(tensor_size_mb, bandwidth_matrix, latency_matrix)
        elif topology == TopologyType.TREE:
            return self._estimate_tree_time(tensor_size_mb, bandwidth_matrix, latency_matrix)
        elif topology == TopologyType.BUTTERFLY:
            return self._estimate_butterfly_time(tensor_size_mb, bandwidth_matrix, latency_matrix)
        elif topology == TopologyType.ALL_TO_ALL:
            return self._estimate_all_to_all_time(tensor_size_mb, bandwidth_matrix, latency_matrix)
        elif topology == TopologyType.HIERARCHICAL:
            return self._estimate_hierarchical_time(tensor_size_mb, bandwidth_matrix, latency_matrix)
        else:
            return 1000.0  # Default high value
    
    def _estimate_ring_time(self, tensor_size_mb: float, 
                           bandwidth_matrix: Dict, 
                           latency_matrix: Optional[Dict]) -> float:
        """Estimate ring allreduce time."""
        if not bandwidth_matrix:
            return 1000.0
        
        # Ring allreduce: 2 * (N-1) / N * tensor_size / min_bandwidth
        min_bandwidth = float('inf')
        
        for rank in range(self.world_size):
            next_rank = (rank + 1) % self.world_size
            if rank in bandwidth_matrix and next_rank in bandwidth_matrix[rank]:
                bw = bandwidth_matrix[rank][next_rank]
                min_bandwidth = min(min_bandwidth, bw)
        
        if min_bandwidth == float('inf'):
            min_bandwidth = 1000.0  # Default 1 Gbps
        
        # Ring communication time (simplified model)
        transfer_time = 2 * (self.world_size - 1) / self.world_size * tensor_size_mb * 8 / min_bandwidth * 1000  # ms
        
        # Add latency component
        if latency_matrix:
            avg_latency = self._get_average_latency(latency_matrix)
            latency_overhead = 2 * (self.world_size - 1) * avg_latency
        else:
            latency_overhead = 2 * (self.world_size - 1) * 1.0  # Default 1ms latency
        
        return transfer_time + latency_overhead
    
    def _estimate_tree_time(self, tensor_size_mb: float, 
                           bandwidth_matrix: Dict, 
                           latency_matrix: Optional[Dict]) -> float:
        """Estimate tree allreduce time."""
        if not bandwidth_matrix:
            return 1000.0
        
        # Tree allreduce: log2(N) phases, each phase uses tree bandwidth
        tree_depth = math.ceil(math.log2(self.world_size))
        
        # Find average bandwidth for tree connections
        total_bandwidth = 0
        connection_count = 0
        
        for rank in bandwidth_matrix:
            for target_rank, bw in bandwidth_matrix[rank].items():
                if rank != target_rank:
                    total_bandwidth += bw
                    connection_count += 1
        
        avg_bandwidth = total_bandwidth / max(1, connection_count)
        
        # Tree communication time
        transfer_time = tree_depth * tensor_size_mb * 8 / avg_bandwidth * 1000  # ms
        
        # Add latency component
        if latency_matrix:
            avg_latency = self._get_average_latency(latency_matrix)
            latency_overhead = tree_depth * avg_latency
        else:
            latency_overhead = tree_depth * 1.0
        
        return transfer_time + latency_overhead
    
    def _estimate_butterfly_time(self, tensor_size_mb: float, 
                                bandwidth_matrix: Dict, 
                                latency_matrix: Optional[Dict]) -> float:
        """Estimate butterfly allreduce time."""
        if not bandwidth_matrix:
            return 1000.0
        
        # Butterfly allreduce: log2(N) phases with parallel communication
        phases = math.ceil(math.log2(self.world_size))
        
        # Calculate effective bandwidth (parallel transfers)
        min_parallel_bandwidth = float('inf')
        
        for phase in range(phases):
            phase_bandwidth = 0
            phase_connections = 0
            
            # In each phase, each node communicates with one other node
            for rank in range(self.world_size):
                partner = rank ^ (1 << phase)  # XOR to find butterfly partner
                if partner < self.world_size and rank in bandwidth_matrix:
                    bw = bandwidth_matrix[rank].get(partner, 1000.0)
                    phase_bandwidth += bw
                    phase_connections += 1
            
            if phase_connections > 0:
                avg_phase_bandwidth = phase_bandwidth / phase_connections
                min_parallel_bandwidth = min(min_parallel_bandwidth, avg_phase_bandwidth)
        
        if min_parallel_bandwidth == float('inf'):
            min_parallel_bandwidth = 1000.0
        
        # Butterfly communication time
        transfer_time = phases * tensor_size_mb * 8 / min_parallel_bandwidth * 1000  # ms
        
        # Add latency component
        if latency_matrix:
            avg_latency = self._get_average_latency(latency_matrix)
            latency_overhead = phases * avg_latency
        else:
            latency_overhead = phases * 1.0
        
        return transfer_time + latency_overhead
    
    def _estimate_all_to_all_time(self, tensor_size_mb: float, 
                                 bandwidth_matrix: Dict, 
                                 latency_matrix: Optional[Dict]) -> float:
        """Estimate all-to-all allreduce time."""
        if not bandwidth_matrix:
            return 1000.0
        
        # All-to-all: Each node sends to all others simultaneously
        min_bandwidth = float('inf')
        
        for rank in bandwidth_matrix:
            rank_min_bw = float('inf')
            for target_rank, bw in bandwidth_matrix[rank].items():
                if rank != target_rank:
                    rank_min_bw = min(rank_min_bw, bw)
            
            if rank_min_bw != float('inf'):
                min_bandwidth = min(min_bandwidth, rank_min_bw)
        
        if min_bandwidth == float('inf'):
            min_bandwidth = 1000.0
        
        # All-to-all communication time (limited by slowest link)
        transfer_time = tensor_size_mb * 8 / min_bandwidth * 1000  # ms
        
        # Add latency component (single phase)
        if latency_matrix:
            avg_latency = self._get_average_latency(latency_matrix)
            latency_overhead = avg_latency
        else:
            latency_overhead = 1.0
        
        return transfer_time + latency_overhead
    
    def _estimate_hierarchical_time(self, tensor_size_mb: float, 
                                   bandwidth_matrix: Dict, 
                                   latency_matrix: Optional[Dict]) -> float:
        """Estimate hierarchical allreduce time."""
        # Simplified hierarchical model: local reduction + global reduction
        local_time = self._estimate_ring_time(tensor_size_mb, bandwidth_matrix, latency_matrix) * 0.5
        global_time = self._estimate_tree_time(tensor_size_mb, bandwidth_matrix, latency_matrix) * 0.5
        
        return local_time + global_time
    
    def _get_average_latency(self, latency_matrix: Dict) -> float:
        """Calculate average latency across all connections."""
        total_latency = 0
        connection_count = 0
        
        for rank in latency_matrix:
            for target_rank, latency in latency_matrix[rank].items():
                if rank != target_rank:
                    total_latency += latency
                    connection_count += 1
        
        return total_latency / max(1, connection_count)
    
    def _calculate_bandwidth_utilization(self, topology: TopologyType, 
                                        bandwidth_matrix: Dict) -> float:
        """Calculate expected bandwidth utilization for a topology."""
        if not bandwidth_matrix:
            return 0.5
        
        if topology == TopologyType.RING:
            # Ring uses sequential communication
            return 0.6
        elif topology == TopologyType.TREE:
            # Tree has good utilization with hierarchy
            return 0.7
        elif topology == TopologyType.BUTTERFLY:
            # Butterfly maximizes parallel communication
            return 0.9
        elif topology == TopologyType.ALL_TO_ALL:
            # All-to-all can be bandwidth-limited
            return 0.8
        elif topology == TopologyType.HIERARCHICAL:
            # Hierarchical balances local and global communication
            return 0.75
        else:
            return 0.5
    
    def _calculate_fault_tolerance_score(self, topology: TopologyType, 
                                        bandwidth_matrix: Dict) -> float:
        """Calculate fault tolerance score for a topology."""
        if topology == TopologyType.RING:
            return 0.3  # Ring fails if any link breaks
        elif topology == TopologyType.TREE:
            return 0.5  # Tree has single points of failure
        elif topology == TopologyType.BUTTERFLY:
            return 0.8  # Butterfly has multiple paths
        elif topology == TopologyType.ALL_TO_ALL:
            return 0.9  # All-to-all has maximum redundancy
        elif topology == TopologyType.HIERARCHICAL:
            return 0.7  # Hierarchical has good fault tolerance
        else:
            return 0.5
    
    def _calculate_scalability_score(self, topology: TopologyType) -> float:
        """Calculate scalability score for a topology."""
        if topology == TopologyType.RING:
            return max(0.1, 1.0 - self.world_size / 100.0)  # Ring degrades with size
        elif topology == TopologyType.TREE:
            return 0.8  # Tree scales well
        elif topology == TopologyType.BUTTERFLY:
            return 0.9  # Butterfly scales excellently
        elif topology == TopologyType.ALL_TO_ALL:
            return max(0.2, 1.0 - self.world_size / 50.0)  # All-to-all degrades faster
        elif topology == TopologyType.HIERARCHICAL:
            return 0.95  # Hierarchical scales best
        else:
            return 0.5
    
    def _update_confidence_scores(self, topology_scores: Dict[TopologyType, TopologyPerformance]):
        """Update confidence scores based on recent evaluations."""
        for topology, performance in topology_scores.items():
            # Update confidence based on performance score
            current_confidence = self.topology_confidence[topology]
            new_confidence = 0.7 * current_confidence + 0.3 * performance.overall_score
            self.topology_confidence[topology] = new_confidence
            
            # Store performance history
            if topology not in self.topology_performance_history:
                self.topology_performance_history[topology] = []
            
            self.topology_performance_history[topology].append(performance)
            
            # Keep only recent history
            if len(self.topology_performance_history[topology]) > self.performance_window:
                self.topology_performance_history[topology] = \
                    self.topology_performance_history[topology][-self.performance_window:]
    
    def _select_optimal_topology(self, topology_scores: Dict[TopologyType, TopologyPerformance]) -> TopologyType:
        """Select optimal topology based on scores and confidence."""
        best_topology = TopologyType.RING
        best_weighted_score = 0.0
        
        for topology, performance in topology_scores.items():
            confidence = self.topology_confidence[topology]
            weighted_score = performance.overall_score * confidence
            
            if weighted_score > best_weighted_score:
                best_weighted_score = weighted_score
                best_topology = topology
        
        return best_topology
    
    def _recommend_topology_switch(self, optimal_topology: TopologyType, 
                                  topology_scores: Dict[TopologyType, TopologyPerformance]):
        """Recommend topology switch if beneficial."""
        current_performance = topology_scores.get(self.current_topology)
        optimal_performance = topology_scores.get(optimal_topology)
        
        if current_performance and optimal_performance:
            improvement = optimal_performance.overall_score - current_performance.overall_score
            
            # Only recommend switch if improvement is significant
            if improvement > 0.1:  # 10% improvement threshold
                logger.info(f"Recommending topology switch: {self.current_topology.value} -> "
                          f"{optimal_topology.value} (improvement: {improvement:.3f})")
                
                # Publish recommendation
                if self.threadlet:
                    self.threadlet.publish_event('topology_recommendation', {
                        'current_topology': self.current_topology.value,
                        'recommended_topology': optimal_topology.value,
                        'improvement': improvement,
                        'current_score': current_performance.overall_score,
                        'optimal_score': optimal_performance.overall_score,
                        'timestamp': time.time()
                    })
    
    def optimize_based_on_bandwidth(self, bandwidth_data: Dict[str, Any]):
        """Optimize topology based on updated bandwidth data."""
        logger.info("Optimizing topology based on updated bandwidth data")
        
        # Simulate topology evaluation with new bandwidth data
        if self.network_profiler:
            network_conditions = self.network_profiler.get_current_conditions()
            topology_scores = self._evaluate_all_topologies(network_conditions)
            optimal_topology = self._select_optimal_topology(topology_scores)
            
            if optimal_topology != self.current_topology:
                self._recommend_topology_switch(optimal_topology, topology_scores)
    
    def find_optimal_topology(self, **optimization_params) -> TopologyType:
        """Find optimal topology for current conditions."""
        if self.network_profiler:
            network_conditions = self.network_profiler.get_current_conditions()
            topology_scores = self._evaluate_all_topologies(network_conditions)
            return self._select_optimal_topology(topology_scores)
        else:
            return TopologyType.RING  # Default fallback
    
    def optimize_for_current_conditions(self) -> TopologyType:
        """Optimize topology for current network conditions."""
        if self.network_profiler:
            network_conditions = self.network_profiler.get_current_conditions()
            
            # Consider current network health
            network_health = network_conditions.get('network_health', {})
            health_score = network_health.get('score', 0.5)
            
            if health_score < 0.4:
                # Poor network health: prefer fault-tolerant topologies
                return TopologyType.ALL_TO_ALL
            elif health_score > 0.8:
                # Excellent network health: prefer high-performance topologies
                return TopologyType.BUTTERFLY
            else:
                # Medium network health: balanced approach
                return TopologyType.TREE
        else:
            return TopologyType.RING
    
    def get_topology_confidence(self, topology: TopologyType) -> float:
        """Get confidence score for a specific topology."""
        return self.topology_confidence.get(topology, 0.5)
    
    def get_topology_statistics(self) -> Dict[str, Any]:
        """Get comprehensive topology optimization statistics."""
        stats = {
            'current_topology': self.current_topology.value,
            'topology_confidence': {t.value: c for t, c in self.topology_confidence.items()},
            'optimization_count': sum(len(history) for history in self.topology_performance_history.values()),
            'best_topologies': {}
        }
        
        # Find best topology for each metric
        best_time_topology = min(TopologyType, 
                               key=lambda t: self.topology_confidence.get(t, float('inf')))
        best_bandwidth_topology = max(TopologyType, 
                                    key=lambda t: self.topology_confidence.get(t, 0))
        
        stats['best_topologies'] = {
            'overall': best_bandwidth_topology.value,
            'fault_tolerance': TopologyType.ALL_TO_ALL.value,
            'scalability': TopologyType.HIERARCHICAL.value
        }
        
        return stats
    
    def force_topology_evaluation(self) -> Dict[TopologyType, TopologyPerformance]:
        """Force immediate evaluation of all topologies."""
        if self.network_profiler:
            network_conditions = self.network_profiler.get_current_conditions()
            return self._evaluate_all_topologies(network_conditions)
        else:
            # Return default evaluations
            default_performance = {}
            for topology in TopologyType:
                default_performance[topology] = TopologyPerformance(
                    topology=topology,
                    estimated_time_ms=500.0,
                    bandwidth_utilization=0.5,
                    fault_tolerance_score=0.5,
                    scalability_score=0.5,
                    overall_score=0.5
                )
            return default_performance


if __name__ == "__main__":
    # Test the topology optimizer
    class MockNetworkProfiler:
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
    
    profiler = MockNetworkProfiler()
    optimizer = TopologyOptimizer("test-optimizer", 4, profiler)
    
    # Test topology evaluation
    evaluations = optimizer.force_topology_evaluation()
    for topology, performance in evaluations.items():
        print(f"{topology.value}: score={performance.overall_score:.3f}, "
              f"time={performance.estimated_time_ms:.1f}ms")
    
    optimal = optimizer.find_optimal_topology()
    print(f"Optimal topology: {optimal.value}") 