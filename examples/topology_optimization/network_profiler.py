"""
Network Profiler for torchLoom Topology Optimization

This module provides real-time network profiling capabilities
to measure bandwidth, latency, and other network characteristics
for optimizing allreduce topology decisions.
"""

import logging
import random
import threading
import time
import socket
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NetworkMetrics:
    """Network performance metrics between nodes."""
    bandwidth_mbps: float
    latency_ms: float
    packet_loss_pct: float
    jitter_ms: float
    timestamp: float


class NetworkProfiler:
    """
    Real-time network profiler for distributed training optimization.
    
    Monitors network conditions between training nodes to enable
    topology optimization decisions.
    """
    
    def __init__(self, process_id: str, rank: int, world_size: int, threadlet=None):
        self.process_id = process_id
        self.rank = rank
        self.world_size = world_size
        self.threadlet = threadlet
        
        # Profiling state
        self.is_running = False
        self.profiling_thread = None
        
        # Network topology discovery
        self.node_addresses = self._discover_node_addresses()
        self.network_matrix = {}
        
        # Profiling configuration
        self.profile_interval = 10.0  # seconds
        self.measurement_history = {}
        
        # Simulated network conditions
        self.base_conditions = self._initialize_base_conditions()
        
        logger.info(f"Network profiler initialized for rank {rank}/{world_size}")
    
    def _discover_node_addresses(self) -> Dict[int, str]:
        """Discover network addresses of all nodes in the cluster."""
        # In a real implementation, this would discover actual node addresses
        # For simulation, generate mock addresses
        addresses = {}
        for rank in range(self.world_size):
            addresses[rank] = f"10.0.0.{rank + 1}"
        
        return addresses
    
    def _initialize_base_conditions(self) -> Dict[str, Any]:
        """Initialize baseline network conditions."""
        return {
            'intra_rack_bandwidth_mbps': random.uniform(8000, 10000),  # 8-10 Gbps
            'inter_rack_bandwidth_mbps': random.uniform(1000, 2000),   # 1-2 Gbps
            'wan_bandwidth_mbps': random.uniform(100, 500),            # 100-500 Mbps
            'base_latency_ms': random.uniform(0.1, 1.0),               # 0.1-1 ms
            'base_packet_loss': random.uniform(0.001, 0.01),           # 0.001-0.01%
            'jitter_ms': random.uniform(0.01, 0.1)                     # 0.01-0.1 ms
        }
    
    def start(self):
        """Start network profiling."""
        if self.is_running:
            logger.warning("Network profiler already running")
            return
        
        self.is_running = True
        self.profiling_thread = threading.Thread(target=self._profiling_loop, daemon=True)
        self.profiling_thread.start()
        logger.info("Network profiler started")
    
    def stop(self):
        """Stop network profiling."""
        self.is_running = False
        if self.profiling_thread:
            self.profiling_thread.join(timeout=5.0)
        logger.info("Network profiler stopped")
    
    def _profiling_loop(self):
        """Main profiling loop."""
        while self.is_running:
            try:
                # Profile connections to all other nodes
                self._profile_all_connections()
                
                # Update network matrix
                self._update_network_matrix()
                
                # Publish profiling results
                self._publish_profiling_results()
                
                time.sleep(self.profile_interval)
                
            except Exception as e:
                logger.error(f"Error in profiling loop: {e}")
                time.sleep(1.0)
    
    def _profile_all_connections(self):
        """Profile network conditions to all other nodes."""
        for target_rank in range(self.world_size):
            if target_rank == self.rank:
                continue
            
            metrics = self._measure_connection(target_rank)
            
            # Store in measurement history
            key = f"{self.rank}->{target_rank}"
            if key not in self.measurement_history:
                self.measurement_history[key] = []
            
            self.measurement_history[key].append(metrics)
            
            # Keep only recent measurements
            max_history = 100
            if len(self.measurement_history[key]) > max_history:
                self.measurement_history[key] = self.measurement_history[key][-max_history:]
    
    def _measure_connection(self, target_rank: int) -> NetworkMetrics:
        """Measure network metrics to a specific target node."""
        target_address = self.node_addresses[target_rank]
        
        # In a real implementation, these would be actual network measurements
        # For simulation, generate realistic values with variations
        
        # Determine connection type based on topology
        connection_type = self._classify_connection(self.rank, target_rank)
        
        if connection_type == 'intra_rack':
            base_bandwidth = self.base_conditions['intra_rack_bandwidth_mbps']
            base_latency = self.base_conditions['base_latency_ms']
        elif connection_type == 'inter_rack':
            base_bandwidth = self.base_conditions['inter_rack_bandwidth_mbps']
            base_latency = self.base_conditions['base_latency_ms'] * 2
        else:  # wan
            base_bandwidth = self.base_conditions['wan_bandwidth_mbps']
            base_latency = self.base_conditions['base_latency_ms'] * 10
        
        # Add realistic variations
        bandwidth = base_bandwidth * random.uniform(0.8, 1.2)
        latency = base_latency * random.uniform(0.9, 1.5)
        packet_loss = self.base_conditions['base_packet_loss'] * random.uniform(0.5, 2.0)
        jitter = self.base_conditions['jitter_ms'] * random.uniform(0.8, 1.5)
        
        # Simulate network congestion effects
        congestion_factor = self._get_congestion_factor()
        bandwidth *= congestion_factor
        latency /= congestion_factor
        
        return NetworkMetrics(
            bandwidth_mbps=bandwidth,
            latency_ms=latency,
            packet_loss_pct=packet_loss,
            jitter_ms=jitter,
            timestamp=time.time()
        )
    
    def _classify_connection(self, rank1: int, rank2: int) -> str:
        """Classify connection type between two ranks."""
        # Simple topology simulation
        rack_size = 2  # 2 nodes per rack
        
        rack1 = rank1 // rack_size
        rack2 = rank2 // rack_size
        
        if rack1 == rack2:
            return 'intra_rack'
        elif abs(rack1 - rack2) == 1:
            return 'inter_rack'
        else:
            return 'wan'
    
    def _get_congestion_factor(self) -> float:
        """Simulate network congestion effects."""
        # Simulate time-based congestion patterns
        hour = time.localtime().tm_hour
        
        if 9 <= hour <= 17:  # Business hours
            base_congestion = 0.7
        elif 19 <= hour <= 23:  # Evening peak
            base_congestion = 0.6
        else:  # Off-peak
            base_congestion = 0.9
        
        # Add random variation
        congestion = base_congestion * random.uniform(0.8, 1.2)
        return max(0.3, min(1.0, congestion))
    
    def _update_network_matrix(self):
        """Update the network connectivity matrix."""
        matrix = {}
        
        for src_rank in range(self.world_size):
            matrix[src_rank] = {}
            for dst_rank in range(self.world_size):
                if src_rank == dst_rank:
                    matrix[src_rank][dst_rank] = {
                        'bandwidth_mbps': float('inf'),
                        'latency_ms': 0.0,
                        'packet_loss_pct': 0.0,
                        'available': True
                    }
                else:
                    # Get latest measurements
                    key = f"{src_rank}->{dst_rank}"
                    if key in self.measurement_history and self.measurement_history[key]:
                        latest = self.measurement_history[key][-1]
                        matrix[src_rank][dst_rank] = {
                            'bandwidth_mbps': latest.bandwidth_mbps,
                            'latency_ms': latest.latency_ms,
                            'packet_loss_pct': latest.packet_loss_pct,
                            'available': latest.packet_loss_pct < 5.0  # Consider unavailable if >5% loss
                        }
                    else:
                        # Default values if no measurements available
                        matrix[src_rank][dst_rank] = {
                            'bandwidth_mbps': 1000.0,
                            'latency_ms': 10.0,
                            'packet_loss_pct': 0.1,
                            'available': True
                        }
        
        self.network_matrix = matrix
    
    def _publish_profiling_results(self):
        """Publish network profiling results."""
        if not self.threadlet:
            return
        
        try:
            # Calculate aggregate metrics
            aggregate_metrics = self._calculate_aggregate_metrics()
            
            profiling_results = {
                'process_id': self.process_id,
                'rank': self.rank,
                'network_matrix': self.network_matrix,
                'aggregate_metrics': aggregate_metrics,
                'timestamp': time.time(),
                'measurement_count': sum(len(history) for history in self.measurement_history.values())
            }
            
            self.threadlet.publish_status(profiling_results)
            
        except Exception as e:
            logger.error(f"Failed to publish profiling results: {e}")
    
    def _calculate_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate network metrics."""
        if not self.measurement_history:
            return {}
        
        all_measurements = []
        for history in self.measurement_history.values():
            all_measurements.extend(history[-10:])  # Last 10 measurements per connection
        
        if not all_measurements:
            return {}
        
        bandwidths = [m.bandwidth_mbps for m in all_measurements]
        latencies = [m.latency_ms for m in all_measurements]
        packet_losses = [m.packet_loss_pct for m in all_measurements]
        
        return {
            'avg_bandwidth_mbps': sum(bandwidths) / len(bandwidths),
            'min_bandwidth_mbps': min(bandwidths),
            'max_bandwidth_mbps': max(bandwidths),
            'avg_latency_ms': sum(latencies) / len(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'avg_packet_loss_pct': sum(packet_losses) / len(packet_losses),
            'max_packet_loss_pct': max(packet_losses),
            'total_connections': len(self.measurement_history),
            'healthy_connections': sum(1 for history in self.measurement_history.values() 
                                     if history and history[-1].packet_loss_pct < 1.0)
        }
    
    def get_bandwidth_matrix(self) -> Dict[int, Dict[int, float]]:
        """Get current bandwidth matrix."""
        bandwidth_matrix = {}
        for src in range(self.world_size):
            bandwidth_matrix[src] = {}
            for dst in range(self.world_size):
                if src in self.network_matrix and dst in self.network_matrix[src]:
                    bandwidth_matrix[src][dst] = self.network_matrix[src][dst]['bandwidth_mbps']
                else:
                    bandwidth_matrix[src][dst] = 1000.0  # Default bandwidth
        
        return bandwidth_matrix
    
    def get_latency_matrix(self) -> Dict[int, Dict[int, float]]:
        """Get current latency matrix."""
        latency_matrix = {}
        for src in range(self.world_size):
            latency_matrix[src] = {}
            for dst in range(self.world_size):
                if src in self.network_matrix and dst in self.network_matrix[src]:
                    latency_matrix[src][dst] = self.network_matrix[src][dst]['latency_ms']
                else:
                    latency_matrix[src][dst] = 10.0  # Default latency
        
        return latency_matrix
    
    def get_current_conditions(self) -> Dict[str, Any]:
        """Get current network conditions summary."""
        aggregate = self._calculate_aggregate_metrics()
        
        return {
            'bandwidth_matrix': self.get_bandwidth_matrix(),
            'latency_matrix': self.get_latency_matrix(),
            'aggregate_metrics': aggregate,
            'network_health': self._assess_network_health(),
            'timestamp': time.time()
        }
    
    def _assess_network_health(self) -> Dict[str, Any]:
        """Assess overall network health."""
        aggregate = self._calculate_aggregate_metrics()
        
        if not aggregate:
            return {'status': 'unknown', 'score': 0.5}
        
        # Health scoring based on multiple factors
        bandwidth_score = min(1.0, aggregate.get('avg_bandwidth_mbps', 1000) / 5000)  # Normalize to 5 Gbps
        latency_score = max(0.0, 1.0 - aggregate.get('avg_latency_ms', 10) / 100)    # Penalty for >100ms
        loss_score = max(0.0, 1.0 - aggregate.get('avg_packet_loss_pct', 0) / 5)     # Penalty for >5% loss
        
        overall_score = (bandwidth_score + latency_score + loss_score) / 3
        
        if overall_score > 0.8:
            status = 'excellent'
        elif overall_score > 0.6:
            status = 'good'
        elif overall_score > 0.4:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'status': status,
            'score': overall_score,
            'bandwidth_score': bandwidth_score,
            'latency_score': latency_score,
            'loss_score': loss_score
        }
    
    def run_comprehensive_profile(self, duration: float = 60.0, 
                                 detailed: bool = True) -> Dict[str, Any]:
        """Run a comprehensive network profiling session."""
        logger.info(f"Starting comprehensive network profile for {duration}s")
        
        start_time = time.time()
        profile_results = {
            'start_time': start_time,
            'duration': duration,
            'measurements': [],
            'topology_analysis': {},
            'recommendations': []
        }
        
        # Intensive profiling for specified duration
        intensive_interval = 1.0  # 1 second intervals for detailed profiling
        while time.time() - start_time < duration:
            # Take measurements
            measurement_batch = {}
            for target_rank in range(self.world_size):
                if target_rank != self.rank:
                    metrics = self._measure_connection(target_rank)
                    measurement_batch[target_rank] = metrics
            
            profile_results['measurements'].append({
                'timestamp': time.time(),
                'measurements': measurement_batch
            })
            
            time.sleep(intensive_interval)
        
        # Analyze results
        if detailed:
            profile_results['topology_analysis'] = self._analyze_topology_characteristics()
            profile_results['recommendations'] = self._generate_topology_recommendations()
        
        logger.info("Comprehensive network profile completed")
        return profile_results
    
    def _analyze_topology_characteristics(self) -> Dict[str, Any]:
        """Analyze network topology characteristics."""
        bandwidth_matrix = self.get_bandwidth_matrix()
        latency_matrix = self.get_latency_matrix()
        
        analysis = {
            'symmetry': self._analyze_symmetry(bandwidth_matrix),
            'hierarchy': self._analyze_hierarchy(bandwidth_matrix),
            'bottlenecks': self._identify_bottlenecks(bandwidth_matrix),
            'optimal_patterns': self._identify_optimal_patterns(bandwidth_matrix, latency_matrix)
        }
        
        return analysis
    
    def _analyze_symmetry(self, bandwidth_matrix: Dict) -> Dict[str, Any]:
        """Analyze bandwidth symmetry in the network."""
        asymmetric_links = 0
        total_links = 0
        max_asymmetry = 0.0
        
        for src in bandwidth_matrix:
            for dst in bandwidth_matrix[src]:
                if src != dst:
                    bw_forward = bandwidth_matrix[src][dst]
                    bw_reverse = bandwidth_matrix.get(dst, {}).get(src, bw_forward)
                    
                    if bw_forward > 0 and bw_reverse > 0:
                        asymmetry = abs(bw_forward - bw_reverse) / max(bw_forward, bw_reverse)
                        max_asymmetry = max(max_asymmetry, asymmetry)
                        
                        if asymmetry > 0.2:  # >20% difference
                            asymmetric_links += 1
                    
                    total_links += 1
        
        return {
            'asymmetric_link_ratio': asymmetric_links / max(1, total_links),
            'max_asymmetry': max_asymmetry,
            'is_symmetric': max_asymmetry < 0.1
        }
    
    def _analyze_hierarchy(self, bandwidth_matrix: Dict) -> Dict[str, Any]:
        """Analyze hierarchical structure in the network."""
        # Simplified hierarchy detection based on bandwidth patterns
        tiers = {}
        
        for src in bandwidth_matrix:
            avg_bandwidth = sum(bw for dst, bw in bandwidth_matrix[src].items() if dst != src) / (len(bandwidth_matrix[src]) - 1)
            
            if avg_bandwidth > 8000:
                tier = 'high_bandwidth'
            elif avg_bandwidth > 2000:
                tier = 'medium_bandwidth'
            else:
                tier = 'low_bandwidth'
            
            if tier not in tiers:
                tiers[tier] = []
            tiers[tier].append(src)
        
        return {
            'detected_tiers': tiers,
            'tier_count': len(tiers),
            'has_hierarchy': len(tiers) > 1
        }
    
    def _identify_bottlenecks(self, bandwidth_matrix: Dict) -> List[Dict[str, Any]]:
        """Identify network bottlenecks."""
        bottlenecks = []
        
        # Find connections with significantly lower bandwidth
        all_bandwidths = []
        for src in bandwidth_matrix:
            for dst, bw in bandwidth_matrix[src].items():
                if src != dst:
                    all_bandwidths.append(bw)
        
        if all_bandwidths:
            avg_bandwidth = sum(all_bandwidths) / len(all_bandwidths)
            threshold = avg_bandwidth * 0.5  # 50% below average
            
            for src in bandwidth_matrix:
                for dst, bw in bandwidth_matrix[src].items():
                    if src != dst and bw < threshold:
                        bottlenecks.append({
                            'src': src,
                            'dst': dst,
                            'bandwidth_mbps': bw,
                            'severity': (threshold - bw) / threshold
                        })
        
        return sorted(bottlenecks, key=lambda x: x['severity'], reverse=True)
    
    def _identify_optimal_patterns(self, bandwidth_matrix: Dict, 
                                 latency_matrix: Dict) -> List[str]:
        """Identify optimal communication patterns."""
        patterns = []
        
        # Check if ring topology would be efficient
        if self._is_ring_efficient(bandwidth_matrix):
            patterns.append('ring')
        
        # Check if tree topology would be efficient
        if self._is_tree_efficient(bandwidth_matrix, latency_matrix):
            patterns.append('tree')
        
        # Check if butterfly/hypercube would be efficient
        if self._is_butterfly_efficient(bandwidth_matrix):
            patterns.append('butterfly')
        
        return patterns
    
    def _is_ring_efficient(self, bandwidth_matrix: Dict) -> bool:
        """Check if ring topology would be efficient."""
        # Ring is efficient when neighbor connections have high bandwidth
        for rank in range(self.world_size):
            next_rank = (rank + 1) % self.world_size
            if bandwidth_matrix.get(rank, {}).get(next_rank, 0) < 2000:  # < 2 Gbps
                return False
        return True
    
    def _is_tree_efficient(self, bandwidth_matrix: Dict, latency_matrix: Dict) -> bool:
        """Check if tree topology would be efficient."""
        # Tree is efficient when there's a clear hierarchy and low latency to root
        hierarchy = self._analyze_hierarchy(bandwidth_matrix)
        return hierarchy['has_hierarchy'] and hierarchy['tier_count'] >= 2
    
    def _is_butterfly_efficient(self, bandwidth_matrix: Dict) -> bool:
        """Check if butterfly topology would be efficient."""
        # Butterfly is efficient when all-to-all bandwidth is high and uniform
        all_bandwidths = []
        for src in bandwidth_matrix:
            for dst, bw in bandwidth_matrix[src].items():
                if src != dst:
                    all_bandwidths.append(bw)
        
        if not all_bandwidths:
            return False
        
        avg_bw = sum(all_bandwidths) / len(all_bandwidths)
        variance = sum((bw - avg_bw) ** 2 for bw in all_bandwidths) / len(all_bandwidths)
        coefficient_of_variation = (variance ** 0.5) / avg_bw
        
        return avg_bw > 5000 and coefficient_of_variation < 0.3  # High bandwidth, low variance
    
    def _generate_topology_recommendations(self) -> List[Dict[str, Any]]:
        """Generate topology optimization recommendations."""
        recommendations = []
        
        network_health = self._assess_network_health()
        bandwidth_matrix = self.get_bandwidth_matrix()
        
        # Recommendation based on network health
        if network_health['score'] < 0.5:
            recommendations.append({
                'type': 'health_improvement',
                'priority': 'high',
                'description': 'Network health is poor, consider using fault-tolerant topology',
                'suggested_topology': 'ring',
                'reasoning': 'Ring topology provides better fault tolerance for degraded networks'
            })
        
        # Recommendation based on bandwidth patterns
        optimal_patterns = self._identify_optimal_patterns(bandwidth_matrix, self.get_latency_matrix())
        
        if 'butterfly' in optimal_patterns:
            recommendations.append({
                'type': 'performance_optimization',
                'priority': 'medium',
                'description': 'Network supports high-performance butterfly topology',
                'suggested_topology': 'butterfly',
                'reasoning': 'Uniform high bandwidth enables efficient butterfly allreduce'
            })
        elif 'tree' in optimal_patterns:
            recommendations.append({
                'type': 'hierarchy_optimization',
                'priority': 'medium',
                'description': 'Hierarchical network structure detected',
                'suggested_topology': 'tree',
                'reasoning': 'Tree topology aligns with network hierarchy'
            })
        
        return recommendations


def main():
    """Test the network profiler."""
    # Create network profiler
    profiler = NetworkProfiler(
        process_id="test-profiler",
        rank=0,
        world_size=4
    )
    
    try:
        # Start profiling
        profiler.start()
        
        # Let it run for a bit
        time.sleep(15)
        
        # Run comprehensive profile
        results = profiler.run_comprehensive_profile(duration=30.0, detailed=True)
        print(f"Comprehensive profile results: {results['topology_analysis']}")
        print(f"Recommendations: {results['recommendations']}")
        
        # Get current conditions
        conditions = profiler.get_current_conditions()
        print(f"Current network health: {conditions['network_health']}")
        
    except KeyboardInterrupt:
        print("Stopping network profiler test")
    finally:
        profiler.stop()


if __name__ == "__main__":
    main() 