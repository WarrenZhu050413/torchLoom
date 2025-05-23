"""
Status tracking for torchLoom Weaver.

This module manages the state of all replicas, GPUs, and training progress
within the weaver. This is weaver state, not UI state - it just happens
that this state gets published to the UI.
"""

import asyncio
import time
from typing import Dict, Set, Optional, List, Any
from dataclasses import dataclass, field
from torchLoom.log.logger import setup_logger

logger = setup_logger(name="status_tracker")


@dataclass
class GPUState:
    """State information for a single GPU."""
    gpu_id: str
    replica_id: str
    server_id: str
    status: str = "active"  # "active", "offline", "failed"
    utilization: float = 0.0
    temperature: float = 40.0
    memory_used: float = 0.0  # GB
    memory_total: float = 0.0  # GB
    config: Dict[str, str] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)


@dataclass
class ReplicaState:
    """State information for a replica group."""
    replica_id: str
    current_step: int = 0
    step_progress: float = 0.0
    status: str = "training"  # "training", "paused", "offline", "deactivating", "activating"
    last_active_step: int = 0
    fixed_step: Optional[int] = None
    last_updated: float = field(default_factory=time.time)


@dataclass 
class ServerInfo:
    """Information about a server."""
    server_id: str
    replica_group_id: str
    gpu_ids: List[str] = field(default_factory=list)


@dataclass
class NetworkState:
    """Network status information for a server."""
    server_id: str
    bandwidth_usage: float = 0.0  # Mbps
    latency: float = 0.0  # ms
    connection_status: str = "stable"  # "stable", "unstable", "disconnected"
    connected_peers: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


class StatusTracker:
    """Tracks the status of all weaver components."""
    
    def __init__(self):
        self.global_step = 0
        self.communication_status = "stable"
        
        # Weaver state storage
        self.gpus: Dict[str, GPUState] = {}
        self.replicas: Dict[str, ReplicaState] = {}
        self.servers: Dict[str, ServerInfo] = {}
        self.networks: Dict[str, NetworkState] = {}
        
        logger.info("StatusTracker initialized")

    def update_gpu_status(self, gpu_id: str, replica_id: str, server_id: str, 
                         status: Optional[str] = None, utilization: Optional[float] = None, 
                         temperature: Optional[float] = None, memory_used: Optional[float] = None,
                         memory_total: Optional[float] = None, config: Optional[Dict[str, str]] = None):
        """Update GPU status information."""
        if gpu_id not in self.gpus:
            self.gpus[gpu_id] = GPUState(
                gpu_id=gpu_id,
                replica_id=replica_id,
                server_id=server_id
            )
        
        gpu = self.gpus[gpu_id]
        if status is not None:
            gpu.status = status
        if utilization is not None:
            gpu.utilization = utilization
        if temperature is not None:
            gpu.temperature = temperature
        if memory_used is not None:
            gpu.memory_used = memory_used
        if memory_total is not None:
            gpu.memory_total = memory_total
        if config is not None:
            gpu.config.update(config)
        
        gpu.last_updated = time.time()
        
        # Update server info
        if server_id not in self.servers:
            # Extract replica group ID from replica_id (assuming format like "demo_replica_1")
            replica_group_id = replica_id.split('_')[0] if '_' in replica_id else replica_id
            self.servers[server_id] = ServerInfo(
                server_id=server_id,
                replica_group_id=replica_group_id
            )
        
        if gpu_id not in self.servers[server_id].gpu_ids:
            self.servers[server_id].gpu_ids.append(gpu_id)
        
        logger.debug(f"Updated GPU {gpu_id}: status={status}, util={utilization}%, temp={temperature}°C")

    def update_training_progress(self, replica_id: str, current_step: Optional[int] = None,
                               step_progress: Optional[float] = None, status: Optional[str] = None,
                               last_active_step: Optional[int] = None, fixed_step: Optional[int] = None):
        """Update training progress for a replica."""
        if replica_id not in self.replicas:
            self.replicas[replica_id] = ReplicaState(replica_id=replica_id)
        
        replica = self.replicas[replica_id]
        if current_step is not None:
            replica.current_step = current_step
        if step_progress is not None:
            replica.step_progress = step_progress
        if status is not None:
            replica.status = status
        if last_active_step is not None:
            replica.last_active_step = last_active_step
        if fixed_step is not None:
            replica.fixed_step = fixed_step
        
        replica.last_updated = time.time()
        
        logger.debug(f"Updated replica {replica_id}: step={current_step}, status={status}")

    def update_network_status(self, server_id: str, bandwidth_usage: Optional[float] = None,
                            latency: Optional[float] = None, connection_status: Optional[str] = None,
                            connected_peers: Optional[List[str]] = None):
        """Update network status for a server."""
        if server_id not in self.networks:
            self.networks[server_id] = NetworkState(server_id=server_id)
        
        network = self.networks[server_id]
        if bandwidth_usage is not None:
            network.bandwidth_usage = bandwidth_usage
        if latency is not None:
            network.latency = latency
        if connection_status is not None:
            network.connection_status = connection_status
        if connected_peers is not None:
            network.connected_peers = connected_peers.copy()
        
        network.last_updated = time.time()
        
        logger.debug(f"Updated network {server_id}: bandwidth={bandwidth_usage}Mbps, latency={latency}ms")

    def set_global_step(self, step: int):
        """Update the global training step."""
        self.global_step = step

    def set_communication_status(self, status: str):
        """Update communication status."""
        self.communication_status = status
        logger.info(f"Communication status: {status}")

    def deactivate_gpu(self, gpu_id: str):
        """Mark a GPU as offline."""
        if gpu_id in self.gpus:
            self.gpus[gpu_id].status = "offline"
            self.gpus[gpu_id].utilization = 0.0
            self.gpus[gpu_id].temperature = 40.0
            
            # Update replica status if this was the last active GPU
            replica_id = self.gpus[gpu_id].replica_id
            active_gpus = [g for g in self.gpus.values() 
                          if g.replica_id == replica_id and g.status == "active"]
            
            if not active_gpus:
                self.update_training_progress(
                    replica_id, 
                    status="offline",
                    fixed_step=self.global_step
                )
            
            logger.info(f"Deactivated GPU {gpu_id}")

    def reactivate_replica_group(self, replica_id: str):
        """Reactivate all GPUs in a replica group."""
        replica_gpus = [g for g in self.gpus.values() if g.replica_id == replica_id]
        
        for gpu in replica_gpus:
            if gpu.status == "offline":
                gpu.status = "active"
                gpu.utilization = 50.0 + (hash(gpu.gpu_id) % 30)  # 50-80%
                gpu.temperature = 50.0 + (hash(gpu.gpu_id) % 20)  # 50-70°C
        
        self.update_training_progress(replica_id, status="training")
        logger.info(f"Reactivated replica group {replica_id}")

    def get_active_gpus(self) -> List[GPUState]:
        """Get all active GPUs."""
        return [gpu for gpu in self.gpus.values() if gpu.status == "active"]

    def get_active_replicas(self) -> List[ReplicaState]:
        """Get all replicas that are currently training."""
        return [replica for replica in self.replicas.values() if replica.status == "training"]

    def get_replica_gpus(self, replica_id: str) -> List[GPUState]:
        """Get all GPUs associated with a replica."""
        return [gpu for gpu in self.gpus.values() if gpu.replica_id == replica_id]

    def get_server_gpus(self, server_id: str) -> List[GPUState]:
        """Get all GPUs on a specific server."""
        return [gpu for gpu in self.gpus.values() if gpu.server_id == server_id]

    def get_system_summary(self) -> Dict[str, Any]:
        """Get a summary of the current weaver state."""
        active_gpus = len([gpu for gpu in self.gpus.values() if gpu.status == "active"])
        total_gpus = len(self.gpus)
        active_replicas = len([replica for replica in self.replicas.values() if replica.status == "training"])
        total_replicas = len(self.replicas)
        
        return {
            "global_step": self.global_step,
            "communication_status": self.communication_status,
            "active_gpus": active_gpus,
            "total_gpus": total_gpus,
            "active_replicas": active_replicas,
            "total_replicas": total_replicas,
            "servers": len(self.servers),
            "networks": len(self.networks)
        }

    def simulate_training_progress(self):
        """Simulate training progress for demo purposes."""
        self.global_step += 1
        
        # Update active replicas
        for replica in self.replicas.values():
            if replica.status == "training":
                replica.current_step = self.global_step
                replica.step_progress = (self.global_step % 100)
                replica.last_active_step = self.global_step
        
        # Update GPU utilization with some variance
        for gpu in self.gpus.values():
            if gpu.status == "active":
                # Add some realistic variance
                base_util = 70.0
                variance = 10.0 * (0.5 - hash(gpu.gpu_id + str(self.global_step)) % 100 / 100.0)
                gpu.utilization = max(50.0, min(95.0, base_util + variance))
                
                # Temperature correlation with utilization
                gpu.temperature = 45.0 + (gpu.utilization * 0.3)
                
                # Update memory usage
                gpu.memory_used = 1.0 + (gpu.utilization * 0.08)  # 1-8.6 GB based on utilization
                if gpu.memory_total == 0.0:
                    gpu.memory_total = 8.0  # Default 8GB

    def cleanup_stale_entries(self, max_age_seconds: float = 300):
        """Remove entries that haven't been updated recently."""
        current_time = time.time()
        
        stale_gpus = [
            gpu_id for gpu_id, gpu in self.gpus.items()
            if current_time - gpu.last_updated > max_age_seconds
        ]
        
        for gpu_id in stale_gpus:
            logger.info(f"Removing stale GPU entry: {gpu_id}")
            del self.gpus[gpu_id]
        
        stale_replicas = [
            replica_id for replica_id, replica in self.replicas.items()
            if current_time - replica.last_updated > max_age_seconds
        ]
        
        for replica_id in stale_replicas:
            logger.info(f"Removing stale replica entry: {replica_id}")
            del self.replicas[replica_id]
        
        stale_networks = [
            server_id for server_id, network in self.networks.items()
            if current_time - network.last_updated > max_age_seconds
        ]
        
        for server_id in stale_networks:
            logger.info(f"Removing stale network entry: {server_id}")
            del self.networks[server_id] 