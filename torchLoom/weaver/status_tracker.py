"""
Status tracking for torchLoom Weaver.

This module manages the state of all replicas, GPUs, and training progress
to provide real-time updates to the UI.
"""

import asyncio
import time
from typing import Dict, Set, Optional, List
from dataclasses import dataclass, field
from torchLoom.log.logger import setup_logger
from torchLoom.constants import torchLoomConstants
from torchLoom.proto.torchLoom_pb2 import UIStatusUpdate, GPUStatus, TrainingProgress, SystemTopology

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


class StatusTracker:
    """Tracks the status of all system components for UI updates."""
    
    def __init__(self, nats_client=None):
        self.nats_client = nats_client
        self.global_step = 0
        self.communication_status = "stable"
        
        # State storage
        self.gpus: Dict[str, GPUState] = {}
        self.replicas: Dict[str, ReplicaState] = {}
        self.servers: Dict[str, ServerInfo] = {}
        
        # UI update tracking
        self.last_ui_update = 0
        self.update_interval = 1.0  # seconds
        
        logger.info("StatusTracker initialized")

    def update_gpu_status(self, gpu_id: str, replica_id: str, server_id: str, 
                         status: Optional[str] = None, utilization: Optional[float] = None, 
                         temperature: Optional[float] = None, config: Optional[Dict[str, str]] = None):
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

    def get_ui_status_update(self) -> UIStatusUpdate:
        """Generate a complete status update for the UI."""
        ui_update = UIStatusUpdate()
        ui_update.global_step = self.global_step
        ui_update.communication_status = self.communication_status
        
        # Add GPU statuses
        for gpu in self.gpus.values():
            gpu_status = GPUStatus()
            gpu_status.gpu_id = gpu.gpu_id
            gpu_status.replica_id = gpu.replica_id
            gpu_status.server_id = gpu.server_id
            gpu_status.status = gpu.status
            gpu_status.utilization = gpu.utilization
            gpu_status.temperature = gpu.temperature
            
            for key, value in gpu.config.items():
                gpu_status.config[key] = value
                
            ui_update.gpus.append(gpu_status)
        
        # Add training progress
        for replica in self.replicas.values():
            progress = TrainingProgress()
            progress.replica_id = replica.replica_id
            progress.current_step = replica.current_step
            progress.step_progress = replica.step_progress
            progress.status = replica.status
            progress.last_active_step = replica.last_active_step
            if replica.fixed_step is not None:
                progress.fixed_step = replica.fixed_step
                
            ui_update.training_progress.append(progress)
        
        # Add system topology
        for server in self.servers.values():
            topology = SystemTopology()
            topology.server_id = server.server_id
            topology.replica_group_id = server.replica_group_id
            topology.gpu_ids.extend(server.gpu_ids)
            
            ui_update.topology.append(topology)
        
        return ui_update

    async def start_periodic_updates(self):
        """Start periodic UI updates."""
        logger.info("Starting periodic UI updates")
        
        while True:
            try:
                if self.nats_client and time.time() - self.last_ui_update > self.update_interval:
                    await self.publish_ui_update()
                    self.last_ui_update = time.time()
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.exception(f"Error in periodic updates: {e}")
                await asyncio.sleep(1.0)

    async def publish_ui_update(self):
        """Publish status update to UI stream."""
        if not self.nats_client:
            return
            
        try:
            ui_update = self.get_ui_status_update()
            
            js = self.nats_client.jetstream()
            await js.publish(
                torchLoomConstants.subjects.UI_STATUS_UPDATE,
                ui_update.SerializeToString()
            )
            
            logger.debug(f"Published UI update: {len(self.gpus)} GPUs, {len(self.replicas)} replicas")
            
        except Exception as e:
            logger.exception(f"Failed to publish UI update: {e}")

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