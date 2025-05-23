"""
Status tracking for torchLoom Weaver.

This module manages the state of all replicas, devices, and training progress
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
class deviceState:
    """State information for a single device."""
    device_id: str
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
    device_ids: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


class StatusTracker:
    """Tracks the status of all weaver components."""
    
    def __init__(self):
        self.communication_status = "stable"
        
        # Weaver state storage
        self.devices: Dict[str, deviceState] = {}
        self.replicas: Dict[str, ReplicaState] = {}
        self.servers: Dict[str, ServerInfo] = {}
        
        logger.info("StatusTracker initialized")

    def update_device_status(self, device_id: str, replica_id: str, server_id: str, 
                         status: Optional[str] = None, utilization: Optional[float] = None, 
                         temperature: Optional[float] = None, memory_used: Optional[float] = None,
                         memory_total: Optional[float] = None, config: Optional[Dict[str, str]] = None):
        """Update device status information."""
        if device_id not in self.devices:
            self.devices[device_id] = deviceState(
                device_id=device_id,
                replica_id=replica_id,
                server_id=server_id
            )
        
        device = self.devices[device_id]
        if status is not None:
            device.status = status
        if utilization is not None:
            device.utilization = utilization
        if temperature is not None:
            device.temperature = temperature
        if memory_used is not None:
            device.memory_used = memory_used
        if memory_total is not None:
            device.memory_total = memory_total
        if config is not None:
            device.config.update(config)
        
        device.last_updated = time.time()
        
        # Update server info
        if server_id not in self.servers:
            # Extract replica group ID from replica_id (assuming format like "demo_replica_1")
            replica_group_id = replica_id.split('_')[0] if '_' in replica_id else replica_id
            self.servers[server_id] = ServerInfo(
                server_id=server_id,
                replica_group_id=replica_group_id
            )
        
        if device_id not in self.servers[server_id].device_ids:
            self.servers[server_id].device_ids.append(device_id)
        
        # Update server timestamp
        self.servers[server_id].last_updated = time.time()
        
        logger.debug(f"Updated device {device_id}: status={status}, util={utilization}%, temp={temperature}°C")

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

    def set_communication_status(self, status: str):
        """Update communication status."""
        self.communication_status = status
        logger.info(f"Communication status: {status}")

    def deactivate_device(self, device_id: str):
        """Mark a device as offline."""
        if device_id in self.devices:
            self.devices[device_id].status = "offline"
            self.devices[device_id].utilization = 0.0
            self.devices[device_id].temperature = 40.0
            
            # Update replica status if this was the last active device
            replica_id = self.devices[device_id].replica_id
            active_devices = [g for g in self.devices.values() 
                          if g.replica_id == replica_id and g.status == "active"]
            
            if not active_devices:
                self.update_training_progress(
                    replica_id, 
                    status="offline"
                )
            
            logger.info(f"Deactivated device {device_id}")

    def reactivate_replica_group(self, replica_id: str):
        """Reactivate all devices in a replica group."""
        replica_devices = [g for g in self.devices.values() if g.replica_id == replica_id]
        
        for device in replica_devices:
            if device.status == "offline":
                device.status = "active"
                device.utilization = 50.0 + (hash(device.device_id) % 30)  # 50-80%
                device.temperature = 50.0 + (hash(device.device_id) % 20)  # 50-70°C
        
        self.update_training_progress(replica_id, status="training")
        logger.info(f"Reactivated replica group {replica_id}")

    def get_active_devices(self) -> List[deviceState]:
        """Get all active devices."""
        return [device for device in self.devices.values() if device.status == "active"]

    def get_active_replicas(self) -> List[ReplicaState]:
        """Get all replicas that are currently training."""
        return [replica for replica in self.replicas.values() if replica.status == "training"]

    def get_replica_devices(self, replica_id: str) -> List[deviceState]:
        """Get all devices associated with a replica."""
        return [device for device in self.devices.values() if device.replica_id == replica_id]

    def get_server_devices(self, server_id: str) -> List[deviceState]:
        """Get all devices on a specific server."""
        return [device for device in self.devices.values() if device.server_id == server_id]

    def get_system_summary(self) -> Dict[str, Any]:
        """Get a summary of the current weaver state."""
        active_devices = len([device for device in self.devices.values() if device.status == "active"])
        total_devices = len(self.devices)
        active_replicas = len([replica for replica in self.replicas.values() if replica.status == "training"])
        total_replicas = len(self.replicas)
        
        return {
            "communication_status": self.communication_status,
            "active_devices": active_devices,
            "total_devices": total_devices,
            "active_replicas": active_replicas,
            "total_replicas": total_replicas,
            "servers": len(self.servers),
        }

    def simulate_training_progress(self):
        """Simulate training progress for demo purposes."""
        
        # Update active replicas - each replica progresses independently
        for replica in self.replicas.values():
            if replica.status == "training":
                replica.current_step += 1
                replica.step_progress = (replica.current_step % 100)
                replica.last_active_step = replica.current_step
        
        # Update device utilization with some variance
        for device in self.devices.values():
            if device.status == "active":
                # Add some realistic variance
                base_util = 70.0
                variance = 10.0 * (0.5 - hash(device.device_id) % 100 / 100.0)
                device.utilization = max(50.0, min(95.0, base_util + variance))
                
                # Temperature correlation with utilization
                device.temperature = 45.0 + (device.utilization * 0.3)
                
                # Update memory usage
                device.memory_used = 1.0 + (device.utilization * 0.08)  # 1-8.6 GB based on utilization
                if device.memory_total == 0.0:
                    device.memory_total = 8.0  # Default 8GB

    def cleanup_stale_entries(self, max_age_seconds: float = 300):
        """Remove entries that haven't been updated recently."""
        current_time = time.time()
        
        stale_devices = [
            device_id for device_id, device in self.devices.items()
            if current_time - device.last_updated > max_age_seconds
        ]
        
        for device_id in stale_devices:
            logger.info(f"Removing stale device entry: {device_id}")
            del self.devices[device_id]
        
        stale_replicas = [
            replica_id for replica_id, replica in self.replicas.items()
            if current_time - replica.last_updated > max_age_seconds
        ]
        
        for replica_id in stale_replicas:
            logger.info(f"Removing stale replica entry: {replica_id}")
            del self.replicas[replica_id]
        
        stale_servers = [
            server_id for server_id, server in self.servers.items()
            if current_time - server.last_updated > max_age_seconds
        ]
        
        for server_id in stale_servers:
            logger.info(f"Removing stale server entry: {server_id}")
            del self.servers[server_id] 