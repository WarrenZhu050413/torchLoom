"""
UI-specific message handlers for the torchLoom Weaver.

This module contains handlers for UI-related messages including status updates,
training progress, and system topology information.
"""

import logging
from typing import Dict, Optional
from abc import ABC, abstractmethod

from torchLoom.proto.torchLoom_pb2 import EventEnvelope, GPUStatus, TrainingProgress, UICommand
from torchLoom.log.logger import setup_logger
from .handlers import MessageHandler

logger = setup_logger(name="ui_handlers")


class StatusUpdateHandler(MessageHandler):
    """Handler for GPU status update messages."""
    
    def __init__(self, status_tracker):
        self.status_tracker = status_tracker
    
    async def handle(self, env: EventEnvelope) -> None:
        """Handle GPU status update events."""
        if not env.HasField("gpu_status"):
            return
            
        gpu_status: GPUStatus = env.gpu_status
        
        # Convert protobuf config map to dict
        config_dict = dict(gpu_status.config) if gpu_status.config else {}
        
        # Update status tracker
        self.status_tracker.update_gpu_status(
            gpu_id=gpu_status.gpu_id,
            replica_id=gpu_status.replica_id,
            server_id=gpu_status.server_id,
            status=gpu_status.status,
            utilization=gpu_status.utilization,
            temperature=gpu_status.temperature,
            config=config_dict
        )
        
        logger.debug(f"Updated GPU status: {gpu_status.gpu_id} - {gpu_status.status}")


class ProgressUpdateHandler(MessageHandler):
    """Handler for training progress update messages."""
    
    def __init__(self, status_tracker):
        self.status_tracker = status_tracker
    
    async def handle(self, env: EventEnvelope) -> None:
        """Handle training progress update events."""
        if not env.HasField("training_progress"):
            return
            
        progress: TrainingProgress = env.training_progress
        
        # Update status tracker
        self.status_tracker.update_training_progress(
            replica_id=progress.replica_id,
            current_step=progress.current_step,
            step_progress=progress.step_progress,
            status=progress.status,
            last_active_step=progress.last_active_step,
            fixed_step=progress.fixed_step if progress.fixed_step > 0 else None
        )
        
        logger.debug(f"Updated training progress: {progress.replica_id} - step {progress.current_step}")


class UICommandHandler(MessageHandler):
    """Handler for UI command messages."""
    
    def __init__(self, status_tracker, nats_client=None):
        self.status_tracker = status_tracker
        self.nats_client = nats_client
    
    async def handle(self, env: EventEnvelope) -> None:
        """Handle UI command events."""
        if not env.HasField("ui_command"):
            return
            
        command: UICommand = env.ui_command
        command_type = command.command_type
        target_id = command.target_id
        params = dict(command.params) if command.params else {}
        
        logger.info(f"Processing UI command: {command_type} for {target_id}")
        
        try:
            if command_type == "deactivate_gpu":
                await self._handle_deactivate_gpu(target_id)
            elif command_type == "reactivate_group":
                await self._handle_reactivate_group(target_id)
            elif command_type == "update_config":
                await self._handle_update_config(target_id, params)
            else:
                logger.warning(f"Unknown UI command type: {command_type}")
                
        except Exception as e:
            logger.exception(f"Error processing UI command {command_type}: {e}")
    
    async def _handle_deactivate_gpu(self, gpu_id: str):
        """Handle GPU deactivation command."""
        if gpu_id in self.status_tracker.gpus:
            self.status_tracker.set_communication_status("rebuilding")
            
            # Get replica ID for status update
            replica_id = self.status_tracker.gpus[gpu_id].replica_id
            self.status_tracker.update_training_progress(replica_id, status="deactivating")
            
            # Deactivate the GPU
            self.status_tracker.deactivate_gpu(gpu_id)
            
            logger.info(f"Deactivated GPU: {gpu_id}")
        else:
            logger.warning(f"GPU not found for deactivation: {gpu_id}")
    
    async def _handle_reactivate_group(self, replica_id: str):
        """Handle replica group reactivation command."""
        self.status_tracker.set_communication_status("rebuilding")
        self.status_tracker.update_training_progress(replica_id, status="activating")
        
        # Reactivate the replica group
        self.status_tracker.reactivate_replica_group(replica_id)
        
        logger.info(f"Reactivated replica group: {replica_id}")
    
    async def _handle_update_config(self, replica_id: str, params: Dict[str, str]):
        """Handle configuration update command."""
        # Update configuration for all GPUs in the replica
        replica_gpus = [g for g in self.status_tracker.gpus.values() if g.replica_id == replica_id]
        
        for gpu in replica_gpus:
            gpu.config.update(params)
        
        logger.info(f"Updated config for replica {replica_id}: {params}")


class TopologyUpdateHandler(MessageHandler):
    """Handler for system topology update messages."""
    
    def __init__(self, status_tracker):
        self.status_tracker = status_tracker
    
    async def handle(self, env: EventEnvelope) -> None:
        """Handle system topology update events."""
        if not env.HasField("system_topology"):
            return
            
        topology = env.system_topology
        
        # Update server information in status tracker
        server_id = topology.server_id
        replica_group_id = topology.replica_group_id
        gpu_ids = list(topology.gpu_ids)
        
        if server_id not in self.status_tracker.servers:
            from torchLoom.weaver.status_tracker import ServerInfo
            self.status_tracker.servers[server_id] = ServerInfo(
                server_id=server_id,
                replica_group_id=replica_group_id,
                gpu_ids=gpu_ids
            )
        else:
            # Update existing server info
            server_info = self.status_tracker.servers[server_id]
            server_info.replica_group_id = replica_group_id
            server_info.gpu_ids = gpu_ids
        
        logger.debug(f"Updated topology: server {server_id} with {len(gpu_ids)} GPUs")


class UIStatusPublisher:
    """Publishes aggregated status updates to the UI."""
    
    def __init__(self, status_tracker, websocket_server=None):
        self.status_tracker = status_tracker
        self.websocket_server = websocket_server
    
    async def publish_status_update(self):
        """Publish complete status update to UI."""
        if self.websocket_server:
            await self.websocket_server.broadcast_status_update()
    
    def get_system_summary(self) -> Dict:
        """Get a summary of the current system state."""
        active_gpus = sum(1 for gpu in self.status_tracker.gpus.values() if gpu.status == "active")
        total_gpus = len(self.status_tracker.gpus)
        active_replicas = sum(1 for replica in self.status_tracker.replicas.values() if replica.status == "training")
        total_replicas = len(self.status_tracker.replicas)
        
        return {
            "global_step": self.status_tracker.global_step,
            "communication_status": self.status_tracker.communication_status,
            "active_gpus": active_gpus,
            "total_gpus": total_gpus,
            "active_replicas": active_replicas,
            "total_replicas": total_replicas,
            "servers": len(self.status_tracker.servers)
        }


class DemoDataSimulator:
    """Simulates training data for demo purposes."""
    
    def __init__(self, status_tracker):
        self.status_tracker = status_tracker
        self.demo_replicas = [
            "demo_replica_1",
            "demo_replica_2", 
            "demo_replica_3"
        ]
        self.demo_servers = [
            "server-1-0",
            "server-1-1", 
            "server-2-0",
            "server-3-0"
        ]
        
        logger.info("Demo data simulator initialized")
    
    def initialize_demo_data(self):
        """Initialize demo GPUs and replicas."""
        gpu_counter = 0
        
        for i, replica_id in enumerate(self.demo_replicas):
            # Create 2-4 GPUs per replica
            gpu_count = 2 + (i % 3)  # 2, 3, or 4 GPUs
            server_id = self.demo_servers[i % len(self.demo_servers)]
            
            for j in range(gpu_count):
                gpu_id = f"gpu-{gpu_counter}"
                gpu_counter += 1
                
                # Initialize GPU with demo data
                self.status_tracker.update_gpu_status(
                    gpu_id=gpu_id,
                    replica_id=replica_id,
                    server_id=server_id,
                    status="active",
                    utilization=60.0 + (gpu_counter % 30),  # 60-90%
                    temperature=50.0 + (gpu_counter % 25),  # 50-75°C
                    config={
                        "batch_size": "32",
                        "learning_rate": "0.001", 
                        "optimizer_type": "Adam"
                    }
                )
            
            # Initialize replica progress
            self.status_tracker.update_training_progress(
                replica_id=replica_id,
                current_step=0,
                step_progress=0.0,
                status="training"
            )
        
        logger.info(f"Initialized demo data: {gpu_counter} GPUs across {len(self.demo_replicas)} replicas")
    
    def simulate_training_step(self):
        """Simulate one training step across all demo components."""
        self.status_tracker.simulate_training_progress()
        logger.debug(f"Simulated training step: {self.status_tracker.global_step}") 