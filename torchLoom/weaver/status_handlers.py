"""
Status message handlers for the torchLoom Weaver.

This module contains handlers for processing status messages from training processes,
external monitoring systems, and UI commands, then updating the weaver's StatusTracker.
"""

import logging
from typing import Dict, Optional, Set
import time

from torchLoom.proto.torchLoom_pb2 import EventEnvelope
from torchLoom.log.logger import setup_logger
from .handlers import MessageHandler

logger = setup_logger(name="status_handlers")


class HeartbeatHandler(MessageHandler):
    """Handler for Heartbeat messages from weavelets to track process liveness."""
    
    def __init__(self, status_tracker, nats_client=None, heartbeat_timeout: float = 90.0):
        self.status_tracker = status_tracker
        self.nats_client = nats_client
        self.heartbeat_timeout = heartbeat_timeout  # 90 seconds timeout (3x heartbeat interval)
        self._last_heartbeats: Dict[str, float] = {}  # replica_id -> last_heartbeat_timestamp
        self._dead_replicas: Set[str] = set()  # Track replicas that are considered dead
        
    async def handle(self, env) -> None:
        """Handle Heartbeat events."""
        if not env.HasField("heartbeat"):
            return
            
        heartbeat = env.heartbeat
        replica_id = heartbeat.replica_id
        current_time = time.time()
        
        # Update last heartbeat time
        self._last_heartbeats[replica_id] = current_time
        
        # If this replica was considered dead, mark it as alive again
        if replica_id in self._dead_replicas:
            self._dead_replicas.remove(replica_id)
            logger.info(f"Replica {replica_id} is alive again (received heartbeat)")
            
            # Update status tracker to reflect replica is alive
            if hasattr(self.status_tracker, 'update_replica_status'):
                self.status_tracker.update_replica_status(replica_id, "active")
        
        logger.debug(f"Received heartbeat from replica {replica_id}, status: {heartbeat.status}")
        
    def check_dead_replicas(self) -> Set[str]:
        """Check for replicas that haven't sent heartbeats and are considered dead."""
        current_time = time.time()
        newly_dead_replicas = set()
        
        for replica_id, last_heartbeat in self._last_heartbeats.items():
            if replica_id not in self._dead_replicas:
                time_since_heartbeat = current_time - last_heartbeat
                
                if time_since_heartbeat > self.heartbeat_timeout:
                    # Replica is considered dead
                    self._dead_replicas.add(replica_id)
                    newly_dead_replicas.add(replica_id)
                    logger.warning(
                        f"Replica {replica_id} is considered dead "
                        f"(no heartbeat for {time_since_heartbeat:.1f} seconds)"
                    )
                    
                    # Update status tracker to reflect replica is dead
                    if hasattr(self.status_tracker, 'update_replica_status'):
                        self.status_tracker.update_replica_status(replica_id, "dead")
        
        return newly_dead_replicas
    
    async def publish_replica_fail_events(self, dead_replicas: Set[str]) -> None:
        """Publish replica fail events for dead replicas."""
        if not self.nats_client or not dead_replicas:
            return
            
        try:
            from torchLoom.constants import torchLoomConstants
            
            for replica_id in dead_replicas:
                envelope = EventEnvelope()
                envelope.replica_fail.replica_id = replica_id
                
                await self.nats_client.publish(
                    torchLoomConstants.subjects.REPLICA_FAIL,
                    envelope.SerializeToString()
                )
                
                logger.info(f"Published replica fail event for dead replica: {replica_id}")
                
        except Exception as e:
            logger.exception(f"Failed to publish replica fail events: {e}")
    
    def get_live_replicas(self) -> Set[str]:
        """Get the set of currently live replicas."""
        return set(self._last_heartbeats.keys()) - self._dead_replicas
    
    def get_dead_replicas(self) -> Set[str]:
        """Get the set of currently dead replicas."""
        return self._dead_replicas.copy()


class TrainingStatusHandler(MessageHandler):
    """Handler for TrainingStatus messages from training processes."""
    
    def __init__(self, status_tracker):
        self.status_tracker = status_tracker
    
    async def handle(self, env) -> None:
        """Handle TrainingStatus events."""
        if not env.HasField("training_status"):
            return
            
        training_status = env.training_status
        
        # Update training progress in status tracker
        self.status_tracker.update_training_progress(
            replica_id=training_status.replica_id,
            current_step=training_status.current_step,
            step_progress=training_status.step_progress,
            status=training_status.status,
            last_active_step=training_status.batch_idx,
            fixed_step=None
        )
        
        # Update global step
        self.status_tracker.global_step = max(self.status_tracker.global_step, training_status.current_step)
        
        logger.debug(f"Updated training status: {training_status.replica_id} - {training_status.status_type}")


class GPUStatusHandler(MessageHandler):
    """Handler for GPUStatus messages from training processes and monitoring systems."""
    
    def __init__(self, status_tracker):
        self.status_tracker = status_tracker
    
    async def handle(self, env) -> None:
        """Handle GPUStatus events."""
        if not env.HasField("gpu_status"):
            return
            
        gpu_status = env.gpu_status
        
        # Convert protobuf config map to dict
        config_dict = dict(gpu_status.config) if gpu_status.config else {}
        
        # Update GPU status in status tracker
        self.status_tracker.update_gpu_status(
            gpu_id=gpu_status.gpu_id,
            replica_id=gpu_status.replica_id,
            server_id=gpu_status.server_id,
            status=gpu_status.status,
            utilization=gpu_status.utilization,
            temperature=gpu_status.temperature,
            memory_used=gpu_status.memory_used,
            memory_total=gpu_status.memory_total,
            config=config_dict
        )
        
        logger.debug(f"Updated GPU status: {gpu_status.gpu_id} - {gpu_status.status}")


class NetworkStatusHandler(MessageHandler):
    """Handler for NetworkStatus messages from external monitoring systems."""
    
    def __init__(self, status_tracker):
        self.status_tracker = status_tracker
    
    async def handle(self, env) -> None:
        """Handle NetworkStatus events."""
        if not env.HasField("network_status"):
            return
            
        network_status = env.network_status
        
        # Update network status in status tracker
        self.status_tracker.update_network_status(
            server_id=network_status.server_id,
            bandwidth_usage=network_status.bandwidth_usage,
            latency=network_status.latency,
            connection_status=network_status.connection_status,
            connected_peers=list(network_status.connected_peers)
        )
        
        logger.debug(f"Updated network status: server={network_status.server_id}, "
                    f"bandwidth={network_status.bandwidth_usage}Mbps, "
                    f"latency={network_status.latency}ms, "
                    f"status={network_status.connection_status}")


class UIUpdateHandler(MessageHandler):
    """Handler for publishing consolidated UI updates to the UI."""
    
    def __init__(self, status_tracker, nats_client=None):
        self.status_tracker = status_tracker
        self.nats_client = nats_client
    
    async def publish_ui_update(self) -> None:
        """Publish consolidated UIStatusUpdate to the UI."""
        try:
            if not self.nats_client:
                logger.warning("Cannot publish UI update - no NATS client")
                return
                
            from torchLoom.constants import torchLoomConstants
            from torchLoom.proto.torchLoom_pb2 import EventEnvelope
            
            # Create consolidated UIStatusUpdate
            envelope = EventEnvelope()
            ui_update = envelope.ui_status_update
            ui_update.global_step = self.status_tracker.global_step
            ui_update.communication_status = self.status_tracker.communication_status
            ui_update.timestamp = int(time.time())
            
            # Add all GPU statuses
            for gpu_info in self.status_tracker.gpus.values():
                gpu_status = ui_update.gpus.add()
                gpu_status.gpu_id = gpu_info.gpu_id
                gpu_status.replica_id = gpu_info.replica_id
                gpu_status.server_id = gpu_info.server_id
                gpu_status.status = gpu_info.status
                gpu_status.utilization = gpu_info.utilization
                gpu_status.temperature = gpu_info.temperature
                gpu_status.memory_used = gpu_info.memory_used
                gpu_status.memory_total = gpu_info.memory_total
                
                # Add config
                for key, value in gpu_info.config.items():
                    gpu_status.config[key] = str(value)
            
            # Add all training statuses
            for replica_info in self.status_tracker.replicas.values():
                training_status = ui_update.training_status.add()
                training_status.replica_id = replica_info.replica_id
                training_status.status_type = "training_update"
                training_status.current_step = replica_info.current_step
                training_status.step_progress = replica_info.step_progress
                training_status.status = replica_info.status
                training_status.batch_idx = replica_info.last_active_step
            
            # Add network statuses
            for network_info in self.status_tracker.networks.values():
                network_status = ui_update.network_status.add()
                network_status.server_id = network_info.server_id
                network_status.bandwidth_usage = network_info.bandwidth_usage
                network_status.latency = network_info.latency
                network_status.connection_status = network_info.connection_status
                for peer in network_info.connected_peers:
                    network_status.connected_peers.append(peer)
            
            # Add topology information
            for server_info in self.status_tracker.servers.values():
                topology = ui_update.topology.add()
                topology.server_id = server_info.server_id
                topology.replica_group_id = server_info.replica_group_id
                for gpu_id in server_info.gpu_ids:
                    topology.gpu_ids.append(gpu_id)
            
            # Publish to UI
            await self.nats_client.publish(
                torchLoomConstants.subjects.UI_UPDATE,
                envelope.SerializeToString()
            )
            
            logger.debug(f"Published UI update: global_step={ui_update.global_step}")
            
        except Exception as e:
            logger.exception(f"Failed to publish UI update: {e}")

    async def handle(self, env) -> None:
        """This handler doesn't process incoming messages, only publishes updates."""
        pass


class WeaverCommandHandler(MessageHandler):
    """Handler for converting UI commands to weaver commands and publishing them."""
    
    def __init__(self, status_tracker, nats_client=None):
        self.status_tracker = status_tracker
        self.nats_client = nats_client
    
    async def handle(self, env) -> None:
        """Handle UI command events and convert them to weaver commands."""
        if not env.HasField("ui_command"):
            return
            
        ui_command = env.ui_command
        command_type = ui_command.command_type
        target_id = ui_command.target_id
        params = dict(ui_command.params) if ui_command.params else {}
        
        logger.info(f"Processing UI command: {command_type} for {target_id}")
        
        try:
            if command_type == "deactivate_gpu":
                await self._handle_deactivate_gpu(target_id)
            elif command_type == "reactivate_group":
                await self._handle_reactivate_group(target_id)
            elif command_type == "update_config":
                await self._handle_update_config(target_id, params)
            elif command_type == "pause_training":
                await self._handle_pause_training(target_id)
            elif command_type == "resume_training":
                await self._handle_resume_training(target_id)
            else:
                logger.warning(f"Unknown UI command type: {command_type}")
                
        except Exception as e:
            logger.exception(f"Error processing UI command {command_type}: {e}")
    
    async def _publish_weaver_command(self, command_type: str, target_replica_id: str, params: Optional[Dict[str, str]] = None):
        """Publish a weaver command to training processes."""
        try:
            if not self.nats_client:
                logger.warning("Cannot publish weaver command - no NATS client")
                return
                
            from torchLoom.constants import torchLoomConstants
            from torchLoom.proto.torchLoom_pb2 import EventEnvelope
            
            envelope = EventEnvelope()
            weaver_command = envelope.weaver_command
            weaver_command.command_type = command_type
            weaver_command.target_replica_id = target_replica_id
            
            if params:
                for key, value in params.items():
                    weaver_command.params[key] = str(value)
            
            await self.nats_client.publish(
                torchLoomConstants.subjects.WEAVER_COMMANDS,
                envelope.SerializeToString()
            )
            
            logger.info(f"Published weaver command: {command_type} to {target_replica_id}")
            
        except Exception as e:
            logger.exception(f"Failed to publish weaver command: {e}")
    
    async def _handle_deactivate_gpu(self, gpu_id: str):
        """Handle GPU deactivation command."""
        if gpu_id in self.status_tracker.gpus:
            self.status_tracker.set_communication_status("rebuilding")
            
            # Get replica ID for the GPU
            replica_id = self.status_tracker.gpus[gpu_id].replica_id
            self.status_tracker.update_training_progress(replica_id, status="deactivating")
            
            # Deactivate the GPU
            self.status_tracker.deactivate_gpu(gpu_id)
            
            # Send pause command to the replica
            await self._publish_weaver_command("pause", replica_id)
            
            logger.info(f"Deactivated GPU: {gpu_id}")
        else:
            logger.warning(f"GPU not found for deactivation: {gpu_id}")
    
    async def _handle_reactivate_group(self, replica_id: str):
        """Handle replica group reactivation command."""
        self.status_tracker.set_communication_status("rebuilding")
        self.status_tracker.update_training_progress(replica_id, status="activating")
        
        # Reactivate the replica group
        self.status_tracker.reactivate_replica_group(replica_id)
        
        # Send resume command to the replica
        await self._publish_weaver_command("resume", replica_id)
        
        logger.info(f"Reactivated replica group: {replica_id}")
    
    async def _handle_update_config(self, replica_id: str, params: Dict[str, str]):
        """Handle configuration update command."""
        # Update configuration for all GPUs in the replica
        replica_gpus = [g for g in self.status_tracker.gpus.values() if g.replica_id == replica_id]
        
        for gpu in replica_gpus:
            gpu.config.update(params)
        
        # Send config update command to the replica
        await self._publish_weaver_command("update_config", replica_id, params)
        
        logger.info(f"Updated config for replica {replica_id}: {params}")
    
    async def _handle_pause_training(self, replica_id: str):
        """Handle pause training command."""
        self.status_tracker.update_training_progress(replica_id, status="paused")
        await self._publish_weaver_command("pause", replica_id)
        logger.info(f"Paused training for replica: {replica_id}")
    
    async def _handle_resume_training(self, replica_id: str):
        """Handle resume training command."""
        self.status_tracker.update_training_progress(replica_id, status="training")
        await self._publish_weaver_command("resume", replica_id)
        logger.info(f"Resumed training for replica: {replica_id}")


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
                    memory_used=2.0 + (gpu_counter % 6),   # 2-8GB
                    memory_total=8.0,
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