"""
Inbound message handlers for the torchLoom Weaver.

This module contains handlers for processing messages sent TO the weaver from different sources:
- Weavelet handlers: Process updates from weavelets/training processes
- External handlers: Process messages from external monitoring systems  
- UI handlers: Process commands from the UI
"""

import logging
from typing import Dict, Set
import time
from abc import ABC, abstractmethod

from torchLoom.proto.torchLoom_pb2 import EventEnvelope, MonitoredFailEvent
from torchLoom.constants import torchLoomConstants
from torchLoom.log.logger import setup_logger

logger = setup_logger(name="inbound_handlers")


class MessageHandler(ABC):
    """Abstract base class for message handlers."""
    
    @abstractmethod
    async def handle(self, env: EventEnvelope) -> None:
        """Handle a specific type of message."""
        pass


# ===========================================
# WEAVELET HANDLERS (Training Process -> Weaver)
# ===========================================

class DeviceRegistrationHandler(MessageHandler):
    """Handler for device registration messages from weavelets."""
    
    def __init__(self, device_mapper: 'DeviceReplicaMapper'):
        self.device_mapper = device_mapper
    
    async def handle(self, env: EventEnvelope) -> None:
        """Handle device registration events from weavelets."""
        device_uuid: str = env.register_device.device_uuid
        replica_id: str = env.register_device.replica_id

        logger.info("\n" + "-" * 100)
        logger.info(f"[WEAVELET->WEAVER] Device registration: {device_uuid} -> {replica_id}")
        
        # Update mappings using the device mapper
        device_added = self.device_mapper.add_device_replica_mapping(device_uuid, replica_id)
        replica_added = self.device_mapper.add_replica_device_mapping(replica_id, device_uuid)
        
        if device_added:
            logger.info(f"New device mapping: {device_uuid} -> {self.device_mapper.get_replicas_for_device(device_uuid)}")
        
        if replica_added:
            logger.info(f"New replica mapping: {replica_id} -> {self.device_mapper.get_devices_for_replica(replica_id)}")


class HeartbeatHandler(MessageHandler):
    """Handler for heartbeat messages from weavelets to track process liveness."""
    
    def __init__(self, status_tracker, nats_client=None, heartbeat_timeout: float = 90.0):
        self.status_tracker = status_tracker
        self.nats_client = nats_client
        self.heartbeat_timeout = heartbeat_timeout  # 90 seconds timeout (3x heartbeat interval)
        self._last_heartbeats: Dict[str, float] = {}  # replica_id -> last_heartbeat_timestamp
        self._dead_replicas: Set[str] = set()  # Track replicas that are considered dead
        
    async def handle(self, env: EventEnvelope) -> None:
        """Handle heartbeat events from weavelets."""
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
            logger.info(f"[WEAVELET->WEAVER] Replica {replica_id} is alive again (received heartbeat)")
            
            # Update status tracker to reflect replica is alive
            if hasattr(self.status_tracker, 'update_replica_status'):
                self.status_tracker.update_replica_status(replica_id, "active")
        
        logger.debug(f"[WEAVELET->WEAVER] Heartbeat from {replica_id}, status: {heartbeat.status}")


class TrainingStatusHandler(MessageHandler):
    """Handler for training status messages from weavelets."""
    
    def __init__(self, status_tracker):
        self.status_tracker = status_tracker
    
    async def handle(self, env: EventEnvelope) -> None:
        """Handle training status events from weavelets."""
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
        
        logger.debug(f"[WEAVELET->WEAVER] Training status: {training_status.replica_id} - {training_status.status_type}")


class GPUStatusHandler(MessageHandler):
    """Handler for GPU status messages from weavelets."""
    
    def __init__(self, status_tracker):
        self.status_tracker = status_tracker
    
    async def handle(self, env: EventEnvelope) -> None:
        """Handle GPU status events from weavelets."""
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
        
        logger.debug(f"[WEAVELET->WEAVER] GPU status: {gpu_status.gpu_id} - {gpu_status.status}")


# ===========================================
# EXTERNAL HANDLERS (External Systems -> Weaver)
# ===========================================

class FailureHandler(MessageHandler):
    """Handler for failure messages from external monitoring systems."""
    
    def __init__(self, device_mapper: 'DeviceReplicaMapper', nats_client):
        self.device_mapper = device_mapper
        self.nats_client = nats_client
    
    async def handle(self, env: EventEnvelope) -> None:
        """Handle GPU failure events from external monitoring systems."""
        try:
            fail_event: MonitoredFailEvent = env.monitored_fail
            device_uuid: str = fail_event.device_uuid
            
            replica_ids: Set[str] = self.device_mapper.get_replicas_for_device(device_uuid)
            if replica_ids:
                logger.info(f"[EXTERNAL->WEAVER] GPU failure detected: {device_uuid}")
                logger.info(f"[EXTERNAL->WEAVER] Associated replicas: {replica_ids}")
                
                for replica_id in replica_ids:
                    await self.send_replica_fail_event(replica_id)
            else:
                logger.warning(f"[EXTERNAL->WEAVER] Device {device_uuid} not found in device-to-replicas map")
        except Exception as e:
            logger.exception(f"Error handling external failure message: {e}")

    async def send_replica_fail_event(self, replica_id: str) -> None:
        """Send a replica failure event to training processes."""
        if not self.nats_client:
            raise RuntimeError("NATS connection is not initialized")
        
        env: EventEnvelope = EventEnvelope()
        env.replica_fail.replica_id = replica_id
        await self.nats_client.publish(torchLoomConstants.subjects.REPLICA_FAIL, env.SerializeToString())
        logger.info(f"[WEAVER->WEAVELET] Published replica fail event for {replica_id}")


class NetworkStatusHandler(MessageHandler):
    """Handler for network status messages from external monitoring systems."""
    
    def __init__(self, status_tracker):
        self.status_tracker = status_tracker
    
    async def handle(self, env: EventEnvelope) -> None:
        """Handle network status events from external monitoring systems."""
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
        
        logger.debug(f"[EXTERNAL->WEAVER] Network status: server={network_status.server_id}, "
                    f"bandwidth={network_status.bandwidth_usage}Mbps, "
                    f"latency={network_status.latency}ms, "
                    f"status={network_status.connection_status}")


# ===========================================
# UI HANDLERS (UI -> Weaver)  
# ===========================================

class UICommandHandler(MessageHandler):
    """Handler for commands from the UI that need to be processed by the weaver."""
    
    def __init__(self, status_tracker, nats_client=None):
        self.status_tracker = status_tracker
        self.nats_client = nats_client
    
    async def handle(self, env: EventEnvelope) -> None:
        """Handle UI command events and execute corresponding weaver actions."""
        if not env.HasField("ui_command"):
            return
            
        ui_command = env.ui_command
        command_type = ui_command.command_type
        target_id = ui_command.target_id
        params = dict(ui_command.params) if ui_command.params else {}
        
        logger.info(f"[UI->WEAVER] Processing command: {command_type} for {target_id}")
        
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
                logger.warning(f"[UI->WEAVER] Unknown command type: {command_type}")
                
        except Exception as e:
            logger.exception(f"Error processing UI command {command_type}: {e}")
    
    async def _publish_weaver_command(self, command_type: str, target_replica_id: str, params: Dict[str, str] = None):
        """Publish a weaver command to training processes."""
        try:
            if not self.nats_client:
                logger.warning("Cannot publish weaver command - no NATS client")
                return
                
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
            
            logger.info(f"[WEAVER->WEAVELET] Published command: {command_type} to {target_replica_id}")
            
        except Exception as e:
            logger.exception(f"Failed to publish weaver command: {e}")
    
    async def _handle_deactivate_gpu(self, gpu_id: str):
        """Handle GPU deactivation command from UI."""
        if gpu_id in self.status_tracker.gpus:
            self.status_tracker.set_communication_status("rebuilding")
            
            # Get replica ID for the GPU
            replica_id = self.status_tracker.gpus[gpu_id].replica_id
            self.status_tracker.update_training_progress(replica_id, status="deactivating")
            
            # Deactivate the GPU
            self.status_tracker.deactivate_gpu(gpu_id)
            
            # Send pause command to the replica
            await self._publish_weaver_command("pause", replica_id)
            
            logger.info(f"[UI->WEAVER] Deactivated GPU: {gpu_id}")
        else:
            logger.warning(f"[UI->WEAVER] GPU not found for deactivation: {gpu_id}")
    
    async def _handle_reactivate_group(self, replica_id: str):
        """Handle replica group reactivation command from UI."""
        self.status_tracker.set_communication_status("rebuilding")
        self.status_tracker.update_training_progress(replica_id, status="activating")
        
        # Reactivate the replica group
        self.status_tracker.reactivate_replica_group(replica_id)
        
        # Send resume command to the replica
        await self._publish_weaver_command("resume", replica_id)
        
        logger.info(f"[UI->WEAVER] Reactivated replica group: {replica_id}")
    
    async def _handle_update_config(self, replica_id: str, params: Dict[str, str]):
        """Handle configuration update command from UI."""
        # Update configuration for all GPUs in the replica
        replica_gpus = [g for g in self.status_tracker.gpus.values() if g.replica_id == replica_id]
        
        for gpu in replica_gpus:
            gpu.config.update(params)
        
        # Send config update command to the replica
        await self._publish_weaver_command("update_config", replica_id, params)
        
        logger.info(f"[UI->WEAVER] Updated config for replica {replica_id}: {params}")
    
    async def _handle_pause_training(self, replica_id: str):
        """Handle pause training command from UI."""
        self.status_tracker.update_training_progress(replica_id, status="paused")
        await self._publish_weaver_command("pause", replica_id)
        logger.info(f"[UI->WEAVER] Paused training for replica: {replica_id}")
    
    async def _handle_resume_training(self, replica_id: str):
        """Handle resume training command from UI."""
        self.status_tracker.update_training_progress(replica_id, status="training")
        await self._publish_weaver_command("resume", replica_id)
        logger.info(f"[UI->WEAVER] Resumed training for replica: {replica_id}")


class ConfigurationHandler(MessageHandler):
    """Handler for general configuration change messages (legacy support)."""
    
    def __init__(self, nats_client):
        self.nats_client = nats_client
    
    async def handle(self, env: EventEnvelope) -> None:
        """Handle configuration change events."""
        if env.HasField("config_info"):
            await self._handle_config_info_change(env)
    
    async def _handle_config_info_change(self, env: EventEnvelope) -> None:
        """Handle config_info change events."""
        config_params: Dict[str, str] = dict(env.config_info.config_params)
        
        logger.info("\n" + "-" * 100)
        logger.info(f"[CONFIG] Received config change with parameters: {config_params}")
        
        try:
            if not self.nats_client:
                raise RuntimeError("NATS connection is not initialized.")
            
            js = self.nats_client.jetstream()
            
            # Publish the entire config change to a general subject
            logger.debug(f"Publishing config change to {torchLoomConstants.subjects.CONFIG_INFO}")
            await js.publish(
                torchLoomConstants.subjects.CONFIG_INFO, 
                env.SerializeToString()
            )
            logger.info(f"Published config changes to {torchLoomConstants.subjects.CONFIG_INFO}")
        except Exception as e:
            logger.exception(f"Failed to publish config changes: {e}")
            # Re-raise to ensure the callback wrapper can handle it
            raise


# ===========================================
# UTILITY CLASSES
# ===========================================

class DeviceReplicaMapper:
    """Manages mapping between devices and replicas."""
    
    def __init__(self):
        # Many-to-many mapping between devices and replicas
        self.device_to_replicas: Dict[str, Set[str]] = {}  # device_uuid -> set of replica_ids
        self.replica_to_devices: Dict[str, Set[str]] = {}  # replica_id -> set of device_uuids
    
    def add_device_replica_mapping(self, device_uuid: str, replica_id: str) -> bool:
        """Add a device-to-replica mapping. Returns True if this is a new association."""
        if device_uuid not in self.device_to_replicas:
            self.device_to_replicas[device_uuid] = set()
        
        is_new_association = replica_id not in self.device_to_replicas[device_uuid]
        if is_new_association:
            self.device_to_replicas[device_uuid].add(replica_id)
        
        return is_new_association
    
    def add_replica_device_mapping(self, replica_id: str, device_uuid: str) -> bool:
        """Add a replica-to-device mapping. Returns True if this is a new association."""
        if replica_id not in self.replica_to_devices:
            self.replica_to_devices[replica_id] = set()
        
        is_new_association = device_uuid not in self.replica_to_devices[replica_id]
        if is_new_association:
            self.replica_to_devices[replica_id].add(device_uuid)
        
        return is_new_association
    
    def get_replicas_for_device(self, device_uuid: str) -> Set[str]:
        """Get all replicas associated with a device."""
        return self.device_to_replicas.get(device_uuid, set())
    
    def get_devices_for_replica(self, replica_id: str) -> Set[str]:
        """Get all devices associated with a replica."""
        return self.replica_to_devices.get(replica_id, set()) 