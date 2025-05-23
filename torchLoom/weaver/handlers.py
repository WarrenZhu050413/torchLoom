"""
Message handlers for the torchLoom Weaver.

This module contains handlers for different types of messages received by the Weaver,
following the single responsibility principle from AGENTS.md.
"""

import logging
from typing import Dict, Set
from abc import ABC, abstractmethod

from torchLoom.proto.torchLoom_pb2 import EventEnvelope, MonitoredFailEvent
from torchLoom.constants import torchLoomConstants
from torchLoom.log.logger import setup_logger

logger = setup_logger(name="weaver_handlers")


class MessageHandler(ABC):
    """Abstract base class for message handlers."""
    
    @abstractmethod
    async def handle(self, env: EventEnvelope) -> None:
        """Handle a specific type of message."""
        pass


class DeviceRegistrationHandler(MessageHandler):
    """Handler for device registration messages."""
    
    def __init__(self, device_mapper: 'DeviceReplicaMapper'):
        self.device_mapper = device_mapper
    
    async def handle(self, env: EventEnvelope) -> None:
        """Handle device registration events."""
        device_uuid: str = env.register_device.device_uuid
        replica_id: str = env.register_device.replica_id

        logger.info("\n" + "-" * 100)
        logger.info(f"Received register_device event for device {device_uuid} and replica {replica_id}")
        
        # Update mappings using the device mapper
        device_added = self.device_mapper.add_device_replica_mapping(device_uuid, replica_id)
        replica_added = self.device_mapper.add_replica_device_mapping(replica_id, device_uuid)
        
        if device_added:
            logger.info(f"New Mapping: Device {device_uuid} is now associated with replicas: {self.device_mapper.get_replicas_for_device(device_uuid)}")
        
        if replica_added:
            logger.info(f"New Mapping: Replica {replica_id} is now associated with devices: {self.device_mapper.get_devices_for_replica(replica_id)}")


class FailureHandler(MessageHandler):
    """Handler for failure-related messages."""
    
    def __init__(self, device_mapper: 'DeviceReplicaMapper', nats_client):
        self.device_mapper = device_mapper
        self.nats_client = nats_client
    
    async def handle(self, env: EventEnvelope) -> None:
        """Handle GPU failure events."""
        try:
            fail_event: MonitoredFailEvent = env.monitored_fail
            device_uuid: str = fail_event.device_uuid
            
            replica_ids: Set[str] = self.device_mapper.get_replicas_for_device(device_uuid)
            if replica_ids:
                logger.info(f"[GPU FAILURE] Associated Replica IDs: {replica_ids}")
                
                for replica_id in replica_ids:
                    await self.send_replica_fail_event(replica_id)
            else:
                logger.warning(f"[GPU FAILURE] Device {device_uuid} not found in device-to-replicas map")
        except Exception as e:
            logger.exception(f"Error handling GPU failure message: {e}")

    async def send_replica_fail_event(self, replica_id: str) -> None:
        """Send a replica failure event."""
        if not self.nats_client:
            raise RuntimeError("NATS connection is not initialized")
        
        env: EventEnvelope = EventEnvelope()
        env.replica_fail.replica_id = replica_id
        await self.nats_client.publish(torchLoomConstants.subjects.REPLICA_FAIL, env.SerializeToString())
        logger.info(f"Published replica fail event for replica {replica_id}")


class ConfigurationHandler(MessageHandler):
    """Handler for configuration change messages."""
    
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
        logger.info(f"Received config_info event with parameters: {config_params}")
        
        try:
            if not self.nats_client:
                raise RuntimeError("NATS connection is not initialized.")
            
            js = self.nats_client.jetstream()
            
            # Handle learning rate change specifically for backward compatibility
            if "learning_rate" in config_params:
                lr = config_params["learning_rate"]
                await js.publish("torchLoom.training.reset_lr", str(lr).encode("utf-8"))
                logger.info(f"Published new learning rate {lr} to torchLoom.training.reset_lr")
            
            # Publish the entire config change to a general subject
            await js.publish(
                torchLoomConstants.subjects.CONFIG_INFO, 
                env.SerializeToString()
            )
            logger.info(f"Published config changes to {torchLoomConstants.subjects.CONFIG_INFO}")
        except Exception as e:
            logger.exception(f"Failed to publish config changes: {e}")


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