"""
Refactored torchLoom Weaver implementation.

This module contains the main Weaver class that orchestrates message handling
and subscription management using the extracted components.
"""

import asyncio
from typing import Dict, Set

from nats.aio.msg import Msg
from torchLoom.config import Config
from torchLoom.constants import torchLoomConstants
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope
from .handlers import (
    DeviceRegistrationHandler,
    FailureHandler,
    ConfigurationHandler,
    DeviceReplicaMapper,
    MessageHandler,
)
from .subscription import (
    SubscriptionManager,
    ConnectionManager,
)

logger = setup_logger(name="torchLoom_weaver", log_file=Config.torchLoom_CONTROLLER_LOG_FILE)


class Weaver:
    """Weaver for torchLoom.

    This class is responsible for managing the training process, including
    handling events and managing resources. It runs in a separate thread to
    avoid blocking the main thread.

    It is the sole producer of events.
    """

    def __init__(self, torchLoom_addr: str = torchLoomConstants.DEFAULT_ADDR) -> None:
        self._stop_nats = asyncio.Event()
        self.seq = 0
        
        # Connection management
        self._connection_manager = ConnectionManager(torchLoom_addr)
        self._subscription_manager = None
        
        # Device-replica mapping
        self._device_mapper = DeviceReplicaMapper()
        
        # Message handlers
        self._handlers: Dict[str, MessageHandler] = {}
        
        logger.info(f"Weaver initialized with NATS address: {torchLoom_addr}")

    async def initialize(self) -> None:
        """Initialize NATS connection and JetStream."""
        nc, js = await self._connection_manager.initialize()
        
        # Initialize subscription manager
        self._subscription_manager = SubscriptionManager(nc, js, self._stop_nats)
        
        # Initialize message handlers
        self._handlers = {
            'register_device': DeviceRegistrationHandler(self._device_mapper),
            'failure': FailureHandler(self._device_mapper, nc),
            'configuration': ConfigurationHandler(nc),
        }

    async def message_handler(self, msg: Msg) -> None:
        """Main message handler that dispatches to specific handlers."""
        try:
            env = EventEnvelope()
            env.ParseFromString(msg.data)
            logger.debug(f"Received message: {env}")
            
            # Handle different event types using dedicated handlers
            if env.HasField("register_device"):
                await self._handlers['register_device'].handle(env)
            
            if env.HasField("monitored_fail"):
                await self._handlers['failure'].handle(env)
            
            if env.HasField("config_info"):
                await self._handlers['configuration'].handle(env)

        except Exception as e:
            logger.exception(f"Error handling message: {e}")

    def get_replicas_for_device(self, device_uuid: str) -> Set[str]:
        """Get all replicas associated with a device."""
        return self._device_mapper.get_replicas_for_device(device_uuid)
    
    def get_devices_for_replica(self, replica_id: str) -> Set[str]:
        """Get all devices associated with a replica."""
        return self._device_mapper.get_devices_for_replica(replica_id)

    async def stop(self) -> None:
        """Stop the weaver and clean up resources."""
        logger.info("Stopping Weaver")
        
        # Signal all loops to exit
        self._stop_nats.set()

        # Stop all subscriptions
        if self._subscription_manager:
            await self._subscription_manager.stop_all_subscriptions()

        # Close connection
        await self._connection_manager.close()

    async def subscribe_js(self, stream: str, subject: str, consumer: str, message_handler) -> None:
        """Subscribe to a JetStream subject."""
        if not self._subscription_manager:
            raise RuntimeError("Weaver not initialized. Call initialize() first.")
        await self._subscription_manager.subscribe_js(stream, subject, consumer, message_handler)

    async def subscribe_nc(self, subject: str, message_handler) -> None:
        """Subscribe to a regular NATS subject."""
        if not self._subscription_manager:
            raise RuntimeError("Weaver not initialized. Call initialize() first.")
        await self._subscription_manager.subscribe_nc(subject, message_handler)

    @property
    def device_to_replicas(self) -> Dict[str, Set[str]]:
        """Get device to replicas mapping."""
        return self._device_mapper.device_to_replicas

    @property
    def replica_to_devices(self) -> Dict[str, Set[str]]:
        """Get replica to devices mapping."""
        return self._device_mapper.replica_to_devices


async def main():
    """Main function to start the weaver."""
    try:
        logger.info("Starting torchLoom Weaver")
        weaver = Weaver()
        await weaver.initialize()
        logger.info("Weaver initialized")

        async with asyncio.TaskGroup() as tg:
            tg.create_task(weaver.subscribe_js(
                torchLoomConstants.weaver_stream.STREAM,
                torchLoomConstants.weaver_stream.subjects.DR_SUBJECT,
                torchLoomConstants.weaver_stream.CONSUMER,
                weaver.message_handler
            ))
            tg.create_task(weaver.subscribe_nc(
                subject=torchLoomConstants.subjects.MONITOR, 
                message_handler=weaver.message_handler
            ))
            tg.create_task(weaver.subscribe_nc(
                subject=torchLoomConstants.subjects.CONFIG_INFO,
                message_handler=weaver.message_handler
            ))
            logger.info("Started subscribing to all subjects")

        logger.info("Subscribed to all subjects")

        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Weaver stopped by user")
    except Exception as e:
        logger.exception(f"Unexpected error in weaver: {e}")
    finally:
        await weaver.stop()


if __name__ == '__main__':
    asyncio.run(main()) 