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
    ConfigurationHandler,
    DeviceRegistrationHandler,
    DeviceReplicaMapper,
    FailureHandler,
    MessageHandler,
    TrainingStatusHandler,
    GPUStatusHandler,
    NetworkStatusHandler,
    UICommandHandler,
)
from .publishers import (
    UIUpdatePublisher,
    DemoDataSimulator,
)
from .status_tracker import StatusTracker
from .subscription import ConnectionManager, SubscriptionManager
from .websocket_server import WebSocketServer

logger = setup_logger(
    name="torchLoom_weaver", log_file=Config.torchLoom_CONTROLLER_LOG_FILE
)


class Weaver:
    """Weaver for torchLoom.

    This class is responsible for managing the training process, including
    handling events and managing resources. It runs in a separate thread to
    avoid blocking the main thread.

    It is the sole producer of events.
    """

    def __init__(
        self,
        torchLoom_addr: str = torchLoomConstants.DEFAULT_ADDR,
        enable_ui: bool = True,
        ui_host: str = "0.0.0.0",
        ui_port: int = 8080,
    ) -> None:
        self._stop_nats = asyncio.Event()
        self.seq = 0

        # Connection management
        self._connection_manager = ConnectionManager(torchLoom_addr)
        self._subscription_manager = None

        # Device-replica mapping
        self._device_mapper = DeviceReplicaMapper()

        # Status tracking and UI
        self.status_tracker = StatusTracker()
        self.websocket_server = None
        self.demo_simulator = None
        self.enable_ui = enable_ui
        self.ui_host = ui_host
        self.ui_port = ui_port

        # Message handlers
        self._handlers: Dict[str, MessageHandler] = {}

        logger.info(f"Weaver initialized with NATS address: {torchLoom_addr}")
        if enable_ui:
            logger.info(f"UI server will be started on {ui_host}:{ui_port}")

    async def initialize(self) -> None:
        """Initialize NATS connection and JetStream."""
        nc, js = await self._connection_manager.initialize()

        # Initialize subscription manager
        self._subscription_manager = SubscriptionManager(nc, js, self._stop_nats)

        # Initialize WebSocket server if UI is enabled
        if self.enable_ui:
            self.websocket_server = WebSocketServer(
                status_tracker=self.status_tracker,
                nats_client=nc,
                host=self.ui_host,
                port=self.ui_port,
            )

        # Initialize message handlers
        self._handlers = {
            "register_device": DeviceRegistrationHandler(self._device_mapper),
            "failure": FailureHandler(self._device_mapper, nc),
            "configuration": ConfigurationHandler(nc),
            "training_status": TrainingStatusHandler(self.status_tracker),
            "gpu_status": GPUStatusHandler(self.status_tracker),
            "network_status": NetworkStatusHandler(self.status_tracker),
            "ui_commands": UICommandHandler(self.status_tracker, nc),
        }

        # Initialize UI update handler (for publishing consolidated updates)
        self.ui_update_handler = UIUpdatePublisher(self.status_tracker, nc)

        # Initialize demo data simulator
        self.demo_simulator = DemoDataSimulator(self.status_tracker)
        self.demo_simulator.initialize_demo_data()

        logger.info("Weaver fully initialized with UI support")

    async def message_handler(self, msg: Msg) -> None:
        """Main message handler that dispatches to specific handlers."""
        try:
            env = EventEnvelope()
            env.ParseFromString(msg.data)
            logger.debug(f"Received message: {env}")

            # Handle different event types using dedicated handlers
            if env.HasField("register_device"):
                await self._handlers["register_device"].handle(env)

            if env.HasField("monitored_fail"):
                await self._handlers["failure"].handle(env)

            if env.HasField("config_info"):
                await self._handlers["configuration"].handle(env)

            # UI-related message handling
            if env.HasField("ui_command"):
                await self._handlers["ui_commands"].handle(env)

            # Training Process -> Weaver message handling
            if env.HasField("training_status"):
                await self._handlers["training_status"].handle(env)

            if env.HasField("gpu_status"):
                await self._handlers["gpu_status"].handle(env)

            # External Monitoring -> Weaver message handling  
            if env.HasField("network_status"):
                await self._handlers["network_status"].handle(env)

            # Legacy message handling (backward compatibility)
            if env.HasField("ui_status_update"):
                await self.ui_update_handler.handle(env)

        except Exception as e:
            logger.exception(f"Error handling message: {e}")

    def get_replicas_for_device(self, device_uuid: str) -> Set[str]:
        """Get all replicas associated with a device."""
        return self._device_mapper.get_replicas_for_device(device_uuid)

    def get_devices_for_replica(self, replica_id: str) -> Set[str]:
        """Get all devices associated with a replica."""
        return self._device_mapper.get_devices_for_replica(replica_id)

    async def start_ui_server(self) -> None:
        """Start the WebSocket UI server."""
        if self.websocket_server:
            logger.info("Starting WebSocket UI server")
            await self.websocket_server.run_with_status_broadcaster()

    async def start_ui_update_publisher(self) -> None:
        """Start the background task to publish UI updates periodically."""
        logger.info("Starting UI update publisher")

        while not self._stop_nats.is_set():
            try:
                # Publish UI update every 2 seconds
                if self.ui_update_handler:
                    await self.ui_update_handler.publish_ui_update()

                await asyncio.sleep(2.0)

            except Exception as e:
                logger.exception(f"Error in UI update publisher: {e}")
                await asyncio.sleep(2.0)

    async def start_demo_simulation(self) -> None:
        """Start demo training simulation."""
        logger.info("Starting demo training simulation")

        while not self._stop_nats.is_set():
            try:
                # Simulate training progress
                if self.demo_simulator:
                    self.demo_simulator.simulate_training_step()

                # Cleanup stale entries periodically
                if self.status_tracker.global_step % 100 == 0:
                    self.status_tracker.cleanup_stale_entries()

                await asyncio.sleep(1.5)  # Simulate training step every 1.5 seconds

            except Exception as e:
                logger.exception(f"Error in demo simulation: {e}")
                await asyncio.sleep(2.0)

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

    async def subscribe_js(
        self, stream: str, subject: str, consumer: str, message_handler
    ) -> None:
        """Subscribe to a JetStream subject."""
        if not self._subscription_manager:
            raise RuntimeError("Weaver not initialized. Call initialize() first.")
        await self._subscription_manager.subscribe_js(
            stream, subject, consumer, message_handler
        )

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
        logger.info("Starting torchLoom Weaver with UI integration")
        weaver = Weaver(enable_ui=True)
        await weaver.initialize()
        logger.info("Weaver initialized")

        # Start all services concurrently
        async with asyncio.TaskGroup() as tg:
            # NATS subscriptions
            tg.create_task(
                weaver.subscribe_js(
                    torchLoomConstants.weaver_stream.STREAM,
                    torchLoomConstants.weaver_stream.subjects.DR_SUBJECT,
                    torchLoomConstants.weaver_stream.CONSUMER,
                    weaver.message_handler,
                )
            )
            tg.create_task(
                weaver.subscribe_nc(
                    subject=torchLoomConstants.subjects.MONITOR,
                    message_handler=weaver.message_handler,
                )
            )
            tg.create_task(
                weaver.subscribe_nc(
                    subject=torchLoomConstants.subjects.CONFIG_INFO,
                    message_handler=weaver.message_handler,
                )
            )

            # Training Process -> Weaver subscriptions
            tg.create_task(
                weaver.subscribe_nc(
                    subject=torchLoomConstants.subjects.TRAINING_STATUS,
                    message_handler=weaver.message_handler,
                )
            )
            tg.create_task(
                weaver.subscribe_nc(
                    subject=torchLoomConstants.subjects.GPU_STATUS,
                    message_handler=weaver.message_handler,
                )
            )

            # External Monitoring -> Weaver subscriptions
            tg.create_task(
                weaver.subscribe_nc(
                    subject=torchLoomConstants.subjects.NETWORK_STATUS,
                    message_handler=weaver.message_handler,
                )
            )

            # UI -> Weaver subscriptions  
            tg.create_task(
                weaver.subscribe_nc(
                    subject=torchLoomConstants.subjects.UI_COMMANDS,
                    message_handler=weaver.message_handler,
                )
            )

            # UI services
            if weaver.enable_ui:
                tg.create_task(weaver.start_ui_server())

            # Demo simulation
            tg.create_task(weaver.start_demo_simulation())

            # UI update publisher
            tg.create_task(weaver.start_ui_update_publisher())

            logger.info("All services started successfully")

        logger.info("Subscribed to all subjects")

        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Weaver stopped by user")
    except Exception as e:
        logger.exception(f"Unexpected error in weaver: {e}")
    finally:
        await weaver.stop()


if __name__ == "__main__":
    asyncio.run(main())
