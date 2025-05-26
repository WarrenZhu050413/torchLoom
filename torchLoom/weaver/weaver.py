"""
Refactored torchLoom Weaver implementation.

This module contains the main Weaver class that orchestrates message handling
and subscription management using the extracted components.
"""

import asyncio
from typing import Dict, Optional, Set

from nats.aio.msg import Msg

from torchLoom.common.config import Config
from torchLoom.common.constants import torchLoomConstants
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

from .handlers import (
    DeviceReplicaMapper,
    ExternalHandler,
    MessageHandler,
    UIHandler,
    ThreadletHandler,
)
from .publishers import UIUpdatePublisher, ThreadletCommandPublisher
from .status_tracker import StatusTracker
from torchLoom.common.subscription import SubscriptionManager
from .websocket_server import WebSocketServer

logger = setup_logger(
    name="torchLoom_weaver", log_file=Config.torchLoom_CONTROLLER_LOG_FILE
)

from torchLoom.common.constants import TimeConstants

class Weaver:
    """Weaver for torchLoom.

    This class is responsible for managing the training process, including
    handling events and managing resources. It runs in a separate thread to
    avoid blocking the main thread.

    It is the sole producer of events and comes with comprehensive default handlers
    for all standard torchLoom message types.

    ## Default Handlers

    The Weaver includes the following default handlers out of the box:

    - **DeviceRegistrationHandler**: Handles device registration from threadlets
    - **HeartbeatHandler**: Monitors threadlet liveness via heartbeat messages
    - **TrainingStatusHandler**: Processes training progress updates from threadlets
    - **deviceStatusHandler**: Handles device status and utilization updates
    - **NetworkStatusHandler**: Processes network connectivity and performance data
    - **FailureHandler**: Manages device and replica failure scenarios
    - **DrainEventHandler**: Handles graceful device drain requests
    - **UICommandHandler**: Processes commands from the UI (pause/resume/config changes)
    - **WeaverCommandHandler**: Handles command acknowledgments from threadlets
    - **ConfigurationHandler**: Manages configuration change events

    ## Customizing Handlers

    Users can easily customize or replace any default handler:

    ```python
    # Create custom handler
    class MyCustomHeartbeatHandler(MessageHandler):
        async def handle(self, env: EventEnvelope) -> None:
            # Custom heartbeat logic
            pass

    # Initialize weaver with defaults
    weaver = Weaver()
    await weaver.initialize()

    # Override specific handler
    custom_handler = MyCustomHeartbeatHandler()
    weaver.override_handler("heartbeat", custom_handler)

    # Add completely new handler
    weaver.add_custom_handler("my_custom_event", my_custom_handler)

    # List all handlers
    handlers = weaver.list_handlers()
    print(handlers)  # Shows all registered handlers
    ```

    ## Zero-Configuration Usage

    For most use cases, no handler configuration is needed:

    ```python
    weaver = Weaver(enable_ui=True)
    await weaver.initialize()
    # All default handlers are ready to use!
    ```
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

        # Initialize SubscriptionManager directly
        self._subscription_manager = SubscriptionManager(
            torchLoom_addr=torchLoom_addr, 
            stop_event=self._stop_nats
        )

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
        """Initialize NATS connection and JetStream via SubscriptionManager."""
        # Initialize SubscriptionManager's connection
        await self._subscription_manager.initialize()

        # Centralized stream setup - create ALL streams here with complete subject lists
        await self._setup_all_streams()

        # Initialize WebSocket server if UI is enabled
        if self.enable_ui:
            self.websocket_server = WebSocketServer(
                status_tracker=self.status_tracker,
                weaver=self,
                # Get nc from SubscriptionManager
                nats_client=self._subscription_manager.nc,
                host=self.ui_host,
                port=self.ui_port,
            )

        # Initialize consolidated message handlers
        self._handlers = {
            "threadlet": ThreadletHandler(
                self._device_mapper, 
                self.status_tracker, 
                self._subscription_manager.nc # Get nc from SubscriptionManager
            ),
            "external": ExternalHandler(
                self._device_mapper, 
                self._subscription_manager.nc, # Get nc from SubscriptionManager
                self.status_tracker
            ),
            "ui": UIHandler(
                self.status_tracker, 
                self._subscription_manager.nc # Get nc from SubscriptionManager
            ),
        }

        # Initialize publishers
        self.ui_update_handler = UIUpdatePublisher(
            self.status_tracker, 
            self._subscription_manager.nc # Get nc from SubscriptionManager
        )
        self.threadlet_command_handler = ThreadletCommandPublisher(
            self._subscription_manager.nc, # Get nc from SubscriptionManager
            self._handlers["threadlet"]
        )

        logger.info("Weaver fully initialized with UI support")

    async def _setup_all_streams(self) -> None:
        """Centralized setup of all JetStream streams with complete subject configurations."""
        logger.info("Setting up all JetStream streams...")
        print("DEBUG: _setup_all_streams called")

        if not self._subscription_manager or not self._subscription_manager.js: # Check js for stream ops
            raise RuntimeError("Subscription manager or JetStream not initialized")

        # WEAVELET_STREAM: Used for weaver-threadlet communication
        await self._subscription_manager.stream_manager.maybe_create_stream(
            torchLoomConstants.weaver_stream.STREAM,
            [
                torchLoomConstants.weaver_stream.subjects.DR_SUBJECT,  # Device registration
                torchLoomConstants.subjects.CONFIG_INFO,  # Config updates
                torchLoomConstants.subjects.WEAVER_COMMANDS,  # Weaver commands
            ],
        )

        logger.info("All JetStream streams setup completed")

    async def message_handler(self, msg: Msg) -> None:
        """Main message handler that dispatches to consolidated handlers."""
        try:
            env = EventEnvelope()
            env.ParseFromString(msg.data)
            logger.debug(f"Received message: {env}")

            # Route messages to appropriate consolidated handlers

            # Threadlet messages (Training Process -> Weaver)
            if (
                env.HasField("register_device")
                or env.HasField("heartbeat")
                or env.HasField("training_status")
                or env.HasField("device_status")
                or env.HasField("drain")
            ):
                await self._handlers["threadlet"].handle(env)

            # External system messages (External Systems -> Weaver)
            if env.HasField("monitored_fail") or env.HasField("config_info"):
                await self._handlers["external"].handle(env)

            # UI messages (UI -> Weaver)
            if env.HasField("ui_command"):
                await self._handlers["ui"].handle(env)

            # Note: ui_status_update is handled by the UIUpdatePublisher in the background task
            # No need to handle it here as it's an outbound message type

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

    async def start_heartbeat_monitor(self) -> None:
        """Start the background task to monitor dead replicas and publish failure events."""
        logger.info("Starting heartbeat monitor")

        while not self._stop_nats.is_set():
            try:
                # Check for dead replicas every 30 seconds
                if self.threadlet_command_handler:
                    dead_replicas = (
                        await self.threadlet_command_handler.check_and_publish_dead_replicas()
                    )
                    if dead_replicas:
                        logger.info(
                            f"Published failure events for dead replicas: {dead_replicas}"
                        )

                await asyncio.sleep(TimeConstants.HEARTBEAT_MONITOR_INTERVAL)

            except Exception as e:
                logger.exception(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(30.0)

    async def stop(self) -> None:
        """Stop the weaver, its components, and close NATS connection."""
        logger.info("Stopping Weaver...")
        self._stop_nats.set()  # Signal all async tasks managed by stop_event to terminate

        # Stop WebSocket server
        if self.websocket_server:
            await self.websocket_server.stop()
            logger.info("WebSocket server stopped.")

        # Stop UI update publisher (if it has a specific stop method, otherwise it relies on NATS closure)
        # Assuming UIUpdatePublisher's tasks will end when NATS connection closes or via stop_event if integrated.

        # Stop heartbeat monitor (if it has a specific stop method or task to cancel)
        # Similar to above, assuming it will terminate.

        # Stop SubscriptionManager (this will stop all its subscriptions and close NATS)
        if self._subscription_manager:
            await self._subscription_manager.close()
            logger.info("Subscription manager stopped and NATS connection closed.")
        
        # Cancel any other top-level tasks specific to Weaver not covered by _stop_nats
        # For example, if start_ui_server or start_heartbeat_monitor create tasks not directly
        # using _stop_nats in their loops, they would need explicit cancellation here.
        # However, based on current structure, they seem to rely on NATS or are short-lived setup.

        logger.info("Weaver stopped successfully.")

    @property
    def device_to_replicas(self) -> Dict[str, Set[str]]:
        """Get device to replicas mapping."""
        return self._device_mapper.device_to_replicas

    @property
    def replica_to_devices(self) -> Dict[str, Set[str]]:
        """Get replica to devices mapping."""
        return self._device_mapper.replica_to_devices

    def override_handler(self, handler_category: str, handler: MessageHandler) -> None:
        """Override a consolidated handler with a custom implementation.

        Args:
            handler_category: The handler category to override ("threadlet", "external", or "ui")
            handler: The custom handler instance to use

        Example:
            # Override the threadlet handler with custom logic
            custom_handler = MyCustomThreadletHandler(...)
            weaver.override_handler("threadlet", custom_handler)
        """
        if not self._handlers:
            raise RuntimeError("Weaver not initialized. Call initialize() first.")

        valid_categories = ["threadlet", "external", "ui"]
        if handler_category not in valid_categories:
            raise ValueError(
                f"Invalid handler category: {handler_category}. Must be one of {valid_categories}"
            )

        self._handlers[handler_category] = handler
        logger.info(f"Overrode {handler_category} handler")

    def add_custom_handler(
        self, handler_category: str, handler: MessageHandler
    ) -> None:
        """Add a handler for a custom category.

        Args:
            handler_category: The custom handler category name
            handler: The handler instance to use

        Note: You'll also need to add the corresponding message routing logic
        in the message_handler method for custom categories.
        """
        if not self._handlers:
            raise RuntimeError("Weaver not initialized. Call initialize() first.")

        self._handlers[handler_category] = handler
        logger.info(f"Added custom handler for category: {handler_category}")

    def get_handler(self, handler_category: str) -> Optional[MessageHandler]:
        """Get the current handler for a category.

        Args:
            handler_category: The handler category to get ("threadlet", "external", or "ui")

        Returns:
            The handler instance or None if not found
        """
        return self._handlers.get(handler_category) if self._handlers else None

    def list_handlers(self) -> Dict[str, str]:
        """List all currently registered handlers.

        Returns:
            Dictionary mapping handler categories to handler class names
        """
        if not self._handlers:
            return {}
        return {
            category: handler.__class__.__name__
            for category, handler in self._handlers.items()
        }

    def get_supported_events(self) -> Dict[str, str]:
        """Get all supported event types from the EventEnvelope protobuf.

        Returns:
            Dictionary mapping protobuf field names to their descriptions
        """
        return {
            "register_device": "Device registration from threadlets",
            "monitored_fail": "Device failure notifications",
            "config_info": "Configuration change events",
            "drain": "Graceful device drain requests",
            "heartbeat": "Threadlet liveness monitoring",
            "weaver_command": "Command acknowledgments from threadlets",
            "training_status": "Training progress updates",
            "device_status": "device status and utilization data",
            "ui_command": "Commands from the UI",
            "ui_status_update": "UI status updates (outbound only)",
        }


async def main():
    """Main function to start the weaver."""
    try:
        logger.info("Starting torchLoom Weaver with UI integration")
        weaver = Weaver(enable_ui=True)
        await weaver.initialize()
        logger.info("Weaver initialized")

        # Start all services concurrently
        async with asyncio.TaskGroup() as tg:

            # NATS subscriptions (distributed communication)
            # Streams are already created in weaver.initialize() -> _setup_all_streams()
            tg.create_task(
                weaver._subscription_manager.subscribe_js(
                    torchLoomConstants.weaver_stream.STREAM,
                    torchLoomConstants.weaver_stream.subjects.DR_SUBJECT,
                    torchLoomConstants.weaver_stream.CONSUMER,
                    weaver.message_handler,
                )
            )
            tg.create_task(
                weaver._subscription_manager.subscribe_nc(
                    subject=torchLoomConstants.subjects.MONITOR,
                    message_handler=weaver.message_handler,
                )
            )
            tg.create_task(
                weaver._subscription_manager.subscribe_nc(
                    subject=torchLoomConstants.subjects.CONFIG_INFO,
                    message_handler=weaver.message_handler,
                )
            )

            # Training Process -> Weaver subscriptions
            tg.create_task(
                weaver._subscription_manager.subscribe_nc(
                    subject=torchLoomConstants.subjects.TRAINING_STATUS,
                    message_handler=weaver.message_handler,
                )
            )
            tg.create_task(
                weaver._subscription_manager.subscribe_nc(
                    subject=torchLoomConstants.subjects.device_STATUS,
                    message_handler=weaver.message_handler,
                )
            )
            tg.create_task(
                weaver._subscription_manager.subscribe_nc(
                    subject=torchLoomConstants.subjects.HEARTBEAT,
                    message_handler=weaver.message_handler,
                )
            )

            # UI services (direct WebSocket communication - no NATS)
            if weaver.enable_ui:
                tg.create_task(weaver.start_ui_server())

            # Background tasks
            tg.create_task(weaver.start_ui_update_publisher())
            tg.create_task(weaver.start_heartbeat_monitor())

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
