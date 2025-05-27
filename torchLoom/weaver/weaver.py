"""
Refactored torchLoom Weaver implementation.

This module contains the main Weaver class that orchestrates message handling
and subscription management using the extracted components.
"""

import asyncio
from typing import Dict, Optional, Set

from nats.aio.msg import Msg

from torchLoom.common.config import Config
from torchLoom.common.constants import (
    HandlerConstants,
    NetworkConstants,
    TimeConstants,
    torchLoomConstants,
)
from torchLoom.common.handlers import *
from torchLoom.common.subscription import SubscriptionManager
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

from .handlers import (
    ExternalHandler,
    ThreadletHandler,
    UIHandler,
    create_weaver_message_registry,
)
from .publishers import (
    StatusTracker,
    ThreadletCommandPublisher,
    UIUpdatePublisher,
    WeaverUIStatusPublisher,
)
from .websocket_server import WebSocketServer

logger = setup_logger(
    name="torchLoom_weaver", log_file=Config.torchLoom_CONTROLLER_LOG_FILE
)


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
    class MyCustomHeartbeatHandler(BaseHandler):
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
        ui_host: str = NetworkConstants.DEFAULT_UI_HOST,
        ui_port: int = NetworkConstants.DEFAULT_UI_PORT,
    ) -> None:
        self._stop_nats = asyncio.Event()
        self.seq = 0

        self._subscription_manager = SubscriptionManager(
            torchLoom_addr=torchLoom_addr, stop_event=self._stop_nats
        )

        self.status_tracker = StatusTracker()
        self.websocket_server = None
        self.enable_ui = enable_ui
        self.ui_host = ui_host
        self.ui_port = ui_port

        self._handlers: Dict[str, BaseHandler] = {}
        self._message_registry: Optional[HandlerRegistry] = None
        self.ui_update_handler: Optional[UIUpdatePublisher] = None
        self.threadlet_command_handler: Optional[ThreadletCommandPublisher] = None

        logger.info(f"Weaver initialized with NATS address: {torchLoom_addr}")
        if enable_ui:
            logger.info(f"UI server will be started on {ui_host}:{ui_port}")

    async def initialize(self) -> None:
        """Initialize NATS connection and JetStream via SubscriptionManager."""
        await self._subscription_manager.initialize()
        await self._setup_all_streams()

        if self.enable_ui:
            self.websocket_server = WebSocketServer(
                status_tracker=self.status_tracker,
                weaver=self,
                host=self.ui_host,
                port=self.ui_port,
            )

        # Create and inject the UI status publisher into the status tracker
        ui_status_publisher = WeaverUIStatusPublisher(self.status_tracker)
        self.status_tracker.set_ui_publisher(ui_status_publisher)

        self.ui_update_handler = UIUpdatePublisher(self.status_tracker)
        self.threadlet_command_handler = ThreadletCommandPublisher(
            nats_client=self._subscription_manager.nc,
        )

        # Initialize the unified message registry
        self._message_registry = create_weaver_message_registry()

        # Create and register handlers
        threadlet_handler = ThreadletHandler(
            self.status_tracker,
            heartbeat_timeout=TimeConstants.HEARTBEAT_TIMEOUT,
        )
        external_handler = ExternalHandler(self.status_tracker)
        ui_handler = UIHandler(
            self.status_tracker,
            weaver_publish_command_func=self.threadlet_command_handler.publish_weaver_command,
        )

        # Register handlers
        self._handlers = {
            "threadlet": threadlet_handler,
            "external": external_handler,
            "ui": ui_handler,
        }

        # Register in the message registry with event type mappings
        self._message_registry.register_message_handler(
            "threadlet", threadlet_handler, HandlerConstants.THREADLET_EVENTS
        )
        self._message_registry.register_message_handler(
            "external", external_handler, HandlerConstants.EXTERNAL_EVENTS
        )
        self._message_registry.register_message_handler(
            "ui", ui_handler, HandlerConstants.UI_EVENTS
        )

        logger.info("Weaver fully initialized with unified handler registry")

    async def _setup_all_streams(self) -> None:
        """Centralized setup of all JetStream streams with complete subject configurations."""
        logger.info("Setting up all JetStream streams...")
        if not self._subscription_manager or not self._subscription_manager.js:
            logger.error(
                "Subscription manager or JetStream not initialized before stream setup."
            )
            raise RuntimeError("Subscription manager or JetStream not initialized")

        sm = self._subscription_manager.stream_manager

        # WEAVER_INGRESS_STREAM: For all incoming messages to the Weaver
        # This stream will handle events from threadlets and external systems.
        weaver_ingress_subjects = [
            torchLoomConstants.subjects.THREADLET_EVENTS,
            torchLoomConstants.subjects.EXTERNAL_EVENTS,
        ]
        await sm.maybe_create_stream(
            stream=torchLoomConstants.weaver_ingress_stream.STREAM,
            subjects=list(set(weaver_ingress_subjects)),
        )
        logger.info(
            f"Set up stream: {torchLoomConstants.weaver_ingress_stream.STREAM} with subjects: {weaver_ingress_subjects}"
        )

        # WEAVELET_STREAM: Now primarily for outbound Weaver -> Threadlet commands
        weaver_commands_stream_name = torchLoomConstants.weaver_stream.STREAM
        weaver_commands_subjects = [torchLoomConstants.subjects.WEAVER_COMMANDS]
        await sm.maybe_create_stream(
            stream=weaver_commands_stream_name, subjects=weaver_commands_subjects
        )
        logger.info(
            f"Set up stream: {weaver_commands_stream_name} with subjects: {weaver_commands_subjects}"
        )

        # UI_STREAM (if used and is JetStream based)
        # Based on original constants, UI_STREAM exists. Assuming it's still needed.
        if (
            hasattr(torchLoomConstants, "ui_stream")
            and torchLoomConstants.ui_stream.STREAM
        ):
            ui_stream_subjects = []
            if hasattr(torchLoomConstants.subjects, "UI_COMMANDS"):
                ui_stream_subjects.append(torchLoomConstants.subjects.UI_COMMANDS)

            if ui_stream_subjects:
                await sm.maybe_create_stream(
                    stream=torchLoomConstants.ui_stream.STREAM,
                    subjects=list(set(ui_stream_subjects)),
                )
                logger.info(
                    f"Set up stream: {torchLoomConstants.ui_stream.STREAM} with subjects: {ui_stream_subjects}"
                )
            else:
                logger.info(
                    f"Skipping UI_STREAM setup as no relevant subjects (e.g., UI_COMMANDS) are defined for it."
                )
        else:
            logger.info("UI_STREAM not defined in constants, skipping its setup.")

        logger.info("All JetStream streams setup completed based on new design")

    async def message_handler(self, msg: Msg) -> None:
        """Main message handler that dispatches to specific handlers using the unified registry."""
        try:
            env = EventEnvelope()
            env.ParseFromString(msg.data)
            payload_type = env.WhichOneof("payload")
            logger.debug(
                f"Received message with payload type: {payload_type} on subject {msg.subject}"
            )

            if payload_type is None:
                logger.warning(f"Received message with no payload: {msg.subject}")
                return

            # Use the unified message registry to find the appropriate handler
            handler = None
            if self._message_registry:
                handler = self._message_registry.get_handler_for_event_type(
                    payload_type
                )

            if handler:
                await handler.handle(env)
                logger.debug(
                    f"Successfully dispatched {payload_type} to {handler.__class__.__name__}"
                )
            else:
                logger.warning(
                    f"No handler found for payload type '{payload_type}' on subject '{msg.subject}'"
                )

        except Exception as e:
            logger.exception(
                f"Error in Weaver message_handler: {e} while processing message from subject {msg.subject if msg else 'N/A'}"
            )

    def get_replicas_for_device(self, device_uuid: str) -> Set[str]:
        """Get all replicas associated with a device."""
        return self.status_tracker.get_replicas_for_device(device_uuid)

    def get_devices_for_replica(self, replica_id: str) -> Set[str]:
        """Get all devices associated with a replica."""
        return self.status_tracker.get_devices_for_replica(replica_id)

    async def start_ui_server(self) -> None:
        """Start the WebSocket UI server."""
        if self.websocket_server:
            logger.info("Starting WebSocket UI server")
            await self.websocket_server.run_with_status_broadcaster()
        else:
            logger.info("UI server is not enabled or not initialized.")

    async def start_ui_update_publisher(self) -> None:
        """Start the background task to publish UI updates periodically."""
        logger.info("Starting UI update publisher task")
        while not self._stop_nats.is_set():
            try:
                if self.ui_update_handler and self.websocket_server:
                    constructed_envelope = (
                        await self.status_tracker.create_ui_status_update()
                    )
                    if constructed_envelope:
                        logger.debug(
                            "UIStatusUpdate envelope constructed by status tracker for potential internal use/logging."
                        )
                await asyncio.sleep(TimeConstants.UI_UPDATE_INTERVAL)
            except asyncio.CancelledError:
                logger.info("UI update publisher task cancelled.")
                break
            except Exception as e:
                logger.exception(f"Error in UI update publisher task: {e}")
                await asyncio.sleep(TimeConstants.ERROR_RETRY_SLEEP)

    async def start_heartbeat_monitor(self) -> None:
        """Start the background task to monitor dead replicas and publish failure events."""
        logger.info("Starting heartbeat monitor task")
        threadlet_handler = self._handlers.get("threadlet")
        if not isinstance(threadlet_handler, ThreadletHandler):
            logger.error(
                "ThreadletHandler not found or incorrect type for heartbeat monitor."
            )
            return

        while not self._stop_nats.is_set():
            try:
                dead_replicas = threadlet_handler.check_dead_replicas()
                if dead_replicas:
                    logger.warning(
                        f"Heartbeat monitor identified dead replicas: {dead_replicas}"
                    )
                    for replica_id in dead_replicas:
                        self.status_tracker.update_training_progress(
                            replica_id=replica_id, status="dead"
                        )
                await asyncio.sleep(TimeConstants.HEARTBEAT_MONITOR_INTERVAL)
            except asyncio.CancelledError:
                logger.info("Heartbeat monitor task cancelled.")
                break
            except Exception as e:
                logger.exception(f"Error in heartbeat monitor task: {e}")
                await asyncio.sleep(TimeConstants.HEARTBEAT_MONITOR_INTERVAL)

    async def stop(self) -> None:
        """Stop the weaver, its components, and close NATS connection."""
        logger.info("Stopping Weaver...")
        self._stop_nats.set()

        if self.websocket_server and hasattr(self.websocket_server, "stop"):
            try:
                await self.websocket_server.stop()
                logger.info("WebSocket server signaled to stop.")
            except Exception as e:
                logger.exception(f"Error stopping WebSocket server: {e}")

        await asyncio.sleep(TimeConstants.CLEANUP_SLEEP)

        if self._subscription_manager:
            await self._subscription_manager.close()
            logger.info("Subscription manager stopped and NATS connection closed.")

        logger.info("Weaver stopped successfully.")

    @property
    def device_to_replicas(self) -> Dict[str, Set[str]]:
        """Get device to replicas mapping."""
        return self.status_tracker.device_to_replicas

    @property
    def replica_to_devices(self) -> Dict[str, Set[str]]:
        """Get replica to devices mapping."""
        return self.status_tracker.replica_to_devices

    def override_handler(self, handler_category: str, handler: BaseHandler) -> None:
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

        # Also update the message registry if it exists
        if self._message_registry and handler_category in [
            "threadlet",
            "external",
            "ui",
        ]:
            event_types = {
                "threadlet": HandlerConstants.THREADLET_EVENTS,
                "external": HandlerConstants.EXTERNAL_EVENTS,
                "ui": HandlerConstants.UI_EVENTS,
            }.get(handler_category, [])

            self._message_registry.register_message_handler(
                handler_category, handler, event_types
            )

        logger.info(
            f"Overrode {handler_category} handler with {handler.__class__.__name__}"
        )

    def add_custom_handler(self, handler_category: str, handler: BaseHandler) -> None:
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
        logger.info(
            f"Added custom handler for category: {handler_category} with {handler.__class__.__name__}"
        )

    def get_handler(self, handler_category: str) -> Optional[BaseHandler]:
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
        """Get all supported event types from the unified registry.

        Returns:
            Dictionary mapping event types to their handler categories
        """
        if self._message_registry:
            return self._message_registry.get_supported_events()

        # Fallback to static mapping if registry not available
        return {
            "register_device": "threadlet",
            "heartbeat": "threadlet",
            "training_status": "threadlet",
            "device_status": "threadlet",
            "drain": "threadlet",
            "monitored_fail": "external",
            "config_info": "ui",
            "ui_command": "ui",
        }

    def get_registry_info(self) -> Dict[str, any]:
        """Get information about the unified handler registry.

        Returns:
            Dictionary with registry information including handlers and event mappings
        """
        if not self._message_registry:
            return {"error": "Message registry not initialized"}

        return {
            "handlers": self._message_registry.list_handlers(),
            "event_mappings": self._message_registry.get_supported_events(),
            "registry_name": self._message_registry.name,
        }


async def main():
    """Main function to start the weaver."""
    weaver: Optional[Weaver] = None
    try:
        logger.info("Starting torchLoom Weaver")
        weaver = Weaver(enable_ui=True)
        await weaver.initialize()
        logger.info("Weaver initialized successfully.")

        async with asyncio.TaskGroup() as tg:
            ingress_subjects_to_subscribe = [
                torchLoomConstants.subjects.THREADLET_EVENTS,
                torchLoomConstants.subjects.EXTERNAL_EVENTS,
                torchLoomConstants.subjects.UI_COMMANDS,
            ]

            for i, subject in enumerate(list(set(ingress_subjects_to_subscribe))):
                consumer_name = f"weaver-ingress-consumer-{subject.replace('.', '-')}-{weaver.seq+i}"
                tg.create_task(
                    weaver._subscription_manager.subscribe_js(
                        stream=torchLoomConstants.weaver_ingress_stream.STREAM,
                        subject=subject,
                        consumer=consumer_name,
                        message_handler=weaver.message_handler,
                    )
                )
                logger.info(
                    f"Subscribed to JS subject: {subject} on stream {torchLoomConstants.weaver_ingress_stream.STREAM} with consumer {consumer_name}"
                )

            if weaver.enable_ui and weaver.websocket_server:
                tg.create_task(weaver.start_ui_server())
                logger.info("UI server task created.")

            tg.create_task(weaver.start_heartbeat_monitor())
            logger.info("Background tasks (heartbeat monitor) created.")

            logger.info("Weaver running. All core services and subscriptions started.")
            await asyncio.Event().wait()

    except KeyboardInterrupt:
        logger.info("Weaver shutting down due to KeyboardInterrupt...")
    except Exception as e:
        logger.exception(f"Critical error in Weaver main loop: {e}")
    finally:
        if weaver:
            logger.info("Performing final cleanup for Weaver...")
            await weaver.stop()
            logger.info("Weaver cleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())
