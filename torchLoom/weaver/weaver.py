"""
Refactored torchLoom Weaver implementation.

This module contains the main Weaver class that orchestrates message handling
and subscription management using the extracted components.
"""

import asyncio
import time
from typing import Any, Dict, Optional, Set

from nats.aio.msg import Msg

from torchLoom.common.constants import (
    HandlerConstants,
    LoggerConstants,
    NatsConstants,
    TimeConstants,
    UINetworkConstants,
)
from torchLoom.common.handlers import HandlerRegistry
from torchLoom.common.subscription import SubscriptionManager
from torchLoom.log_utils.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

from . import handlers
from .publishers import ThreadletCommandPublisher
from .status_tracker import StatusTracker
from .ui_interface import UINotificationManager
from .websocket_server import WebSocketServer

logger = setup_logger(
    name="torchLoom_weaver", log_file=LoggerConstants.torchLoom_CONTROLLER_LOG_FILE
)


class Weaver:
    """Weaver for torchLoom.

    This class is responsible for managing the training process, including
    handling events and managing resources. It runs in a separate thread to
    avoid blocking the main thread.
    """

    def __init__(
        self,
        torchLoom_addr: str = NatsConstants.DEFAULT_ADDR,
        enable_ui: bool = True,
        ui_host: str = UINetworkConstants.DEFAULT_UI_HOST,
        ui_port: int = UINetworkConstants.DEFAULT_UI_PORT,
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

        self.ui_notification_manager = UINotificationManager()

        self._handler_registry: Optional[HandlerRegistry] = None
        self.threadlet_command_handler: Optional[ThreadletCommandPublisher] = None

        self._heartbeat_tracker = {
            "last_heartbeats": {},
            "dead_processes": set(),
        }

        logger.info(f"Weaver initialized with NATS address: {torchLoom_addr}")
        if enable_ui:
            logger.info(f"UI server will be started on {ui_host}:{ui_port}")

    async def initialize(self) -> None:
        """Initialize NATS connection and JetStream via SubscriptionManager."""
        await self._subscription_manager.initialize()
        await self._setup_all_streams()

        if self.enable_ui:
            self.websocket_server = WebSocketServer(
                host=self.ui_host,
                port=self.ui_port,
            )
            self.ui_notification_manager.set_websocket_send_func(
                self.websocket_server.send_to_all
            )
            self.ui_notification_manager.set_status_tracker(self.status_tracker)

            self.websocket_server.set_ui_command_handler(
                self._handle_ui_websocket_command
            )
            self.websocket_server.set_initial_status_provider(
                self.ui_notification_manager.get_status_data_for_initial_connection
            )

        self.status_tracker.connect_ui_manager(self.ui_notification_manager)

        self.threadlet_command_handler = ThreadletCommandPublisher(
            nats_client=self._subscription_manager.nc,
            js_client=self._subscription_manager.js,
        )

        self._handler_registry = HandlerRegistry("weaver_main")
        self._register_handlers()

        logger.info("Weaver fully initialized with simplified handler registry")

    def _register_handlers(self) -> None:
        """Register all message handlers directly with the registry."""
        self._handler_registry.register_handler(
            "register_device", handlers.handle_device_registration
        )
        self._handler_registry.register_handler("heartbeat", handlers.handle_heartbeat)
        self._handler_registry.register_handler(
            "training_status", handlers.handle_training_status
        )
        self._handler_registry.register_handler(
            "device_status", handlers.handle_device_status
        )
        self._handler_registry.register_handler(
            "ui_command", handlers.handle_ui_command
        )
        logger.info("Registered all handlers with simplified registry")

    async def _handle_message(self, env: EventEnvelope) -> None:
        try:
            payload_type = env.WhichOneof("body")
            logger.debug(f"Received message with payload type: {payload_type}")

            if payload_type is None:
                logger.warning(f"Received message with no body")
                return

            if self._handler_registry and self._handler_registry.has_handler(
                payload_type
            ):
                handler_func = self._handler_registry.get_handler(payload_type)

                await handler_func(
                    env=env,
                    status_tracker=self.status_tracker,
                    heartbeat_tracker=self._heartbeat_tracker,
                    weaver_publish_command_func=self.threadlet_command_handler.publish_weaver_command,
                )
                logger.debug(f"Successfully dispatched {payload_type} to handler")
            else:
                logger.warning(f"No handler found for payload type '{payload_type}'")

        except Exception as e:
            logger.exception(f"Error in handle_message: {e}")

    async def _handle_ui_websocket_command(self, websocket_data: dict) -> None:
        try:
            command_type = websocket_data.get("type")
            logger.debug(f"Processing WebSocket UI command: {command_type}")

            envelope = EventEnvelope()

            if command_type == "ui_command":
                ui_command_data = websocket_data.get("data", {})
                ui_command = envelope.ui_command
                ui_command.command_type = ui_command_data.get("command_type", "")
                ui_command.process_id = ui_command_data.get("process_id", "")

                params = ui_command_data.get("params", {})
                for key, value in params.items():
                    ui_command.params[key] = str(value)

            else:
                logger.warning(f"Unknown WebSocket command type: {command_type}")
                return

            await self._handle_message(envelope)

        except Exception as e:
            logger.exception(f"Error handling WebSocket UI command: {e}")

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
            NatsConstants.subjects.THREADLET_EVENTS,
            NatsConstants.subjects.EXTERNAL_EVENTS,
        ]
        await sm.maybe_create_stream(
            stream=NatsConstants.weaver_ingress_stream.STREAM,
            subjects=list(set(weaver_ingress_subjects)),
        )
        logger.info(
            f"Set up stream: {NatsConstants.weaver_ingress_stream.STREAM} with subjects: {weaver_ingress_subjects}"
        )

        # WEAVELET_STREAM: Primarily for outbound Weaver -> Threadlet commands
        weaver_commands_stream_name = NatsConstants.weaver_outgress_stream.STREAM
        weaver_commands_subjects = [NatsConstants.subjects.WEAVER_COMMANDS]
        await sm.maybe_create_stream(
            stream=weaver_commands_stream_name, subjects=weaver_commands_subjects
        )
        logger.info(
            f"Set up stream: {weaver_commands_stream_name} with subjects: {weaver_commands_subjects}"
        )

    async def message_handler(self, msg: Msg) -> None:
        """NATS message handler that parses the message and delegates to handle_message."""
        try:
            env = EventEnvelope()
            env.ParseFromString(msg.data)
            logger.debug(f"Received NATS message on subject {msg.subject}")
            await self._handle_message(env)

        except Exception as e:
            logger.exception(
                f"Error in NATS message_handler: {e} while processing message from subject {msg.subject if msg else 'N/A'}"
            )

    def get_pid_for_device(self, device_uuid: str) -> Set[str]:
        """Get all replicas associated with a device."""
        return self.status_tracker.get_pid_for_device(device_uuid)

    def get_devices_for_pid(self, process_id: str) -> Set[str]:
        """Get all devices associated with a replica."""
        return self.status_tracker.get_devices_for_pid(process_id)

    async def start_ui_server(self) -> None:
        """Start the WebSocket UI server and status broadcaster."""
        if self.websocket_server and self.ui_notification_manager:
            logger.info("Starting WebSocket UI server and status broadcaster")

            # Start the status broadcaster task in UINotificationManager
            self.ui_notification_manager.start_broadcaster_task()

            # Start the WebSocket server (this will block)
            await self.websocket_server.run_server()
        else:
            logger.info("UI server is not enabled or not initialized.")

    async def start_heartbeat_monitor(self) -> None:
        """Start the background task to monitor dead replicas and publish failure events."""
        logger.info("Starting heartbeat monitor task")

        while not self._stop_nats.is_set():
            try:
                current_time = time.time()
                dead_processes = set()

                for process_id, last_heartbeat in self._heartbeat_tracker[
                    "last_heartbeats"
                ].items():
                    if current_time - last_heartbeat > TimeConstants.HEARTBEAT_TIMEOUT:
                        if process_id not in self._heartbeat_tracker["dead_processes"]:
                            logger.warning(
                                f"Replica {process_id} marked as dead (no heartbeat for {current_time - last_heartbeat:.1f}s)"
                            )
                            self._heartbeat_tracker["dead_processes"].add(process_id)
                            dead_processes.add(process_id)

                if dead_processes:
                    logger.warning(
                        f"Heartbeat monitor identified dead replicas: {dead_processes}"
                    )
                    for process_id in dead_processes:
                        # Delete the process completely from tracking instead of just marking as dead
                        deleted = self.status_tracker.delete_process(process_id)
                        if deleted:
                            logger.info(
                                f"Successfully removed dead process: {process_id}"
                            )
                        else:
                            logger.warning(
                                f"Failed to remove dead process: {process_id}"
                            )

                        # Also remove from heartbeat tracker's dead_processes set for cleanup
                        # (it will be re-added if the process somehow comes back to life)
                        self._heartbeat_tracker["last_heartbeats"].pop(process_id, None)

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

        # Stop UI notification manager broadcaster
        if self.ui_notification_manager:
            try:
                await self.ui_notification_manager.stop_broadcaster()
                logger.info("UI notification manager broadcaster stopped.")
            except Exception as e:
                logger.exception(f"Error stopping UI notification manager: {e}")

        # WebSocket server doesn't need explicit stop method anymore (it's just connection management)
        if self.websocket_server:
            logger.info("WebSocket server will stop when main task ends.")

        await asyncio.sleep(TimeConstants.CLEANUP_SLEEP)

        if self._subscription_manager:
            await self._subscription_manager.close()
            logger.info("Subscription manager stopped and NATS connection closed.")

        logger.info("Weaver stopped successfully.")

    @property
    def device_to_pid(self) -> Dict[str, Set[str]]:
        """Get device to replicas mapping."""
        return self.status_tracker.device_to_pid

    @property
    def pid_to_devices(self) -> Dict[str, Set[str]]:
        """Get replica to devices mapping."""
        return self.status_tracker.pid_to_devices


async def main():
    weaver: Optional[Weaver] = None
    try:
        logger.info("Starting torchLoom Weaver")
        weaver = Weaver(enable_ui=True)
        await weaver.initialize()
        logger.info("Weaver initialized successfully.")

        async with asyncio.TaskGroup() as tg:
            ingress_subjects_to_subscribe = [
                NatsConstants.subjects.THREADLET_EVENTS,
                NatsConstants.subjects.EXTERNAL_EVENTS,
            ]

            for i, subject in enumerate(list(set(ingress_subjects_to_subscribe))):
                consumer_name = f"weaver-ingress-consumer-{subject.replace('.', '-')}-{weaver.seq+i}"
                tg.create_task(
                    weaver._subscription_manager.subscribe_js(
                        stream=NatsConstants.weaver_ingress_stream.STREAM,
                        subject=subject,
                        consumer=consumer_name,
                        message_handler=weaver.message_handler,
                    )
                )
                logger.info(
                    f"Subscribed to JS subject: {subject} on stream {NatsConstants.weaver_ingress_stream.STREAM} with consumer {consumer_name}"
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
