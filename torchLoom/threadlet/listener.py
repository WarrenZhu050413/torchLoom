"""
Async threadlet listener implementation.
"""

import asyncio
import json
import multiprocessing
import os
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Optional, Union

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from multiprocessing import Event

from nats.aio.client import Client
from nats.aio.msg import Msg
from nats.js.api import RetentionPolicy, StorageType, StreamConfig
from nats.js.client import JetStreamContext

import nats
from torchLoom.common import TrainingStatus, deviceStatus
from torchLoom.common.config import Config
from torchLoom.common.constants import TimeConstants, WeaverStream, torchLoomConstants
from torchLoom.common.publishers import EventPublisher
from torchLoom.common.subscription import SubscriptionManager
from torchLoom.common.utils import get_device_uuid
from torchLoom.log.logger import setup_logger
from torchLoom.proto import torchLoom_pb2
from torchLoom.proto.torchLoom_pb2 import EventEnvelope, RegisterDevice

from .message import (
    CommandType,
    MessageFactory,
    MessageType,
    deserialize_message,
    serialize_message,
)
from .publishers import ThreadletEventPublisher

logger = setup_logger(name="threadlet_listener")


class ThreadletListener:
    """Async implementation of threadlet listener that runs inside the subprocess."""

    def __init__(
        self,
        replica_id: str,
        torchLoom_addr: str,
        pipe_to_main_process: "Connection",
        stop_event: "Event",
    ):
        # Setup logging
        self._logger = setup_logger(
            name="threadlet_logger",
            log_file=Config.MANAGER_torchLoom_LOG_FILE,
            format_log=True,
            print_to_console=False,
        )

        # Core identifiers
        self._replica_id = replica_id
        self._device_uuid: Optional[str] = None

        # NATS connection setup via SubscriptionManager
        self._torchLoom_addr = torchLoom_addr
        # Create an asyncio.Event that mirrors the multiprocessing.Event
        # This is because SubscriptionManager expects an asyncio.Event
        self._async_stop_event = asyncio.Event()
        self._subscription_manager = SubscriptionManager(
            torchLoom_addr=self._torchLoom_addr,
            stop_event=self._async_stop_event,  # Pass the asyncio.Event here
        )
        # self._nc, self._js, self._subscriptions are now managed by _subscription_manager

        # Common publisher for all event publishing
        self._event_publisher: Optional[EventPublisher] = None
        self._threadlet_publisher: Optional[ThreadletEventPublisher] = None

        # Inter-process communication using a single duplex pipe
        self._pipe_to_main_process = pipe_to_main_process
        self._stop_event = stop_event

        # Configuration from constants
        self._nc_timeout = Config.NC_TIMEOUT or TimeConstants.PIPE_POLL_INTERVAL
        self._exception_sleep = (
            Config.EXCEPTION_RETRY_TIME or TimeConstants.ERROR_RETRY_SLEEP
        )

        self._logger.info(
            f"ThreadletListener initialized with replica_id: {self._replica_id}"
        )

    async def run(self) -> None:
        """Main run loop for the async threadlet listener."""
        try:
            # Start a task to monitor the multiprocessing stop_event and set the asyncio stop_event
            mp_event_monitor_task = asyncio.create_task(self._monitor_mp_stop_event())

            await self._subscription_manager.initialize()  # Replaces self._connect()

            # Initialize the common event publisher
            self._event_publisher = EventPublisher(
                nats_client=self._subscription_manager.nc,
                js_client=self._subscription_manager.js,
            )

            await self._setup_subscriptions_with_manager()  # New method using SubscriptionManager
            await self._register_device()

            # Initialize the threadlet-specific publisher after device registration
            if self._device_uuid:
                self._threadlet_publisher = ThreadletEventPublisher(
                    replica_id=self._replica_id,
                    device_uuid=self._device_uuid,
                    event_publisher=self._event_publisher,
                )

            # Start background tasks including the async pipe listener
            async_tasks = [
                asyncio.create_task(self._heartbeat_loop()),
                asyncio.create_task(self._async_pipe_message_processor()),
            ]

            self._logger.info(
                "ThreadletListener async initialization completed, starting main loop"
            )

            # Wait for stop signal or task completion
            done, pending = await asyncio.wait(
                async_tasks + [mp_event_monitor_task],  # Add the new monitor task
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    self._logger.debug(f"Async task {task.get_name()} cancelled.")
                except Exception as e:
                    self._logger.warning(
                        f"Error during cancellation of async task: {e}"
                    )

            self._logger.info("ThreadletListener main loop completed")

        except Exception as e:
            self._logger.exception(f"Error in threadlet listener run loop: {e}")
        finally:
            # Ensure async_stop_event is set if cleanup is triggered by mp_event_monitor_task completion
            self._async_stop_event.set()
            await self._cleanup()

    async def _monitor_mp_stop_event(self) -> None:
        """Monitors the multiprocessing.Event and sets the asyncio.Event."""
        self._logger.debug("Started monitoring multiprocessing stop_event.")
        while not self._stop_event.is_set():  # self._stop_event is the mp.Event
            await asyncio.sleep(TimeConstants.MONITOR_STOP_EVENT_SLEEP)
        self._logger.info(
            "Multiprocessing stop_event detected, setting asyncio stop_event."
        )
        self._async_stop_event.set()  # self._async_stop_event is the asyncio.Event

    async def _setup_subscriptions_with_manager(self) -> None:
        """Set up subscriptions using SubscriptionManager."""
        try:
            # The weaver should have already created the WEAVELET_STREAM with all necessary subjects
            # Just subscribe to existing stream without trying to create or modify it
            self._logger.info(
                f"Setting up subscriptions to existing stream {WeaverStream.STREAM} using SubscriptionManager"
            )

            # Subscribe to WEAVER_COMMANDS
            await self._subscription_manager.subscribe_js(
                stream=WeaverStream.STREAM,
                subject=torchLoomConstants.subjects.WEAVER_COMMANDS,
                consumer=f"threadlet-{self._replica_id}",
                message_handler=self._handle_weaver_command,
            )
            # Add other subscriptions if needed, e.g., _subscribe_nc from original
            # For example, if there were _subscribe_nc calls, they would look like:
            # await self._subscription_manager.subscribe_nc(...)

            self._logger.info(
                "ThreadletListener subscriptions set up successfully via SubscriptionManager"
            )
        except Exception as e:
            self._logger.exception(
                f"Failed to set up subscriptions via SubscriptionManager: {e}"
            )
            raise

    async def _register_device(self) -> None:
        """Register this device with the weaver using the common publisher."""
        try:
            if not self._event_publisher:
                raise RuntimeError("Event publisher not initialized")

            self._device_uuid = get_device_uuid()

            await self._event_publisher.publish_device_registration(
                device_uuid=self._device_uuid,
                replica_id=self._replica_id,
            )

            self._logger.info(
                f"Registered device {self._device_uuid} with replica {self._replica_id}"
            )
        except Exception as e:
            self._logger.exception(f"Failed to register device: {e}")
            raise

    async def _handle_weaver_command(self, msg: Msg) -> None:
        """Handle incoming weaver commands."""
        try:
            if not EventEnvelope:
                self._logger.error("EventEnvelope not available, cannot parse message")
                return

            envelope = EventEnvelope()
            envelope.ParseFromString(msg.data)

            if envelope.HasField("weaver_command"):
                weaver_command = envelope.weaver_command
                command_type = weaver_command.command_type
                target_replica_id = weaver_command.target_replica_id
                params = dict(weaver_command.params) if weaver_command.params else {}

                # Only handle commands targeting this replica
                if target_replica_id == self._replica_id:
                    self._logger.info(
                        f"Received weaver command: {command_type} with params: {params}"
                    )

                    # Send command to main process via pipe using structured messages
                    try:
                        # Convert string command type to protobuf enum value
                        cmd_type = command_type  # Use the string directly since create_command handles conversion

                        command_message = MessageFactory.create_command(
                            replica_id=self._replica_id,
                            command_type=cmd_type,
                            params=params,
                        )
                        self._send_message_to_threadlet(command_message)
                        self._logger.info(
                            f"Sent weaver command to main process: {command_type}"
                        )
                    except Exception as e:
                        self._logger.warning(
                            f"Error sending weaver command via pipe: {e}"
                        )
                else:
                    self._logger.debug(
                        f"Ignoring command for different replica: {target_replica_id}"
                    )

            await msg.ack()
        except Exception as e:
            self._logger.exception(f"Error handling weaver command: {e}")

    async def _heartbeat_loop(self) -> None:
        """Background task to send periodic heartbeat messages to the weaver."""
        self._logger.info("Started heartbeat loop")

        while not self._async_stop_event.is_set():  # Use asyncio stop event
            try:
                await self._send_heartbeat()
                await asyncio.sleep(TimeConstants.HEARTBEAT_SEND_INTERVAL)
            except Exception as e:
                self._logger.exception(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(TimeConstants.ERROR_RETRY_SLEEP)

    async def _send_heartbeat(self) -> None:
        """Send a heartbeat message to the weaver using the threadlet publisher."""
        try:
            if not self._threadlet_publisher:
                self._logger.warning(
                    "Cannot send heartbeat - threadlet publisher not initialized"
                )
                return

            # Create metadata
            metadata = {
                "process_id": str(os.getpid()) if hasattr(os, "getpid") else "unknown",
                "nats_addr": self._torchLoom_addr,
            }

            await self._threadlet_publisher.publish_heartbeat(
                status="active",
                metadata=metadata,
            )

            self._logger.debug(f"Sent heartbeat for replica {self._replica_id}")

        except Exception as e:
            self._logger.exception(f"Failed to send heartbeat: {e}")

    async def _async_pipe_message_processor(self) -> None:
        """Async task that listens for messages on the pipe from the main Threadlet process."""
        self._logger.info("Async pipe message processor started for ThreadletListener.")

        try:
            while not self._async_stop_event.is_set():  # Use asyncio stop event
                try:
                    # Use asyncio.to_thread to run the blocking poll operation
                    has_data = await asyncio.to_thread(
                        self._pipe_to_main_process.poll,
                        TimeConstants.PIPE_POLL_INTERVAL,
                    )

                    if has_data:
                        # Use asyncio.to_thread to run the blocking recv operation
                        raw_message = await asyncio.to_thread(
                            self._pipe_to_main_process.recv
                        )
                        self._logger.debug(
                            f"Received raw message from main process: {raw_message}"
                        )

                        await self._process_pipe_message(raw_message)

                    # Small sleep to prevent busy waiting and allow other tasks to run
                    await asyncio.sleep(TimeConstants.ASYNC_TASK_SLEEP)

                except EOFError:
                    self._logger.info(
                        "Pipe to main process closed, main process likely terminated."
                    )
                    self._async_stop_event.set()  # Ensure other loops stop
                    break
                except Exception as e:
                    if not self._async_stop_event.is_set():  # Check asyncio stop event
                        self._logger.exception(
                            f"Error in async pipe message processor: {e}"
                        )
                        await asyncio.sleep(TimeConstants.BRIEF_PAUSE)

        except asyncio.CancelledError:
            self._logger.info("Async pipe message processor cancelled.")
            raise
        finally:
            self._logger.info(
                "Async pipe message processor stopped for ThreadletListener."
            )

    async def _process_pipe_message(self, raw_message) -> None:
        """Process a message received from the pipe."""
        try:
            # Try to deserialize the message using protobuf
            message = None
            if isinstance(raw_message, bytes):
                # Try to deserialize as different message types
                message = (
                    deserialize_message(raw_message, "METRICS")
                    or deserialize_message(raw_message, "HEARTBEAT")
                    or deserialize_message(raw_message, "COMMAND")
                )

            if message:
                if message.message_type == MessageType.METRICS:
                    await self._handle_metrics_message(message)
                elif message.message_type == MessageType.HEARTBEAT:
                    self._logger.debug(
                        f"Received heartbeat from main process: {message.status}"
                    )
                else:
                    self._logger.warning(
                        f"Received unexpected message type: {message.message_type}"
                    )
        except Exception as e:
            self._logger.exception(f"Error processing pipe message: {e}")

    async def _handle_metrics_message(self, message) -> None:
        """Handle metrics message from Threadlet using the threadlet publisher."""
        try:
            if not self._threadlet_publisher:
                self._logger.warning(
                    "Cannot publish metrics - threadlet publisher not initialized"
                )
                return

            # Extract metrics from the message
            loss = message.loss if message.HasField("loss") else None
            accuracy = message.accuracy if message.HasField("accuracy") else None
            gradient_norm = (
                message.gradient_norm if message.HasField("gradient_norm") else None
            )

            # Get additional metrics
            additional_metrics = dict(message.metrics)

            await self._threadlet_publisher.publish_metrics(
                step=message.step,
                epoch=message.epoch,
                loss=loss,
                accuracy=accuracy,
                gradient_norm=gradient_norm,
                **additional_metrics,
            )

            self._logger.debug(f"Published metrics for replica {self._replica_id}")
        except Exception as e:
            self._logger.exception(f"Error handling metrics message: {e}")

    def _send_message_to_threadlet(
        self,
        message: Union[
            torchLoom_pb2.PipeHeartbeatMessage,
            torchLoom_pb2.PipeMetricsMessage,
            torchLoom_pb2.PipeCommandMessage,
        ],
    ) -> None:
        """Send a structured protobuf message to the main Threadlet process as a (type_str, bytes) tuple."""
        try:
            if self._pipe_to_main_process and not self._pipe_to_main_process.closed:
                message_type_str = ""
                if isinstance(message, torchLoom_pb2.PipeHeartbeatMessage):
                    message_type_str = "HEARTBEAT"
                elif isinstance(message, torchLoom_pb2.PipeMetricsMessage):
                    message_type_str = "METRICS"
                elif isinstance(message, torchLoom_pb2.PipeCommandMessage):
                    message_type_str = "COMMAND"
                else:
                    # This case should ideally not be reached if type hints are respected
                    self._logger.warning(
                        f"Attempting to send unknown or non-pipe message type {type(message)} via pipe."
                    )
                    return

                serialized_data = serialize_message(message)  # Returns bytes
                self._pipe_to_main_process.send((message_type_str, serialized_data))
                self._logger.debug(
                    f"Sent {message_type_str} message to Threadlet via pipe."
                )
        except (BrokenPipeError, OSError):
            self._logger.warning(
                "Pipe to main process is broken or closed, dropping message."
            )
        except Exception as e:
            self._logger.warning(f"Error sending message to Threadlet via pipe: {e}")

    async def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._logger.info("Starting ThreadletListener cleanup...")
            self._async_stop_event.set()  # Ensure all async loops are signaled

            # Close SubscriptionManager (handles NATS connection and its subscriptions)
            if self._subscription_manager:
                await self._subscription_manager.close()
                self._logger.info("SubscriptionManager closed.")

            # Perform cleanup in logical order
            await self._cleanup_pipes()

            self._logger.info("ThreadletListener cleanup completed")
        except Exception as e:
            self._logger.exception(f"Error during cleanup: {e}")

    async def _cleanup_pipes(self) -> None:
        """Close pipe connections."""
        try:
            self._logger.info("Cleaning up pipes...")

            if self._pipe_to_main_process and not self._pipe_to_main_process.closed:
                self._pipe_to_main_process.close()
                self._logger.info("Pipe to main process closed")
            else:
                self._logger.debug(
                    "Pipe to main process already closed or not available"
                )

        except Exception as e:
            self._logger.warning(f"Error closing pipe to main process: {e}")
