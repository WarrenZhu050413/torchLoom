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
from torchLoom.common.constants import WeaverStream, torchLoomConstants
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

        # Inter-process communication using a single duplex pipe
        self._pipe_to_main_process = pipe_to_main_process
        self._stop_event = stop_event

        # Configuration
        self._nc_timeout = Config.NC_TIMEOUT or 1
        self._exception_sleep = Config.EXCEPTION_RETRY_TIME or 1

        self._logger.info(
            f"ThreadletListener initialized with replica_id: {self._replica_id}"
        )

    async def run(self) -> None:
        """Main run loop for the async threadlet listener."""
        try:
            # Start a task to monitor the multiprocessing stop_event and set the asyncio stop_event
            mp_event_monitor_task = asyncio.create_task(self._monitor_mp_stop_event())

            await self._subscription_manager.initialize()  # Replaces self._connect()
            await self._setup_subscriptions_with_manager()  # New method using SubscriptionManager
            await self._register_device()

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
            await asyncio.sleep(0.1)
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
        """Register this device with the weaver."""
        try:
            if not self._subscription_manager or not self._subscription_manager.js:
                raise RuntimeError("JetStream not initialized")
            if not EventEnvelope:
                raise RuntimeError("EventEnvelope not available")

            self._device_uuid = get_device_uuid()

            envelope = EventEnvelope()
            envelope.register_device.device_uuid = self._device_uuid
            envelope.register_device.replica_id = self._replica_id

            await self._subscription_manager.js.publish(  # Use SubscriptionManager's js
                torchLoomConstants.subjects.THREADLET_EVENTS,  # Changed subject
                envelope.SerializeToString(),
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
        heartbeat_interval = 30.0  # Send heartbeat every 30 seconds

        while not self._async_stop_event.is_set():  # Use asyncio stop event
            try:
                await self._send_heartbeat()
                await asyncio.sleep(heartbeat_interval)
            except Exception as e:
                self._logger.exception(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5.0)  # Wait a bit before retrying

    async def _send_heartbeat(self) -> None:
        """Send a heartbeat message to the weaver."""
        try:
            if (
                not self._subscription_manager
                or not self._subscription_manager.nc
                or self._subscription_manager.nc.is_closed
            ):  # Check via SM
                self._logger.warning(
                    "Cannot send heartbeat - NATS not connected via SubscriptionManager"
                )
                return

            # Import protobuf message types
            if not EventEnvelope:
                self._logger.warning("EventEnvelope not available, skipping heartbeat")
                return

            # Create heartbeat message
            envelope = EventEnvelope()
            heartbeat = envelope.heartbeat
            heartbeat.replica_id = self._replica_id
            heartbeat.device_uuid = self._device_uuid or f"device_{self._replica_id}"
            heartbeat.timestamp = int(time.time())
            heartbeat.status = "active"

            # Add some metadata (this could be extended to include training metrics)
            heartbeat.metadata["process_id"] = (
                str(os.getpid()) if hasattr(os, "getpid") else "unknown"
            )
            heartbeat.metadata["nats_addr"] = self._torchLoom_addr

            # Publish heartbeat
            await self._subscription_manager.nc.publish(  # Use SubscriptionManager's nc
                torchLoomConstants.subjects.THREADLET_EVENTS,  # Changed subject
                envelope.SerializeToString(),
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
                        self._pipe_to_main_process.poll, 0.1
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
                    await asyncio.sleep(0.01)

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
                        await asyncio.sleep(0.1)  # Brief pause before retrying

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
        """Handle metrics message from Threadlet."""
        try:
            status_dict = {
                "replica_id": message.replica_id,
                "status_type": "batch_update",
                "current_step": message.step,
                "epoch": message.epoch,
                "metrics": {
                    **dict(message.metrics),
                    "loss": message.loss if message.HasField("loss") else None,
                    "accuracy": (
                        message.accuracy if message.HasField("accuracy") else None
                    ),
                    "gradient_norm": (
                        message.gradient_norm
                        if message.HasField("gradient_norm")
                        else None
                    ),
                },
            }
            self._logger.debug(f"Received metrics to publish: {status_dict}")
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

    async def _publish_status(self, status: Dict[str, Any]) -> None:
        """Publish status to NATS using protobuf messages based on status type."""
        try:
            if (
                not self._subscription_manager
                or not self._subscription_manager.nc
                or self._subscription_manager.nc.is_closed
            ):  # Check via SM
                self._logger.warning(
                    "Cannot publish status - NATS not connected via SubscriptionManager"
                )
                return

            # Import protobuf message types
            if not EventEnvelope:
                self._logger.warning(
                    "EventEnvelope not available, skipping status publish"
                )
                return

            # Determine status type and route accordingly
            status_type = status.get("status_type", "unknown")

            if status_type == "training_status":
                await self._publish_training_status(status)
            elif status_type == "device_status":
                await self._publish_device_status(status)
            else:
                self._logger.warning(f"Unknown status type: {status_type}")

        except Exception as e:
            self._logger.exception(f"Failed to publish status: {e}")

    async def _publish_training_status(self, status: Dict[str, Any]) -> None:
        """Publish TrainingStatus message to NATS."""
        # Ensure replica_id is set
        if "replica_id" not in status:
            status["replica_id"] = self._replica_id

        training_status_obj = TrainingStatus.from_dict(status)
        envelope = EventEnvelope()

        # Use helper to copy fields
        self._copy_training_status_fields(envelope.training_status, training_status_obj)

        await self._publish_envelope(
            torchLoomConstants.subjects.THREADLET_EVENTS,  # Changed subject
            envelope,
            f"TrainingStatus: {training_status_obj.status_type} for {training_status_obj.replica_id} to THREADLET_EVENTS",
        )

    async def _publish_device_status(self, status: Dict[str, Any]) -> None:
        """Publish deviceStatus message to NATS."""
        device_status_obj = deviceStatus.from_dict(status)
        envelope = EventEnvelope()

        # Use helper to copy fields
        self._copy_device_status_fields(envelope.device_status, device_status_obj)

        await self._publish_envelope(
            torchLoomConstants.subjects.THREADLET_EVENTS,  # Changed subject
            envelope,
            f"deviceStatus: {device_status_obj.device_id} status={device_status_obj.status} to THREADLET_EVENTS",
        )

    def _copy_training_status_fields(
        self, protobuf_msg, status_obj: TrainingStatus
    ) -> None:
        """Copy fields from TrainingStatus object to protobuf message."""
        # Define field mappings (protobuf_field_name: object_attribute_name)
        field_mappings = {
            "replica_id": "replica_id",
            "status_type": "status_type",
            "current_step": "current_step",
            "epoch": "epoch",
            "status": "status",
            "training_time": "training_time",
        }

        # Copy simple fields
        self._copy_fields_by_mapping(protobuf_msg, status_obj, field_mappings)

        # Copy metrics map
        protobuf_msg.metrics.update({k: str(v) for k, v in status_obj.metrics.items()})

    def _copy_device_status_fields(
        self, protobuf_msg, status_obj: deviceStatus
    ) -> None:
        """Copy fields from deviceStatus object to protobuf message."""
        # Define field mappings (protobuf_field_name: object_attribute_name)
        field_mappings = {
            "device_id": "device_id",
            "replica_id": "replica_id",
            "server_id": "server_id",
            "status": "status",
            "utilization": "utilization",
            "temperature": "temperature",
            "memory_used": "memory_used",
            "memory_total": "memory_total",
        }

        # Copy simple fields
        self._copy_fields_by_mapping(protobuf_msg, status_obj, field_mappings)

        # Copy config map
        protobuf_msg.config.update({k: str(v) for k, v in status_obj.config.items()})

    def _copy_fields_by_mapping(
        self, protobuf_msg, source_obj, field_mappings: Dict[str, str]
    ) -> None:
        """Generic helper to copy fields based on a mapping dictionary."""
        for protobuf_field, source_attr in field_mappings.items():
            if hasattr(source_obj, source_attr):
                value = getattr(source_obj, source_attr)
                if value is not None:  # Add check for None before setting attribute
                    setattr(protobuf_msg, protobuf_field, value)

    async def _publish_envelope(
        self, subject: str, envelope: EventEnvelope, log_message: str
    ) -> None:
        """Helper method to publish protobuf envelope to NATS with error handling."""
        try:
            # Use SubscriptionManager's nc for publishing
            await self._subscription_manager.nc.publish(
                subject, envelope.SerializeToString()
            )
            self._logger.debug(f"Published {log_message}")
        except Exception as e:
            self._logger.exception(f"Failed to publish to {subject}: {e}")

    async def _monitor_stop_event(self) -> None:
        """Monitor the ASYNC stop event and exit when signaled."""
        await self._async_stop_event.wait()  # Wait for the asyncio event
        self._logger.info(
            "Async stop event detected, shutting down relevant async tasks"
        )

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
