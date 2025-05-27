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
from torchLoom.common.constants import Config, TimeConstants, WeaverOutgressStream, NatsConstants
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
    """Async implementation of threadlet listener that runs inside the subprocess.

    This class automatically sends heartbeats at regular intervals via its internal
    heartbeat loop. The main threadlet process cannot control heartbeat timing,
    ensuring consistent heartbeat delivery regardless of the main process state.
    """

    def __init__(
        self,
        process_id: str,
        torchLoom_addr: str,
        pipe_to_main_process: "Connection",
        stop_event: "Event",
    ):
        # Setup logging
        self._logger = setup_logger(
            name="threadlet_logger",
            log_file=LoggerConstants.MANAGER_torchLoom_LOG_FILE,
            format_log=True,
            print_to_console=True,
        )

        # Core identifiers
        self._process_id = process_id
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
        self._nc_timeout = TimeConstants.PIPE_POLL_INTERVAL
        self._exception_sleep = (
            TimeConstants.ERROR_RETRY_SLEEP
        )

        self._logger.info(
            f"ThreadletListener initialized with process_id: {self._process_id}"
        )

    async def run(self) -> None:
        """Main run loop for the async threadlet listener."""
        self._logger.info("ThreadletListener run() method started.")
        try:
            # Start a task to monitor the multiprocessing stop_event and set the asyncio stop_event
            mp_event_monitor_task = asyncio.create_task(self._monitor_mp_stop_event())

            await self._subscription_manager.initialize()  # Replaces self._connect()
            self._logger.info("Subscription manager initialized.")

            # Initialize the common event publisher
            self._event_publisher = EventPublisher(
                nats_client=self._subscription_manager.nc,
                js_client=self._subscription_manager.js,
            )
            self._logger.info("Event publisher initialized.")

            await self._setup_subscriptions_with_manager()  # New method using SubscriptionManager
            self._logger.info("Subscriptions set up.")
            await self._register_device()
            self._logger.info("Device registered.")

            # Initialize the threadlet-specific publisher after device registration
            if self._device_uuid:
                self._threadlet_publisher = ThreadletEventPublisher(
                    process_id=self._process_id,
                    device_uuid=self._device_uuid,
                    event_publisher=self._event_publisher,
                )
                self._logger.info("Threadlet publisher initialized.")
            else:
                self._logger.warning("Device UUID not available, Threadlet publisher NOT initialized.")

            # Start background tasks including the async pipe listener
            async_tasks = [
                asyncio.create_task(self._heartbeat_loop(), name="heartbeat_loop_task"),
                asyncio.create_task(self._async_pipe_message_processor(), name="pipe_processor_task"),
            ]

            self._logger.info(
                "ThreadletListener async initialization completed, starting main loop. Waiting on tasks..."
            )

            # Wait for stop signal or task completion
            done, pending = await asyncio.wait(
                async_tasks + [mp_event_monitor_task],  # Add the new monitor task
                return_when=asyncio.FIRST_COMPLETED,
            )

            self._logger.info(f"asyncio.wait completed. Done tasks: {[t.get_name() for t in done if hasattr(t, 'get_name')]}. Pending tasks: {[t.get_name() for t in pending if hasattr(t, 'get_name')]}.")
            for task in done:
                try:
                    # If a task raised an exception, it would be here if not handled inside the task
                    exc = task.exception()
                    if exc:
                        self._logger.error(f"Task {task.get_name()} raised an exception: {exc}", exc_info=exc)
                except asyncio.InvalidStateError:
                    self._logger.debug(f"Task {task.get_name()} state was invalid, could not get exception (might be okay if cancelled).")

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
            self._logger.exception(f"Unhandled exception in ThreadletListener run() loop: {e}")
        finally:
            # Ensure async_stop_event is set if cleanup is triggered by mp_event_monitor_task completion
            self._logger.info("ThreadletListener run() method entering finally block.")
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
                f"Setting up subscriptions to existing stream {WeaverOutgressStream.STREAM} using SubscriptionManager"
            )

            # Subscribe to WEAVER_COMMANDS
            await self._subscription_manager.subscribe_js(
                stream=WeaverOutgressStream.STREAM,
                subject=NatsConstants.subjects.WEAVER_COMMANDS,
                consumer=f"threadlet-{self._process_id}",
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
                process_id=self._process_id,
            )

            self._logger.info(
                f"Registered device {self._device_uuid} with replica {self._process_id}"
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
                target_process_id = weaver_command.target_process_id
                params = dict(weaver_command.params) if weaver_command.params else {}

                # Only handle commands targeting this replica
                if target_process_id == self._process_id:
                    self._logger.info(
                        f"Received weaver command: {command_type} with params: {params}"
                    )

                    # Send command to main process via pipe using structured messages
                    try:
                        # Convert string command type to protobuf enum value
                        cmd_type = command_type  # Use the string directly since create_command handles conversion

                        command_message = MessageFactory.create_command(
                            process_id=self._process_id,
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
                        f"Ignoring command for different replica: {target_process_id}"
                    )

        except Exception as e:
            self._logger.exception(f"Error handling weaver command: {e}")

    async def _heartbeat_loop(self) -> None:
        """Background task to send periodic heartbeat messages to the weaver.

        This loop runs continuously and automatically sends heartbeats at regular intervals.
        The main threadlet process cannot control heartbeat timing - this ensures consistent
        heartbeat delivery regardless of the main process state.
        """
        self._logger.info(
            f"Started automatic heartbeat loop (interval: {TimeConstants.HEARTBEAT_SEND_INTERVAL}s)"
        )

        heartbeat_count = 0
        while not self._async_stop_event.is_set():  # Use asyncio stop event
            try:
                await self._send_heartbeat()
                heartbeat_count += 1

                # Log heartbeat count periodically
                if heartbeat_count % 10 == 0:
                    self._logger.debug(f"Sent {heartbeat_count} automatic heartbeats")

                await asyncio.sleep(TimeConstants.HEARTBEAT_SEND_INTERVAL)
            except Exception as e:
                self._logger.exception(f"Error in automatic heartbeat loop: {e}")
                await asyncio.sleep(TimeConstants.ERROR_RETRY_SLEEP)

        self._logger.info(
            f"Automatic heartbeat loop stopped after {heartbeat_count} heartbeats"
        )

    async def _send_heartbeat(self) -> None:
        """Send an automatic heartbeat message to the weaver using the threadlet publisher.

        This is called automatically by the heartbeat loop and cannot be triggered
        manually by the main threadlet process.
        """
        try:
            if not self._threadlet_publisher:
                self._logger.warning(
                    "Cannot send automatic heartbeat - threadlet publisher not initialized"
                )
                return

            # Create metadata with additional automatic heartbeat info
            metadata = {
                "process_id": str(os.getpid()) if hasattr(os, "getpid") else "unknown",
                "nats_addr": self._torchLoom_addr,
                "heartbeat_type": "automatic",
                "timestamp": time.time(),
            }

            await self._threadlet_publisher.publish_heartbeat(
                status="active",
                metadata=metadata,
            )

            self._logger.debug(
                f"Sent automatic heartbeat for replica {self._process_id}"
            )

        except Exception as e:
            self._logger.exception(f"Failed to send automatic heartbeat: {e}")

    async def _async_pipe_message_processor(self) -> None:
        """Asynchronously processes messages from the pipe connected to the main threadlet process."""
        self._logger.info("Async pipe message processor loop started.")
        try:
            while not self._async_stop_event.is_set():
                # Use asyncio.to_thread to run the blocking recv in a separate thread
                # self._logger.debug("Polling pipe for data...") # Too verbose
                if self._pipe_to_main_process.poll(0.01):  # Non-blocking poll
                    raw_message = await asyncio.to_thread(
                        self._pipe_to_main_process.recv
                    )
                    self._logger.info(f"Received raw data from pipe: {type(raw_message)}")
                    if raw_message:
                        await self._process_pipe_message(raw_message)
                    else:
                        self._logger.warning("Received None from pipe, might indicate EOF or error.")
                        # Consider a small sleep or break if pipe seems closed
                        await asyncio.sleep(0.1)
                else:
                    # If no data, yield control to the event loop
                    await asyncio.sleep(TimeConstants.ASYNC_PIPE_POLL_INTERVAL) # Use a defined constant

        except EOFError:
            self._logger.info(
                "Pipe closed (EOFError), main threadlet process likely terminated."
            )
        except (BrokenPipeError, OSError) as bpe:
            self._logger.warning(f"Pipe error in listener: {bpe}")
        except Exception as e:
            if (
                not self._async_stop_event.is_set()
            ):  # Log only if not intentionally stopping
                self._logger.exception(
                    f"Error in async pipe message processor loop: {e}"
                )
        finally:
            self._logger.info("Async pipe message processor loop stopped.")

    async def _process_pipe_message(self, raw_message) -> None:
        """Deserializes and processes a single message received from the pipe."""
        self._logger.debug(f"Processing raw message from pipe: {type(raw_message)}")
        message = None
        try:
            if isinstance(raw_message, tuple) and len(raw_message) == 2:
                message_type_str, serialized_bytes = raw_message
                if isinstance(serialized_bytes, bytes) and isinstance(
                    message_type_str, str
                ):
                    message = deserialize_message(serialized_bytes, message_type_str)
                    self._logger.info(f"Deserialized message: type={message.message_type}, content_keys={list(message.content.keys()) if hasattr(message, 'content') and message.content else 'N/A'}")
                else:
                    self._logger.warning(
                        f"Received malformed tuple from pipe: types were {type(message_type_str)}, {type(serialized_bytes)}"
                    )
            else:
                self._logger.warning(
                    f"Received unexpected data type from pipe: {type(raw_message)}"
                )

            if not message:
                self._logger.warning("Failed to deserialize message from pipe.")
                return

            # Determine message type and handle accordingly
            if message.message_type == MessageType.TRAINING_STATUS.value:
                self._logger.info("Received TRAINING_STATUS message from pipe, handling...")
                await self._handle_training_status_message(message) # message is PipeTrainingStatusMessage
            elif message.message_type == MessageType.DEVICE_STATUS.value:
                self._logger.info("Received DEVICE_STATUS message from pipe, handling...")
                await self._handle_device_status_message(message) # message is PipeDeviceStatusMessage
            elif message.message_type == MessageType.COMMAND.value:
                self._logger.debug(f"Received COMMAND message from pipe, command_type_enum: {message.command_type}, params: {message.params}")
                # Further check for specific command types, e.g., status updates
                # MessageFactory.create_status sets params["_command_type"] = "STATUS"
                # and uses torchLoom_pb2.UPDATE_CONFIG as the actual protobuf command_type
                if message.command_type == torchLoom_pb2.UPDATE_CONFIG and \
                   message.params.get("_command_type") == "STATUS":
                    self._logger.info("Identified STATUS update within COMMAND message, handling...")
                    await self._handle_status_message(message) # message is PipeCommandMessage
                else:
                    # Generic command handling or log as unhandled specific command
                    self._logger.warning(
                        f"Received COMMAND message with command_type {message.command_type} and params {message.params}, but no specific handler for this command's details."
                    )
            # Add other message types if necessary (e.g., HEARTBEAT_REQUEST from main process)
            else:
                self._logger.warning(
                    f"Received unknown or unhandled message type {message.message_type} from pipe."
                )
        except Exception as e:
            self._logger.exception(f"Error processing pipe message: {e}")

    async def _handle_training_status_message(self, message: torchLoom_pb2.PipeTrainingStatusMessage) -> None:
        """Handles training status messages received from the main threadlet process."""
        self._logger.debug(f"_handle_training_status_message invoked with: {message.training_status}")
        try:
            if not self._threadlet_publisher:
                self._logger.warning(
                    "Threadlet publisher not initialized, cannot send training status."
                )
                return

            # Extract training status details from the TrainingStatus protobuf message
            training_status = message.training_status
            current_step = training_status.current_step
            epoch = training_status.epoch
            metrics = dict(training_status.metrics)  # Convert protobuf map to dict
            training_time = training_status.training_time
            config = dict(training_status.config)  # Convert protobuf config map to dict
            
            # Extract message from metrics if available
            status_message_text = metrics.get("message", "")
            
            self._logger.info(f"Publishing training status update: step={current_step}, epoch={epoch}")

            # Create status_data dictionary for the publisher
            status_data = {
                "status_type": "batch_update",  # Default status type
                "current_step": current_step,
                "epoch": epoch,
                "status": "training",  # Default status since it's no longer in protobuf
                "metrics": metrics,
                "training_time": training_time,
                "max_step": training_status.max_step,
                "max_epoch": training_status.max_epoch,
                "config": config,
            }
            
            if status_message_text:
                status_data["message"] = status_message_text

            await self._threadlet_publisher.publish_training_status(status_data)
            self._logger.info(f"Training status update published successfully for step: {current_step}")
        except Exception as e:
            self._logger.exception(f"Error handling training status message: {e}")

    async def _handle_device_status_message(self, message: torchLoom_pb2.PipeDeviceStatusMessage) -> None:
        """Handles device status messages received from the main threadlet process."""
        self._logger.debug(f"_handle_device_status_message invoked with: {message.device_status}")
        try:
            if not self._threadlet_publisher:
                self._logger.warning(
                    "Threadlet publisher not initialized, cannot send device status."
                )
                return

            # Extract device status details from the deviceStatus protobuf message
            device_status = message.device_status
            device_id = device_status.device_id
            process_id = device_status.process_id
            server_id = device_status.server_id
            utilization = device_status.utilization
            temperature = device_status.temperature
            memory_used = device_status.memory_used
            memory_total = device_status.memory_total
            
            self._logger.info(f"Publishing device status update: device_id='{device_id}', utilization={utilization}")

            # Create status_data dictionary for the publisher
            status_data = {
                "device_id": device_id,
                "process_id": process_id,
                "server_id": server_id,
                "utilization": utilization,
                "temperature": temperature,
                "memory_used": memory_used,
                "memory_total": memory_total,
            }

            await self._threadlet_publisher.publish_device_status(status_data)
            self._logger.info(f"Device status update published successfully for device: {device_id}")
        except Exception as e:
            self._logger.exception(f"Error handling device status message: {e}")

    async def _handle_status_message(self, message: torchLoom_pb2.PipeCommandMessage) -> None:
        """Handles status messages (which are specific command messages) received from the main threadlet process."""
        self._logger.debug(f"_handle_status_message invoked with command params: {message.params}")
        try:
            if not self._threadlet_publisher:
                self._logger.warning(
                    "Threadlet publisher not initialized, cannot send status."
                )
                return

            # Extract status details from the command's params map
            # These were set in MessageFactory.create_status
            status_str = message.params.get("status", TrainingStatus.UNKNOWN.value)
            current_step = int(message.params.get("current_step", "0")) # Params are strings
            epoch = int(message.params.get("epoch", "0"))
            status_message_text = message.params.get("message", "")
            
            self._logger.info(f"Preparing to publish status update via ThreadletEventPublisher: status='{status_str}', step={current_step}")

            await self._threadlet_publisher.publish_training_status_update(
                status=status_str, 
                current_step=current_step,
                epoch=epoch,
                message=status_message_text,
            )
            self._logger.info(f"Status update published successfully: {status_str}")
        except Exception as e:
            self._logger.exception(f"Error handling status message: {e}")

    def _send_message_to_threadlet(
        self,
        message: Union[
            torchLoom_pb2.PipeCommandMessage,
            torchLoom_pb2.PipeTrainingStatusMessage,
            torchLoom_pb2.PipeDeviceStatusMessage,
        ],
    ) -> None:
        """Send a structured protobuf message to the main Threadlet process as a (type_str, bytes) tuple."""
        try:
            if self._pipe_to_main_process and not self._pipe_to_main_process.closed:
                message_type_str = ""
                if isinstance(message, torchLoom_pb2.PipeTrainingStatusMessage):
                    message_type_str = "TRAINING_STATUS"
                elif isinstance(message, torchLoom_pb2.PipeDeviceStatusMessage):
                    message_type_str = "DEVICE_STATUS"
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
