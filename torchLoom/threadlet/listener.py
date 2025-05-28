"""
Async threadlet listener implementation.
"""

import asyncio
import json
import multiprocessing
import os
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from multiprocessing import Event

from nats.aio.client import Client
from nats.aio.msg import Msg
from nats.js.api import RetentionPolicy, StorageType, StreamConfig
from nats.js.client import JetStreamContext

import nats
from torchLoom.common import TrainingStatus, deviceStatus
from torchLoom.common.constants import (
    LoggerConstants,
    NatsConstants,
    TimeConstants,
    WeaverOutgressStream,
)
from torchLoom.common.subscription import SubscriptionManager
from torchLoom.common.utils import (
    create_device_status_dict,
    create_training_status_dict,
    maybe_get_device_uuid,
)
from torchLoom.log.logger import setup_logger
from torchLoom.proto import torchLoom_pb2
from torchLoom.proto.torchLoom_pb2 import EventEnvelope, RegisterDevice

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
        device_uuid: str,
        server_id: str,
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
        self._device_uuid = device_uuid
        self._server_id = server_id

        # Asyncio stop event for the listener
        self._async_stop_event = asyncio.Event()

        # NATS connection setup via SubscriptionManager
        self._torchLoom_addr = torchLoom_addr
        self._subscription_manager = SubscriptionManager(
            torchLoom_addr=self._torchLoom_addr,
            stop_event=self._async_stop_event,
        )

        self._threadlet_publisher: Optional[ThreadletEventPublisher] = None

        # Inter-process communication using a single duplex pipe
        self._pipe_to_main_process = pipe_to_main_process
        self._stop_event = stop_event

        # constants
        self._nc_timeout = TimeConstants.PIPE_POLL_INTERVAL

        self._logger.info(
            f"ThreadletListener initialized with process_id: {self._process_id}"
        )

    async def run(self) -> None:
        """Main run loop for the async threadlet listener."""
        self._logger.info("ThreadletListener run() method started.")
        try:
            await self._subscription_manager.initialize()
            await self._setup_subscriptions_with_manager()
            self._threadlet_publisher = ThreadletEventPublisher(
                nats_client=self._subscription_manager.nc,
                js_client=self._subscription_manager.js,
                process_id=self._process_id,
                device_uuid=self._device_uuid,
            )
            await self._register_device()
            self._logger.info("Threadlet publisher initialized.")

            async_tasks = [
                asyncio.create_task(self._heartbeat_loop(), name="heartbeat_loop_task"),
                asyncio.create_task(
                    self._async_pipe_message_processor(), name="pipe_processor_task"
                ),
                asyncio.create_task(
                    self._monitor_mp_stop_event(), name="mp_event_monitor_task"
                ),
            ]

            self._logger.info(
                "ThreadletListener async initialization completed, starting main loop. Waiting on tasks..."
            )

            done, pending = await asyncio.wait(
                async_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            self._logger.info(
                f"asyncio.wait completed. Done tasks: {[t.get_name() for t in done if hasattr(t, 'get_name')]}. Pending tasks: {[t.get_name() for t in pending if hasattr(t, 'get_name')]}."
            )

            self._cleanup_futures(done, pending)
        except Exception as e:
            self._logger.exception(
                f"Unhandled exception in ThreadletListener run() loop: {e}"
            )
        finally:
            self._logger.info("ThreadletListener run() method entering finally block.")
            self._async_stop_event.set()
            await self._cleanup()

    async def _cleanup_futures(
        self, done: List[asyncio.Future], pending: List[asyncio.Future]
    ) -> None:
        """Clean up done and pending futures."""
        # Check done task exception
        for task in done:
            try:
                # If a task raised an exception, we handle it here
                exc = task.exception()
                if exc:
                    self._logger.error(
                        f"Task {task.get_name()} raised an exception: {exc}",
                        exc_info=exc,
                    )
            except asyncio.InvalidStateError:
                self._logger.debug(
                    f"Task {task.get_name()} state was invalid, could not get exception (might be okay if cancelled)."
                )

        # Cancel remaining pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                self._logger.debug(f"Async task {task.get_name()} cancelled.")
            except Exception as e:
                self._logger.warning(f"Error during cancellation of async task: {e}")

        self._logger.info("ThreadletListener main loop completed")

    async def _monitor_mp_stop_event(self) -> None:
        """Monitors the multiprocessing.Event and sets the asyncio.Event."""
        self._logger.debug("Started monitoring multiprocessing stop_event.")
        while not self._stop_event.is_set():  # self._stop_event is an mp.Event
            await asyncio.sleep(TimeConstants.MONITOR_STOP_EVENT_SLEEP)
        self._logger.info(
            "Multiprocessing stop_event detected, setting asyncio stop_event."
        )
        self._async_stop_event.set()  # self._async_stop_event is an asyncio.Event

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
            if not self._threadlet_publisher:
                raise RuntimeError("Threadlet publisher not initialized")

            await self._threadlet_publisher.publish_device_registration()

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

                    # Send command to main process via pipe using dictionary format
                    try:
                        command_dict = {
                            "message_type": "command",
                            "command_type": command_type,
                            "payload": params,
                            "process_id": self._process_id,
                        }
                        self._send_dict_to_threadlet(command_dict)
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
        """Background task to send periodic heartbeat messages to the weaver."""
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
                await asyncio.sleep(TimeConstants.EXCEPTION_SLEEP)

        self._logger.info(
            f"Automatic heartbeat loop stopped after {heartbeat_count} heartbeats"
        )

    async def _send_heartbeat(self) -> None:
        """Send an automatic heartbeat message to the weaver using the threadlet publisher."""
        try:
            if not self._threadlet_publisher:
                self._logger.warning(
                    "Cannot send automatic heartbeat - threadlet publisher not initialized"
                )
                return

            # Create metadata with additional automatic heartbeat info
            metadata = {
                "process_id": str(os.getpid()) if hasattr(os, "getpid") else "unknown",
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
                if self._pipe_to_main_process.poll(0.01):
                    # Use asyncio.to_thread to run the blocking recv in a separate thread
                    raw_message = await asyncio.to_thread(
                        self._pipe_to_main_process.recv
                    )
                    self._logger.info(
                        f"Received raw data from pipe: {type(raw_message)}"
                    )
                    self._logger.debug(f"Received raw message from pipe: {raw_message}")
                    if raw_message:
                        await self._process_pipe_message(raw_message)
                    else:
                        self._logger.warning(
                            "Received None from pipe, might indicate EOF or error."
                        )
                        # Consider a small sleep or break if pipe seems closed
                        await asyncio.sleep(0.1)
                else:
                    # If no data, yield control to the event loop
                    await asyncio.sleep(
                        TimeConstants.ASYNC_PIPE_POLL_INTERVAL
                    )  # Use a defined constant

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
        """Processes a dictionary message received from the pipe."""
        self._logger.debug(f"Processing raw message from pipe: {type(raw_message)}")
        try:
            if isinstance(raw_message, dict):
                action = raw_message.get("action")

                if action == "publish_event":
                    # Handle publish event requests from main process
                    await self._handle_publish_event_request(raw_message)
                else:
                    self._logger.warning(f"Unknown action in pipe message: {action}")
            else:
                self._logger.warning(
                    f"Received unexpected data type from pipe: {type(raw_message)}. Expected dict."
                )
        except Exception as e:
            self._logger.exception(f"Error processing pipe message: {e}")

    async def _handle_publish_event_request(self, request_dict: Dict[str, Any]) -> None:
        """Handle a publish event request from the main process."""
        try:
            event_type = request_dict.get("event_type")
            event_data = request_dict.get("event_data", {})

            if not self._threadlet_publisher:
                self._logger.warning(
                    "Threadlet publisher not initialized, cannot publish event."
                )
                return

            self._logger.debug(
                f"Publishing event: {event_type} with data: {event_data}"
            )

            if event_type == "training_status":
                status_data = event_data.get("status_data", {})
                process_id = event_data.get("process_id", self._process_id)
                await self._threadlet_publisher.publish_training_status(status_data)

            elif event_type == "device_status":
                status_data = event_data.get("status_data", {})
                device_uuid = event_data.get("device_uuid", self._device_uuid)
                process_id = event_data.get("process_id", self._process_id)
                await self._threadlet_publisher.publish_device_status(status_data)

            elif event_type == "heartbeat":
                status = event_data.get("status", "active")
                metadata = event_data.get("metadata")
                process_id = event_data.get("process_id", self._process_id)
                device_uuid = event_data.get("device_uuid", self._device_uuid)
                await self._threadlet_publisher.publish_heartbeat(status, metadata)

            elif event_type == "device_registration":
                device_uuid = event_data.get("device_uuid", self._device_uuid)
                process_id = event_data.get("process_id", self._process_id)
                await self._threadlet_publisher.publish_device_registration()

            else:
                self._logger.warning(f"Unknown event type for publishing: {event_type}")

            self._logger.debug(f"Successfully published {event_type} event")

        except Exception as e:
            self._logger.exception(f"Error handling publish event request: {e}")

    async def _handle_training_status_message(
        self, message: torchLoom_pb2.PipeTrainingStatusMessage
    ) -> None:
        """Handles training status messages received from the main threadlet process."""
        self._logger.debug(
            f"_handle_training_status_message invoked with: {message.training_status}"
        )
        try:
            if not self._threadlet_publisher:
                self._logger.warning(
                    "Threadlet publisher not initialized, cannot send training status."
                )
                return

            training_status_proto = message.training_status
            self._logger.info(
                f"Publishing training status update: step={training_status_proto.current_step}, epoch={training_status_proto.epoch}"
            )

            status_data = create_training_status_dict(training_status_proto)

            await self._threadlet_publisher.publish_training_status(status_data)
            self._logger.info(
                f"Training status update published successfully for step: {training_status_proto.current_step}"
            )
        except Exception as e:
            self._logger.exception(f"Error handling training status message: {e}")

    async def _handle_device_status_message(
        self, message: torchLoom_pb2.PipeDeviceStatusMessage
    ) -> None:
        """Handles device status messages received from the main threadlet process."""
        self._logger.debug(
            f"_handle_device_status_message invoked with: {message.device_status}"
        )
        try:
            if not self._threadlet_publisher:
                self._logger.warning(
                    "Threadlet publisher not initialized, cannot send device status."
                )
                return

            device_status_proto = message.device_status
            self._logger.info(
                f"Publishing device status update: device_uuid='{device_status_proto.device_uuid}', utilization={device_status_proto.utilization}"
            )

            status_data = create_device_status_dict(device_status_proto)

            await self._threadlet_publisher.publish_device_status(status_data)
            self._logger.info(
                f"Device status update published successfully for device: {device_status_proto.device_uuid}"
            )
        except Exception as e:
            self._logger.exception(f"Error handling device status message: {e}")

    async def _handle_status_message(
        self, message: torchLoom_pb2.PipeCommandMessage
    ) -> None:
        """Handles status messages (which are specific command messages) received from the main threadlet process."""
        self._logger.debug(
            f"_handle_status_message invoked with command params: {message.params}"
        )
        try:
            if not self._threadlet_publisher:
                self._logger.warning(
                    "Threadlet publisher not initialized, cannot send status."
                )
                return

            # Extract status details from the command's params map
            # These were set in MessageFactory.create_status
            status_str = message.params.get("status", TrainingStatus.UNKNOWN.value)
            current_step = int(
                message.params.get("current_step", "0")
            )  # Params are strings
            epoch = int(message.params.get("epoch", "0"))
            status_message_text = message.params.get("message", "")

            self._logger.info(
                f"Preparing to publish status update via ThreadletEventPublisher: status='{status_str}', step={current_step}"
            )

            await self._threadlet_publisher.publish_training_status_update(
                status=status_str,
                current_step=current_step,
                epoch=epoch,
                message=status_message_text,
            )
            self._logger.info(f"Status update published successfully: {status_str}")
        except Exception as e:
            self._logger.exception(f"Error handling status message: {e}")

    def _send_dict_to_threadlet(self, message_dict: Dict[str, Any]) -> None:
        """Send a dictionary message to the main Threadlet process."""
        try:
            if self._pipe_to_main_process and not self._pipe_to_main_process.closed:
                self._pipe_to_main_process.send(message_dict)
                self._logger.debug(
                    f"Sent dict message to Threadlet via pipe: {message_dict.get('message_type', 'unknown')}"
                )
        except (BrokenPipeError, OSError):
            self._logger.warning(
                "Pipe to main process is broken or closed, dropping message."
            )
        except Exception as e:
            self._logger.warning(
                f"Error sending dict message to Threadlet via pipe: {e}"
            )

    async def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._logger.info("Starting ThreadletListener cleanup...")
            self._async_stop_event.set()

            if self._subscription_manager:
                await self._subscription_manager.close()
                self._logger.info("SubscriptionManager closed.")

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
