"""
Async threadlet listener implementation.
"""

import asyncio
import json
import multiprocessing
import os
import time
import threading
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Optional

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from multiprocessing import Event

from nats.aio.client import Client
from nats.aio.msg import Msg
from nats.js.api import RetentionPolicy, StorageType, StreamConfig
from nats.js.client import JetStreamContext

import nats
from torchLoom.common import deviceStatus, TrainingStatus
from torchLoom.common.config import Config
from torchLoom.common.constants import WeaverStream, torchLoomConstants
from torchLoom.common.utils import cancel_subscriptions, get_device_uuid
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope, RegisterDevice

from .message import (
    MessageType, 
    MessageFactory, 
    serialize_message, 
    deserialize_message,
    CommandType
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

        # NATS connection setup
        self._torchLoom_addr = torchLoom_addr
        self._nc: Optional[Client] = None
        self._js: Optional[JetStreamContext] = None
        self._subscriptions: Dict[str, Any] = {}

        # Inter-process communication using a single duplex pipe
        self._pipe_to_main_process = pipe_to_main_process
        self._stop_event = stop_event
        self._pipe_listener_stop_event = threading.Event()

        # Configuration
        self._nc_timeout = Config.NC_TIMEOUT or 1
        self._exception_sleep = Config.EXCEPTION_RETRY_TIME or 1

        self._logger.info(
            f"ThreadletListener initialized with replica_id: {self._replica_id}"
        )

    async def run(self) -> None:
        """Main run loop for the async threadlet listener."""
        try:
            await self._connect()
            await self._setup_subscriptions()
            await self._register_device()

            # Start background tasks
            async_tasks = [
                asyncio.create_task(self._heartbeat_loop()),
                asyncio.create_task(self._monitor_stop_event()),
            ]

            # Start the synchronous pipe listener thread
            self._pipe_listener_thread = threading.Thread(
                target=self._pipe_message_processor_loop,
                name=f"listener-pipe-thread-{self._replica_id}",
                daemon=True
            )
            self._pipe_listener_thread.start()
            self._logger.info("ThreadletListener pipe message processor thread started.")

            self._logger.info(
                "ThreadletListener async initialization completed, starting main loop"
            )

            # Wait for stop signal or task completion
            done, pending = await asyncio.wait(
                async_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    self._logger.debug(f"Async task {task.__name__} cancelled.")
                except Exception as e:
                    self._logger.warning(f"Error during cancellation of async task: {e}")

            self._logger.info("ThreadletListener main loop completed")

        except Exception as e:
            self._logger.exception(f"Error in threadlet listener run loop: {e}")
        finally:
            await self._cleanup()

    async def _connect(self) -> None:
        """Connect to NATS server."""
        try:
            self._nc = await nats.connect(self._torchLoom_addr)
            self._js = self._nc.jetstream()
            self._logger.info(f"Connected to NATS server at {self._torchLoom_addr}")
        except Exception as e:
            self._logger.exception(f"Failed to connect to NATS server: {e}")
            raise

    async def _setup_subscriptions(self) -> None:
        """Set up subscriptions for config updates and other messages."""
        try:
            if not self._js:
                raise RuntimeError("JetStream not initialized")

            # The weaver should have already created the WEAVELET_STREAM with all necessary subjects
            # Just subscribe to existing stream without trying to create or modify it
            self._logger.info(f"Setting up subscriptions to existing stream {WeaverStream.STREAM}")

            await self._subscribe_js(
                stream=WeaverStream.STREAM,
                subject=torchLoomConstants.subjects.CONFIG_INFO,
                consumer=f"threadlet-{self._replica_id}",
                message_handler=self._handle_config_message,
            )

            # Subscribe to replica fail events (regular NATS, not JetStream)
            await self._subscribe_nc(
                subject=torchLoomConstants.subjects.REPLICA_FAIL,
                message_handler=self._handle_replica_fail_message,
            )

            # Subscribe to WEAVER_COMMANDS
            await self._subscribe_js(
                stream=WeaverStream.STREAM,
                subject=torchLoomConstants.subjects.WEAVER_COMMANDS,
                consumer=f"threadlet-{self._replica_id}",
                message_handler=self._handle_weaver_command,
            )

            self._logger.info("ThreadletListener subscriptions set up successfully")
        except Exception as e:
            self._logger.exception(f"Failed to set up subscriptions: {e}")
            raise

    async def _register_device(self) -> None:
        """Register this device with the weaver."""
        try:
            if not self._js:
                raise RuntimeError("JetStream not initialized")
            if not EventEnvelope:
                raise RuntimeError("EventEnvelope not available")

            self._device_uuid = get_device_uuid()

            envelope = EventEnvelope()
            envelope.register_device.device_uuid = self._device_uuid
            envelope.register_device.replica_id = self._replica_id

            await self._js.publish(
                torchLoomConstants.weaver_stream.subjects.DR_SUBJECT,
                envelope.SerializeToString(),
            )

            self._logger.info(
                f"Registered device {self._device_uuid} with replica {self._replica_id}"
            )
        except Exception as e:
            self._logger.exception(f"Failed to register device: {e}")
            raise

    async def _handle_config_message(self, msg: Msg) -> None:
        """Handle incoming configuration update messages."""
        try:
            if not EventEnvelope:
                self._logger.error("EventEnvelope not available, cannot parse message")
                return

            envelope = EventEnvelope()
            envelope.ParseFromString(msg.data)

            if envelope.HasField("config_info"):
                params = dict(envelope.config_info.config_params)
                self._logger.info(f"Received config update: {params}")

                # Send config updates to the main process via pipe using structured messages
                try:
                    config_message = MessageFactory.create_config(
                        replica_id=self._replica_id,
                        config_params=params
                    )
                    self._send_message_to_threadlet(config_message)
                    self._logger.info(
                        f"Sent config update to main process: {params}"
                    )
                except Exception as e:
                    self._logger.warning(f"Error sending config update via pipe: {e}")

            await msg.ack()
        except Exception as e:
            self._logger.exception(f"Error handling config message: {e}")

    async def _handle_replica_fail_message(self, msg: Msg) -> None:
        """Handle replica failure notifications."""
        try:
            if not EventEnvelope:
                self._logger.error("EventEnvelope not available, cannot parse message")
                return

            envelope = EventEnvelope()
            envelope.ParseFromString(msg.data)

            if envelope.HasField("replica_fail"):
                failed_replica_id = envelope.replica_fail.replica_id
                if failed_replica_id == self._replica_id:
                    self._logger.warning(
                        f"This replica ({self._replica_id}) has been marked as failed"
                    )
                    # Could trigger recovery logic here
                else:
                    self._logger.info(f"Another replica failed: {failed_replica_id}")
        except Exception as e:
            self._logger.exception(f"Error handling replica fail message: {e}")

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
                        # Convert string command type to enum
                        cmd_type = CommandType(command_type) if command_type in [e.value for e in CommandType] else CommandType.UPDATE_CONFIG
                        
                        command_message = MessageFactory.create_command(
                            replica_id=self._replica_id,
                            command_type=cmd_type,
                            params=params
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

    async def _subscribe_js(
        self,
        stream: str,
        subject: str,
        consumer: str,
        message_handler: Callable[[Msg], Awaitable[None]],
    ) -> None:
        """Subscribe to JetStream subject (assumes stream already exists)."""
        try:
            # Subscribe to the existing stream (weaver should have created it)
            if not self._js:
                raise RuntimeError("JetStream not initialized")
            
            psub = await self._js.pull_subscribe(
                subject, durable=consumer, stream=stream
            )
            self._logger.info(f"Subscribed to {subject} on existing stream {stream} with consumer {consumer}")

            async def listen_to_js_subscription():
                self._logger.info(f"Started listening on JetStream {subject}")
                while not self._stop_event.is_set():
                    try:
                        msgs = await psub.fetch(1, timeout=self._nc_timeout)
                        for msg in msgs:
                            await message_handler(msg)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        self._logger.exception(
                            f"Error in JetStream subscription loop for {subject}: {e}"
                        )
                        await asyncio.sleep(self._exception_sleep)

            task = asyncio.create_task(listen_to_js_subscription())
            self._subscriptions[subject] = (psub, task)
        except Exception as e:
            self._logger.exception(f"Failed to subscribe to JetStream {subject}: {e}")
            raise

    async def _subscribe_nc(
        self, subject: str, message_handler: Callable[[Msg], Awaitable[None]]
    ) -> None:
        """Subscribe to regular NATS subject."""
        try:
            if not self._nc:
                raise RuntimeError("NATS client not initialized")
            sub = await self._nc.subscribe(subject)
            self._logger.info(f"Subscribed to NATS {subject}")

            async def listen_to_nc_subscription():
                self._logger.info(f"Started listening on NATS {subject}")
                while not self._stop_event.is_set():
                    try:
                        msg = await sub.next_msg(timeout=self._nc_timeout)
                        await message_handler(msg)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        self._logger.exception(
                            f"Error in NATS subscription loop for {subject}: {e}"
                        )
                        await asyncio.sleep(self._exception_sleep)

            task = asyncio.create_task(listen_to_nc_subscription())
            self._subscriptions[subject] = (sub, task)
        except Exception as e:
            self._logger.exception(f"Failed to subscribe to NATS {subject}: {e}")
            raise

    async def _heartbeat_loop(self) -> None:
        """Background task to send periodic heartbeat messages to the weaver."""
        self._logger.info("Started heartbeat loop")
        heartbeat_interval = 30.0  # Send heartbeat every 30 seconds

        while not self._stop_event.is_set():
            try:
                await self._send_heartbeat()
                await asyncio.sleep(heartbeat_interval)
            except Exception as e:
                self._logger.exception(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5.0)  # Wait a bit before retrying

    async def _send_heartbeat(self) -> None:
        """Send a heartbeat message to the weaver."""
        try:
            if not self._nc:
                self._logger.warning("Cannot send heartbeat - not connected to NATS")
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
            await self._nc.publish(
                torchLoomConstants.subjects.HEARTBEAT, envelope.SerializeToString()
            )

            self._logger.debug(f"Sent heartbeat for replica {self._replica_id}")

        except Exception as e:
            self._logger.exception(f"Failed to send heartbeat: {e}")

    async def _monitor_stop_event(self) -> None:
        """Monitor the stop event and exit when signaled."""
        while not self._stop_event.is_set():
            await asyncio.sleep(0.1)
        self._logger.info("Stop event detected, shutting down")

    def _pipe_message_processor_loop(self) -> None:
        """Continuously listens for messages on the pipe from the main Threadlet process."""
        self._logger.info("Pipe message processor loop started for ThreadletListener.")
        loop = asyncio.new_event_loop()  # Create a new event loop for this thread
        asyncio.set_event_loop(loop)

        try:
            while not self._pipe_listener_stop_event.is_set():
                if self._pipe_to_main_process.poll(0.1): # Poll with a timeout
                    raw_message = self._pipe_to_main_process.recv()
                    self._logger.debug(f"Received raw message from main process: {raw_message}")
                    
                    # Try to deserialize the message using the new message types
                    message = deserialize_message(raw_message) if isinstance(raw_message, dict) else None
                    
                    if message:
                        # Handle structured messages
                        if message.message_type == MessageType.HEARTBEAT:
                            loop.run_until_complete(self._handle_heartbeat_message(message))
                        elif message.message_type == MessageType.METRICS:
                            loop.run_until_complete(self._handle_metrics_message(message))
                        elif message.message_type == MessageType.STATUS:
                            loop.run_until_complete(self._handle_status_message(message))
                        else:
                            self._logger.warning(f"Received unknown structured message type: {message.message_type}")
                    else:
                        # Fallback to legacy message handling
                        msg_type = raw_message.get("type") if isinstance(raw_message, dict) else None
                        payload = raw_message.get("payload") if isinstance(raw_message, dict) else None

                        if msg_type == "status_update":
                            if payload:
                                # Run _publish_status in this thread's event loop
                                loop.run_until_complete(self._publish_status(payload))
                            else:
                                self._logger.warning("Received status_update with no payload.")
                        else:
                            self._logger.warning(f"Received unknown legacy message type from main pipe: {msg_type}")
        except EOFError:
            self._logger.info("Pipe to main process closed, main process likely terminated.")
        except Exception as e:
            if not self._pipe_listener_stop_event.is_set(): # Log only if not intentionally stopping
                self._logger.exception(f"Error in listener's pipe message processor loop: {e}")
        finally:
            loop.close()
            self._logger.info("Pipe message processor loop stopped for ThreadletListener.")

    async def _handle_heartbeat_message(self, message) -> None:
        """Handle heartbeat message from Threadlet."""
        try:
            # Just log the heartbeat - no need to publish to NATS
            self._logger.debug(f"Received heartbeat from {message.replica_id}: {message.status}")
        except Exception as e:
            self._logger.exception(f"Error handling heartbeat message: {e}")

    async def _handle_metrics_message(self, message) -> None:
        """Handle metrics message from Threadlet."""
        try:
            status_dict = {
                "type": "training_status",
                "status_type": "metrics_update",
                "replica_id": message.replica_id,
                "current_step": message.step,
                "epoch": message.epoch,
                "metrics": {
                    **message.metrics,
                    "loss": message.loss,
                    "accuracy": message.accuracy,
                    "gradient_norm": message.gradient_norm,
                }
            }
            await self._publish_status(status_dict)
            self._logger.debug(f"Handled metrics message from {message.replica_id}")
        except Exception as e:
            self._logger.exception(f"Error handling metrics message: {e}")

    async def _handle_status_message(self, message) -> None:
        """Handle status message from Threadlet."""
        try:
            status_dict = {
                "type": "training_status",
                "status_type": "status_update",
                "replica_id": message.replica_id,
                "current_step": message.current_step,
                "epoch": message.epoch,
                "status": message.status,
                "message": message.message,
            }
            await self._publish_status(status_dict)
            self._logger.debug(f"Handled status message from {message.replica_id}: {message.status}")
        except Exception as e:
            self._logger.exception(f"Error handling status message: {e}")

    def _send_message_to_threadlet(self, message) -> None:
        """Send a structured message to the main Threadlet process."""
        try:
            if self._pipe_to_main_process and not self._pipe_to_main_process.closed:
                serialized_message = serialize_message(message)
                self._pipe_to_main_process.send(serialized_message)
                self._logger.debug(f"Sent message to Threadlet: {message.message_type}")
        except (BrokenPipeError, OSError):
            self._logger.warning("Pipe to main process is broken, dropping message")
        except Exception as e:
            self._logger.warning(f"Error sending message via pipe: {e}")

    async def _publish_status(self, status: Dict[str, Any]) -> None:
        """Publish status to NATS using protobuf messages based on status type."""
        try:
            if not self._nc:
                self._logger.warning("Cannot publish status - not connected to NATS")
                return

            # Import protobuf message types
            if not EventEnvelope:
                self._logger.warning(
                    "EventEnvelope not available, skipping status publish"
                )
                return

            # Determine status type and route accordingly
            status_type = status.get("type", status.get("status_type", "unknown"))

            if status_type == "training_status" or status_type in [
                "training_start",
                "epoch_start",
                "batch_update",
                "epoch_complete",
                "training_complete",
                "test_complete",
                "training_interrupted",
                "training_failed",
            ]:
                await self._publish_training_status(status)
            elif status_type == "device_status":
                await self._publish_device_status(status)
            else:
                # Fallback to training status for backward compatibility
                await self._publish_training_status(status)

        except Exception as e:
            self._logger.exception(f"Failed to publish status: {e}")

    async def _publish_training_status(self, status: Dict[str, Any]) -> None:
        """Publish TrainingStatus message to NATS."""
        try:
            # Create TrainingStatus from dictionary
            if "replica_id" not in status:
                status["replica_id"] = self._replica_id

            training_status_obj = TrainingStatus.from_dict(status)

            # Create protobuf message
            envelope = EventEnvelope()
            training_status = envelope.training_status
            training_status.replica_id = training_status_obj.replica_id
            training_status.status_type = training_status_obj.status_type
            training_status.current_step = training_status_obj.current_step
            training_status.epoch = training_status_obj.epoch
            training_status.step_progress = training_status_obj.step_progress
            training_status.epoch_progress = training_status_obj.epoch_progress
            training_status.status = training_status_obj.status
            training_status.training_time = training_status_obj.training_time
            training_status.batch_idx = training_status_obj.batch_idx

            # Add metrics
            for key, value in training_status_obj.metrics.items():
                training_status.metrics[key] = str(value)

            # Publish to NATS
            await self._nc.publish(
                torchLoomConstants.subjects.TRAINING_STATUS,
                envelope.SerializeToString(),
            )

            self._logger.debug(
                f"Published TrainingStatus: {training_status_obj.status_type} for {training_status_obj.replica_id}"
            )

        except Exception as e:
            self._logger.exception(f"Failed to publish training status: {e}")

    async def _publish_device_status(self, status: Dict[str, Any]) -> None:
        """Publish deviceStatus message to NATS."""
        try:
            # Create deviceStatus from dictionary
            device_status_obj = deviceStatus.from_dict(status)

            # Create protobuf message
            envelope = EventEnvelope()
            device_status = envelope.device_status
            device_status.device_id = device_status_obj.device_id
            device_status.replica_id = device_status_obj.replica_id
            device_status.server_id = device_status_obj.server_id
            device_status.status = device_status_obj.status
            device_status.utilization = device_status_obj.utilization
            device_status.temperature = device_status_obj.temperature
            device_status.memory_used = device_status_obj.memory_used
            device_status.memory_total = device_status_obj.memory_total

            # Add configuration parameters
            for key, value in device_status_obj.config.items():
                device_status.config[key] = str(value)

            # Publish to NATS
            await self._nc.publish(
                torchLoomConstants.subjects.device_STATUS, envelope.SerializeToString()
            )

            self._logger.debug(
                f"Published deviceStatus: {device_status_obj.device_id} status={device_status_obj.status}"
            )

        except Exception as e:
            self._logger.exception(f"Failed to publish device status: {e}")

    async def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._logger.info("Starting ThreadletListener cleanup...")

            # Signal and wait for the pipe listener thread to stop
            if hasattr(self, '_pipe_listener_thread') and self._pipe_listener_thread.is_alive():
                self._logger.info("Stopping ThreadletListener pipe message processor thread...")
                self._pipe_listener_stop_event.set()
                self._pipe_listener_thread.join(timeout=2)
                if self._pipe_listener_thread.is_alive():
                    self._logger.warning("ThreadletListener pipe processor thread did not stop in time.")
                else:
                    self._logger.info("ThreadletListener pipe processor thread stopped.")

            # Cancel all subscriptions and wait for them to complete
            for subject, (sub_or_psub, task) in self._subscriptions.items(): # Adjusted to handle psub correctly
                try:
                    self._logger.info(f"Cleaning up subscription for {subject}")

                    # Cancel the task first
                    if task and not task.done(): # Check if task exists and is not done
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            self._logger.debug(f"Async task for {subject} cancelled.")
                        except Exception as e_task:
                            self._logger.warning(f"Error awaiting cancelled task for {subject}: {e_task}")

                    # Unsubscribe and drain if it's a subscription object
                    # For pull subscriptions (psub), they don't have unsubscribe, but tasks are cancelled.
                    # For regular NATS subscriptions (sub), they have unsubscribe.
                    if hasattr(sub_or_psub, "unsubscribe"):
                        await sub_or_psub.unsubscribe()
                        self._logger.debug(f"Unsubscribed from {subject}.")
                    # Pull subscriptions are managed by their fetch tasks, which are cancelled above.
                    # JetStream pull subscriptions themselves don't have a direct .unsubscribe() or .drain()
                    # in the same way as core NATS subscriptions or how JetStream push subscriptions might.
                    # The client library handles cleanup when the connection is closed.

                except Exception as e:
                    self._logger.warning(
                        f"Error cleaning up subscription {subject}: {e}"
                    )

            self._subscriptions.clear()

            # Close pipe connections
            try:
                if self._pipe_to_main_process and not self._pipe_to_main_process.closed:
                    self._pipe_to_main_process.close()
                    self._logger.info("Pipe to main process closed")
            except Exception as e:
                self._logger.warning(f"Error closing pipe to main process: {e}")

            # Close NATS connection more thoroughly
            if self._nc and not self._nc.is_closed:
                self._logger.info("Draining and closing NATS connection...")
                try:
                    # Drain the connection first to ensure all pending messages are sent
                    await self._nc.drain()
                except Exception as e:
                    self._logger.warning(f"Error draining NATS connection: {e}")

                try:
                    # Then close the connection
                    await self._nc.close()
                except Exception as e:
                    self._logger.warning(f"Error closing NATS connection: {e}")

            self._logger.info("ThreadletListener cleanup completed")
        except Exception as e:
            self._logger.exception(f"Error during cleanup: {e}")
