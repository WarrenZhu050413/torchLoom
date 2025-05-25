"""
Async threadlet listener implementation.
"""

import asyncio
import json
import multiprocessing
import os
import time
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

logger = setup_logger(name="threadlet_listener")


class ThreadletListener:
    """Async implementation of threadlet listener that runs inside the subprocess."""

    def __init__(
        self,
        replica_id: str,
        torchLoom_addr: str,
        config_sender: "Connection",
        status_receiver: "Connection",
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

        # Inter-process communication using pipes
        self._config_sender = config_sender  # Send config updates to main process
        self._status_receiver = (
            status_receiver  # Receive status updates from main process
        )
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
            await self._connect()
            await self._setup_subscriptions()
            await self._register_device()

            # Start background tasks
            tasks = [
                asyncio.create_task(self._status_publisher_loop()),
                asyncio.create_task(self._heartbeat_loop()),
                asyncio.create_task(self._monitor_stop_event()),
            ]

            self._logger.info(
                "ThreadletListener async initialization completed, starting main loop"
            )

            # Wait for stop signal or task completion
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

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

                # Send config updates to the main process via pipe
                try:
                    if self._config_sender and not self._config_sender.closed:
                        self._config_sender.send(params)
                        self._logger.info(
                            f"Sent config update to main process: {params}"
                        )
                except (BrokenPipeError, OSError):
                    self._logger.warning("Config pipe is broken, dropping update")
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

                    # Send command to main process via config pipe
                    command_data = {
                        "command_type": command_type,
                        "params": params,
                        "_is_weaver_command": True,  # Flag to distinguish from config updates
                    }

                    try:
                        if self._config_sender and not self._config_sender.closed:
                            self._config_sender.send(command_data)
                            self._logger.info(
                                f"Sent weaver command to main process: {command_type}"
                            )
                    except (BrokenPipeError, OSError):
                        self._logger.warning(
                            "Config pipe is broken, dropping weaver command"
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

    async def _status_publisher_loop(self) -> None:
        """Background task to publish training status updates."""
        self._logger.info("Started status publisher loop")
        while not self._stop_event.is_set():
            try:
                # Check for status updates from main process using non-blocking poll
                try:
                    if self._status_receiver and not self._status_receiver.closed:
                        if self._status_receiver.poll(0):  # Non-blocking check
                            status = self._status_receiver.recv()
                            await self._publish_status(status)
                except EOFError:
                    # Pipe has been closed from the other end
                    self._logger.info("Status pipe closed, stopping status publisher")
                    break
                except (BrokenPipeError, OSError):
                    # Pipe is broken
                    self._logger.warning(
                        "Status pipe broken, stopping status publisher"
                    )
                    break
                except Exception as e:
                    self._logger.warning(f"Error receiving status from pipe: {e}")

                # Sleep briefly to avoid busy waiting
                await asyncio.sleep(0.1)
            except Exception as e:
                self._logger.exception(f"Error in status publisher loop: {e}")
                await asyncio.sleep(self._exception_sleep)

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

    async def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._logger.info("Starting ThreadletListener cleanup...")

            # Cancel all subscriptions and wait for them to complete
            for subject, (sub, task) in self._subscriptions.items():
                try:
                    self._logger.info(f"Cleaning up subscription for {subject}")

                    # Cancel the task first
                    if not task.cancelled():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                    # Unsubscribe and drain if it's a subscription object
                    if hasattr(sub, "unsubscribe"):
                        await sub.unsubscribe()
                    elif hasattr(sub, "drain"):
                        await sub.drain()

                except Exception as e:
                    self._logger.warning(
                        f"Error cleaning up subscription {subject}: {e}"
                    )

            self._subscriptions.clear()

            # Close pipe connections
            try:
                if self._config_sender and not self._config_sender.closed:
                    self._config_sender.close()
                    self._logger.info("Config sender pipe closed")
            except Exception as e:
                self._logger.warning(f"Error closing config sender pipe: {e}")

            try:
                if self._status_receiver and not self._status_receiver.closed:
                    self._status_receiver.close()
                    self._logger.info("Status receiver pipe closed")
            except Exception as e:
                self._logger.warning(f"Error closing status receiver pipe: {e}")

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
