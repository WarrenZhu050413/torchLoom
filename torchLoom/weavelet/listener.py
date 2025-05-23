"""
Async weavelet listener implementation.
"""

import asyncio
import multiprocessing
import json
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from nats.aio.client import Client
from nats.aio.msg import Msg
from nats.js.api import RetentionPolicy, StorageType, StreamConfig
from nats.js.client import JetStreamContext

import nats
from torchLoom.config import Config
from torchLoom.constants import torchLoomConstants
from torchLoom.log.logger import setup_logger
try:
    from torchLoom.proto.torchLoom_pb2 import EventEnvelope
except ImportError:
    # Handle case where protobuf isn't generated yet
    EventEnvelope = None
from torchLoom.utils import cancel_subscriptions, get_device_uuid


class WeaveletListener:
    """Async implementation of weavelet listener that runs inside the subprocess."""

    def __init__(
        self,
        replica_id: str,
        torchLoom_addr: str,
        config_queue: "multiprocessing.Queue[Any]",
        status_queue: "multiprocessing.Queue[Any]",
        stop_event: "multiprocessing.Event",
    ):
        # Setup logging
        self._logger = setup_logger(
            name="weavelet_logger",
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

        # Inter-process communication
        self._config_queue = config_queue
        self._status_queue = status_queue
        self._stop_event = stop_event

        # Configuration
        self._nc_timeout = Config.NC_TIMEOUT or 1
        self._exception_sleep = Config.EXCEPTION_RETRY_TIME or 1

        self._logger.info(
            f"WeaveletListener initialized with replica_id: {self._replica_id}"
        )

    async def run(self) -> None:
        """Main run loop for the async weavelet listener."""
        try:
            await self._connect()
            await self._setup_subscriptions()
            await self._register_device()

            # Start background tasks
            tasks = [
                asyncio.create_task(self._status_publisher_loop()),
                asyncio.create_task(self._monitor_stop_event()),
            ]

            self._logger.info(
                "WeaveletListener async initialization completed, starting main loop"
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

            self._logger.info("WeaveletListener main loop completed")

        except Exception as e:
            self._logger.exception(f"Error in weavelet listener run loop: {e}")
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
                
            # Subscribe to config updates
            await self._js.add_stream(
                StreamConfig(
                    name="WEAVELET_STREAM",
                    subjects=[torchLoomConstants.subjects.CONFIG_INFO],
                )
            )

            await self._subscribe_js(
                stream="WEAVELET_STREAM",
                subject=torchLoomConstants.subjects.CONFIG_INFO,
                consumer=f"weavelet-{self._replica_id}",
                message_handler=self._handle_config_message,
            )

            # Subscribe to replica fail events
            await self._subscribe_nc(
                subject=torchLoomConstants.subjects.REPLICA_FAIL,
                message_handler=self._handle_replica_fail_message,
            )

            self._logger.info("WeaveletListener subscriptions set up successfully")
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

                # Send config updates to the main process via queue
                try:
                    self._config_queue.put_nowait(params)
                    self._logger.info(f"Sent config update to main process: {params}")
                except:
                    self._logger.warning("Config queue is full, dropping update")

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

    async def _subscribe_js(
        self,
        stream: str,
        subject: str,
        consumer: str,
        message_handler: Callable[[Msg], Awaitable[None]],
    ) -> None:
        """Subscribe to JetStream subject with proper stream configuration."""
        try:
            # Ensure stream exists with proper configuration
            try:
                stream_config = StreamConfig(
                    name=stream,
                    subjects=[subject, f"{subject}.*"],  # Allow subject patterns
                    retention=RetentionPolicy.LIMITS,  # Retain based on limits
                    max_msgs=10000,  # Maximum number of messages to retain
                    max_bytes=10 * 1024 * 1024,  # 10MB max stream size
                    max_age=3600,  # 1 hour message retention
                    storage=StorageType.FILE,  # Use file storage for persistence
                    num_replicas=1,  # Single replica for simplicity
                )
                
                # Create or update the stream
                try:
                    if not self._js:
                        raise RuntimeError("JetStream not initialized")
                    await self._js.add_stream(stream_config)
                    self._logger.info(f"Created/updated JetStream {stream}")
                except Exception as e:
                    # Stream might already exist with different config
                    self._logger.info(f"Stream {stream} already exists or update failed: {e}")
                    
            except Exception as e:
                self._logger.warning(f"Could not configure stream {stream}: {e}")
                # Continue with subscription attempt even if stream config failed
            
            # Subscribe to the stream
            if not self._js:
                raise RuntimeError("JetStream not initialized")
            psub = await self._js.pull_subscribe(
                subject, durable=consumer, stream=stream
            )
            self._logger.info(f"Subscribed to JetStream {subject} on stream {stream}")

            async def listen_to_js_subscription():
                self._logger.info(f"Started listening on JetStream {subject}")
                while not self._stop_event.is_set():
                    try:
                        msgs = await psub.fetch(1, timeout=1)
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
                # Check for status updates from main process
                try:
                    status = self._status_queue.get_nowait()
                    await self._publish_status(status)
                except:
                    # No status update available, continue
                    pass

                # Sleep briefly to avoid busy waiting
                await asyncio.sleep(0.1)
            except Exception as e:
                self._logger.exception(f"Error in status publisher loop: {e}")
                await asyncio.sleep(self._exception_sleep)

    async def _publish_status(self, status: Dict[str, Any]) -> None:
        """Publish training status to NATS."""
        try:
            if not self._nc:
                self._logger.warning("Cannot publish status - not connected to NATS")
                return

            # Import constants for subject names
            from torchLoom.constants import torchLoomConstants
            
            # Create appropriate protobuf messages based on status type
            if not EventEnvelope:
                self._logger.warning("EventEnvelope not available, skipping status publish")
                return

            # Extract basic information
            replica_id = status.get('replica_id', self._replica_id)
            device_id = status.get('device_id', f"device_{replica_id}")
            status_type = status.get('type', 'training_metrics')
            
            # Publish to UI_STATUS_UPDATE for general UI consumption (JSON format)
            ui_message = {
                'type': status_type,
                'replica_id': replica_id,
                'device_id': device_id,
                'timestamp': status.get('timestamp', time.time()),
                **status  # Include all original status data
            }
            
            await self._nc.publish(
                torchLoomConstants.subjects.UI_STATUS_UPDATE,
                json.dumps(ui_message).encode('utf-8')
            )
            self._logger.debug(f"Published UI status: {status_type} for {replica_id}")
            
            # Also publish specific protobuf messages for structured handling
            if status_type in ['training_metrics', 'epoch_complete', 'epoch_start']:
                # Create TrainingProgress protobuf message
                envelope = EventEnvelope()
                progress = envelope.training_progress
                progress.replica_id = replica_id
                progress.current_step = status.get('epoch', 0)
                progress.step_progress = status.get('step_progress', 0.0)
                progress.status = "training" if status_type != 'training_complete' else "completed"
                progress.last_active_step = status.get('step', status.get('batch_idx', 0))
                
                await self._nc.publish(
                    torchLoomConstants.subjects.TRAINING_PROGRESS,
                    envelope.SerializeToString()
                )
                
                # Create GPUStatus protobuf message if system metrics are available
                if 'system' in status:
                    envelope_gpu = EventEnvelope()
                    gpu_status = envelope_gpu.gpu_status
                    gpu_status.gpu_id = device_id
                    gpu_status.replica_id = replica_id
                    gpu_status.server_id = status.get('server_id', 'local_server')
                    gpu_status.status = "active"
                    
                    system_metrics = status['system']
                    gpu_status.utilization = system_metrics.get('gpu_utilization', 0.0)
                    gpu_status.temperature = 40.0  # Default temperature
                    
                    # Add configuration parameters
                    if 'config' in status:
                        config_data = status['config']
                        if isinstance(config_data, dict):
                            for key, value in config_data.items():
                                gpu_status.config[key] = str(value)
                    
                    # Add current learning rate and other metrics
                    if 'learning_rate' in status:
                        gpu_status.config['learning_rate'] = str(status['learning_rate'])
                    if 'loss' in status:
                        gpu_status.config['current_loss'] = str(status['loss'])
                    if 'accuracy' in status:
                        gpu_status.config['current_accuracy'] = str(status['accuracy'])
                    
                    await self._nc.publish(
                        torchLoomConstants.subjects.GPU_STATUS,
                        envelope_gpu.SerializeToString()
                    )
                    
            self._logger.debug(f"Published training status: {status_type} for {replica_id}")
            
        except Exception as e:
            self._logger.exception(f"Failed to publish training status: {e}")

    async def _monitor_stop_event(self) -> None:
        """Monitor the stop event and exit when signaled."""
        while not self._stop_event.is_set():
            await asyncio.sleep(0.1)
        self._logger.info("Stop event detected, shutting down")

    async def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._logger.info("Starting WeaveletListener cleanup...")
            
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
                    if hasattr(sub, 'unsubscribe'):
                        await sub.unsubscribe()
                    elif hasattr(sub, 'drain'):
                        await sub.drain()
                        
                except Exception as e:
                    self._logger.warning(f"Error cleaning up subscription {subject}: {e}")
            
            self._subscriptions.clear()

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

            self._logger.info("WeaveletListener cleanup completed")
        except Exception as e:
            self._logger.exception(f"Error during cleanup: {e}") 