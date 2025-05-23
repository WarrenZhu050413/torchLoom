import asyncio
import inspect
import multiprocessing
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional, Type, Union

from nats.aio.client import Client
from nats.aio.msg import Msg
from nats.js.api import StreamConfig
from nats.js.client import JetStreamContext

import nats
from torchLoom.config import Config
from torchLoom.constants import JS, NC, torchLoomConstants
from torchLoom.log.log_utils import log_and_raise_exception
from torchLoom.log.logger import setup_logger
from torchLoom.torchLoom_pb2 import EventEnvelope
from torchLoom.utils import cancel_subscriptions, get_device_uuid


class Weavelet:
    """Process-based Weavelet for torchLoom training processes.

    This class manages all communication between training processes and the weaver,
    including receiving configuration updates and sending training status updates.
    It runs in a separate process using multiprocessing.Process and supports
    decorator-based handler registration for automatic configuration management.
    """

    def __init__(
        self,
        replica_id: Optional[str] = None,
        torchLoom_addr: str = torchLoomConstants.DEFAULT_ADDR,
        config_queue: Optional[multiprocessing.Queue] = None,
        status_queue: Optional[multiprocessing.Queue] = None,
    ):
        # Core identifiers
        self._replica_id = replica_id or f"weavelet:{uuid.uuid4()}"
        self._device_uuid: Optional[str] = None

        # NATS connection setup
        self._torchLoom_addr = torchLoom_addr
        self._nc: Optional[Client] = None
        self._js: Optional[JetStreamContext] = None
        self._subscriptions: Dict[str, Any] = {}
        self._stop_event = multiprocessing.Event()

        # Inter-process communication
        self._config_queue = config_queue or multiprocessing.Queue()
        self._status_queue = status_queue or multiprocessing.Queue()

        # Process management
        self._process: Optional[multiprocessing.Process] = None

        # Configuration
        self._nc_timeout = Config.NC_TIMEOUT or 1
        self._exception_sleep = Config.EXCEPTION_RETRY_TIME or 1

        # Enhanced handler system
        self._handlers: Dict[str, Callable] = {}
        self._handler_types: Dict[str, Type] = {}
        self._auto_dispatch = True

    def register_handler(
        self, config_key: str, handler: Callable, expected_type: Optional[Type] = None
    ) -> None:
        """Register a handler for a specific configuration parameter.

        Args:
            config_key: The configuration parameter name (e.g., 'optimizer_type')
            handler: Function to call when this parameter changes
            expected_type: Expected type for the parameter value (inferred if not provided)
        """
        # Infer type from handler signature if not provided
        if expected_type is None:
            sig = inspect.signature(handler)
            params = list(sig.parameters.values())
            if len(params) >= 1:
                param = params[0]  # First parameter (after self if it's a method)
                if param.annotation != inspect.Parameter.empty:
                    expected_type = param.annotation
                else:
                    expected_type = str  # Default to string

        self._handlers[config_key] = handler
        self._handler_types[config_key] = expected_type or str
        print(f"Registered handler for '{config_key}' with type {expected_type}")

    def handler(self, config_key: str, expected_type: Optional[Type] = None):
        """Decorator for registering configuration handlers.

        Args:
            config_key: The configuration parameter name
            expected_type: Expected type for the parameter value

        Usage:
            @weavelet.handler("optimizer_type")
            def update_optimizer(self, new_type: str):
                # Implementation here
                pass
        """

        def decorator(func: Callable) -> Callable:
            self.register_handler(config_key, func, expected_type)
            return func

        return decorator

    def _validate_and_convert_value(self, config_key: str, value: Any) -> Any:
        """Validate and convert a configuration value to the expected type."""
        if config_key not in self._handler_types:
            return value

        expected_type = self._handler_types[config_key]

        # If value is already the correct type, return as-is
        if isinstance(value, expected_type):
            return value

        # Try to convert the value
        try:
            if expected_type == bool:
                # Handle boolean conversion from strings
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                return bool(value)
            elif expected_type == int:
                return int(float(value))  # Handle "1.0" -> 1
            elif expected_type == float:
                return float(value)
            elif expected_type == str:
                return str(value)
            else:
                # For other types, try direct conversion
                return expected_type(value)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Cannot convert value '{value}' to expected type {expected_type.__name__} "
                f"for config parameter '{config_key}': {e}"
            )

    def _dispatch_handlers(self, config_updates: Dict[str, Any]) -> None:
        """Automatically dispatch handlers for configuration updates."""
        for config_key, value in config_updates.items():
            if config_key in self._handlers:
                try:
                    # Validate and convert the value
                    converted_value = self._validate_and_convert_value(
                        config_key, value
                    )

                    # Call the handler
                    handler = self._handlers[config_key]
                    print(
                        f"Calling handler for '{config_key}' with value: {converted_value}"
                    )
                    handler(converted_value)

                except Exception as e:
                    print(f"Error in handler for '{config_key}': {e}")
            else:
                print(f"No handler registered for config key: {config_key}")

    def get_config_update(self, timeout: float = 0.1) -> Optional[Dict[str, str]]:
        """Get configuration update from the weavelet process if available.

        If auto_dispatch is enabled (default), this will automatically call
        registered handlers. Otherwise, it returns the config update dict.
        """
        try:
            config_update = self._config_queue.get_nowait()

            if config_update and self._auto_dispatch:
                # Automatically dispatch to registered handlers
                self._dispatch_handlers(config_update)
                return None  # No need to return since handlers were called

            return config_update
        except:
            return None

    def check_and_apply_updates(self) -> bool:
        """Check for configuration updates and apply them automatically.

        Returns:
            True if updates were found and applied, False otherwise
        """
        config_update = (
            self._config_queue.get_nowait() if not self._config_queue.empty() else None
        )
        if config_update:
            self._dispatch_handlers(config_update)
            return True
        return False

    def start(self) -> None:
        """Start the weavelet in a separate process."""
        try:
            self._process = multiprocessing.Process(
                target=self._run_weavelet_listener_process,
                args=(
                    self._replica_id,
                    self._torchLoom_addr,
                    self._config_queue,
                    self._status_queue,
                    self._stop_event,
                ),
                name=f"weavelet-{self._replica_id}",
            )
            self._process.start()

            # Give the process a moment to start
            time.sleep(0.1)

            print(f"Weavelet process started with PID: {self._process.pid}")
        except Exception as e:
            print(f"Failed to start weavelet process: {e}")
            raise

    def stop(self) -> None:
        """Stop the weavelet process and clean up resources."""
        try:
            if self._process and self._process.is_alive():
                print("Stopping weavelet process")
                self._stop_event.set()

                # Wait for the process to finish gracefully
                self._process.join(timeout=5)

                # If still alive, terminate forcefully
                if self._process.is_alive():
                    print("Force terminating weavelet process")
                    self._process.terminate()
                    self._process.join(timeout=2)

                    # Last resort - kill
                    if self._process.is_alive():
                        self._process.kill()
                        self._process.join()

                print("Weavelet process stopped successfully")
        except Exception as e:
            print(f"Error stopping weavelet process: {e}")

    def publish_training_status(self, status: Dict[str, Any]) -> None:
        """Send training status to the weavelet process for publishing."""
        try:
            self._status_queue.put_nowait(status)
        except:
            # Queue might be full, ignore for now
            pass

    @staticmethod
    def _run_weavelet_listener_process(
        replica_id: str,
        torchLoom_addr: str,
        config_queue: multiprocessing.Queue,
        status_queue: multiprocessing.Queue,
        stop_event: multiprocessing.Event,
    ) -> None:
        """Main function that runs in the separate weavelet listener process."""
        try:
            # Create event loop for this process
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Create the async weavelet listener instance
            weavelet_listener = WeaveletListener(
                replica_id=replica_id,
                torchLoom_addr=torchLoom_addr,
                config_queue=config_queue,
                status_queue=status_queue,
                stop_event=stop_event,
            )

            # Run the async main loop
            loop.run_until_complete(weavelet_listener.run())
        except Exception as e:
            print(f"Error in weavelet listener process: {e}")
        finally:
            # Clean up
            try:
                loop.close()
            except:
                pass


class WeaveletListener:
    """Async implementation of weavelet listener that runs inside the subprocess."""

    def __init__(
        self,
        replica_id: str,
        torchLoom_addr: str,
        config_queue: multiprocessing.Queue,
        status_queue: multiprocessing.Queue,
        stop_event: multiprocessing.Event,
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
        """Subscribe to JetStream subject."""
        try:
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

            # Create a status message (this could be extended with proper protobuf message)
            envelope = EventEnvelope()
            # Add status fields as needed based on your protobuf schema

            await self._nc.publish(
                "torchLoom.training.status", envelope.SerializeToString()
            )
            self._logger.debug(f"Published training status: {status}")
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
            # Cancel all subscriptions
            await cancel_subscriptions(self._subscriptions)
            self._subscriptions.clear()

            # Close NATS connection
            if self._nc and not self._nc.is_closed:
                await self._nc.close()

            self._logger.info("WeaveletListener cleanup completed")
        except Exception as e:
            self._logger.exception(f"Error during cleanup: {e}")