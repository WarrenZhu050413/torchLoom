import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Callable, Awaitable, Any
import threading

import nats
from nats.aio.client import Client
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext
from nats.js.api import StreamConfig

from torchLoom.constants import torchLoomConstants, NC, JS
from torchLoom.config import Config
from torchLoom.log.logger import setup_logger
from torchLoom.log.log_utils import log_and_raise_exception
from torchLoom.torchLoom_pb2 import EventEnvelope
from torchLoom.utils import get_device_uuid, cancel_subscriptions


class Weavelet:
    """Weavelet for torchLoom training processes.

    This class manages all communication between training processes and the weaver,
    including receiving configuration updates and sending training status updates.
    It runs async operations in a background thread similar to the marduk pattern.
    """

    def __init__(self, replica_id: Optional[str] = None, torchLoom_addr: str = torchLoomConstants.DEFAULT_ADDR):
        # Setup logging
        self._logger = setup_logger(
            name="weavelet_logger",
            log_file=Config.MANAGER_torchLoom_LOG_FILE,
            format_log=True,
            print_to_console=False
        )

        # Core identifiers
        self._replica_id = replica_id or f"weavelet:{uuid.uuid4()}"
        self._device_uuid: Optional[str] = None
        
        # NATS connection setup
        self._torchLoom_addr = torchLoom_addr
        self._nc: Optional[Client] = None
        self._js: Optional[JetStreamContext] = None
        self._subscriptions: Dict[str, Any] = {}
        self._stop_nats = asyncio.Event()
        
        # Configuration
        self._nc_timeout = Config.NC_TIMEOUT or 1
        self._exception_sleep = Config.EXCEPTION_RETRY_TIME or 1
        
        # Callback registry for different message types
        self._message_handlers: Dict[str, Callable[[Dict[str, str]], None]] = {}
        
        # Event loop management
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[ThreadPoolExecutor] = None
        
        self._logger.info(f"Weavelet initialized with replica_id: {self._replica_id}")

    def register_config_handler(self, config_key: str, handler: Callable[[str], None]) -> None:
        """Register a handler for specific configuration updates.
        
        Args:
            config_key: The configuration parameter key (e.g., 'optimizer_type')
            handler: Function to call when this config parameter changes
        """
        self._message_handlers[config_key] = handler
        self._logger.info(f"Registered handler for config key: {config_key}")

    def start(self) -> None:
        """Start the weavelet in a background thread."""
        try:
            # Set up event loop
            if asyncio.get_event_loop().is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            else:
                self._loop = asyncio.get_event_loop()
            
            # Initialize async components
            self._loop.run_until_complete(self._async_start())
            
            # Start background thread to keep event loop running
            self._loop_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weavelet_loop")
            self._loop_thread.submit(self._run_event_loop_in_background)
            
            self._logger.info("Weavelet started successfully in background thread")
        except Exception as e:
            self._logger.exception(f"Failed to start weavelet: {e}")
            raise

    async def _async_start(self) -> None:
        """Initialize async components."""
        try:
            await self._connect()
            await self._setup_subscriptions()
            await self._register_device()
            self._logger.info("Weavelet async initialization completed")
        except Exception as e:
            self._logger.exception(f"Error during weavelet async start: {e}")
            raise

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
                    subjects=[torchLoomConstants.subjects.CONFIG_INFO]
                )
            )
            
            await self._subscribe_js(
                stream="WEAVELET_STREAM",
                subject=torchLoomConstants.subjects.CONFIG_INFO,
                consumer=f"weavelet-{self._replica_id}",
                message_handler=self._handle_config_message
            )
            
            # Subscribe to replica fail events
            await self._subscribe_nc(
                subject=torchLoomConstants.subjects.REPLICA_FAIL,
                message_handler=self._handle_replica_fail_message
            )
            
            self._logger.info("Weavelet subscriptions set up successfully")
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
                envelope.SerializeToString()
            )
            
            self._logger.info(f"Registered device {self._device_uuid} with replica {self._replica_id}")
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
                
                # Call registered handlers for each config parameter
                for key, value in params.items():
                    if key in self._message_handlers:
                        try:
                            self._message_handlers[key](value)
                            self._logger.info(f"Applied config update: {key} = {value}")
                        except Exception as e:
                            self._logger.exception(f"Error applying config {key} = {value}: {e}")
            
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
                    self._logger.warning(f"This replica ({self._replica_id}) has been marked as failed")
                    # Could trigger recovery logic here
                else:
                    self._logger.info(f"Another replica failed: {failed_replica_id}")
        except Exception as e:
            self._logger.exception(f"Error handling replica fail message: {e}")

    async def _subscribe_js(self, stream: str, subject: str, consumer: str, message_handler: Callable[[Msg], Awaitable[None]]) -> None:
        """Subscribe to JetStream subject."""
        try:
            psub = await self._js.pull_subscribe(subject, durable=consumer, stream=stream)
            self._logger.info(f"Subscribed to JetStream {subject} on stream {stream}")
            
            async def listen_to_js_subscription():
                self._logger.info(f"Started listening on JetStream {subject}")
                while not self._stop_nats.is_set():
                    try:
                        msgs = await psub.fetch(1, timeout=1)
                        for msg in msgs:
                            await message_handler(msg)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        self._logger.exception(f"Error in JetStream subscription loop for {subject}: {e}")
                        await asyncio.sleep(self._exception_sleep)
            
            task = asyncio.create_task(listen_to_js_subscription())
            self._subscriptions[subject] = (psub, task)
        except Exception as e:
            self._logger.exception(f"Failed to subscribe to JetStream {subject}: {e}")
            raise

    async def _subscribe_nc(self, subject: str, message_handler: Callable[[Msg], Awaitable[None]]) -> None:
        """Subscribe to regular NATS subject."""
        try:
            sub = await self._nc.subscribe(subject)
            self._logger.info(f"Subscribed to NATS {subject}")
            
            async def listen_to_nc_subscription():
                self._logger.info(f"Started listening on NATS {subject}")
                while not self._stop_nats.is_set():
                    try:
                        msg = await sub.next_msg(timeout=self._nc_timeout)
                        await message_handler(msg)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        self._logger.exception(f"Error in NATS subscription loop for {subject}: {e}")
                        await asyncio.sleep(self._exception_sleep)
            
            task = asyncio.create_task(listen_to_nc_subscription())
            self._subscriptions[subject] = (sub, task)
        except Exception as e:
            self._logger.exception(f"Failed to subscribe to NATS {subject}: {e}")
            raise

    def _run_event_loop_in_background(self) -> None:
        """Run the event loop in a background thread."""
        try:
            # Keep the loop running until stopped
            while not self._stop_nats.is_set():
                asyncio.run_coroutine_threadsafe(asyncio.sleep(0.1), self._loop).result(timeout=1)
        except Exception as e:
            self._logger.exception(f"Error in background event loop: {e}")

    def stop(self) -> None:
        """Stop the weavelet and clean up resources."""
        try:
            self._logger.info("Stopping weavelet")
            self._stop_nats.set()
            
            if self._loop and not self._loop.is_closed():
                # Stop async components
                future = asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop)
                future.result(timeout=5)  # Wait up to 5 seconds for cleanup
            
            if self._loop_thread:
                self._loop_thread.shutdown(wait=True)
                
            self._logger.info("Weavelet stopped successfully")
        except Exception as e:
            self._logger.exception(f"Error stopping weavelet: {e}")

    async def _async_stop(self) -> None:
        """Stop async components."""
        try:
            await cancel_subscriptions(self._subscriptions)
            self._subscriptions.clear()
            
            if self._nc and not self._nc.is_closed:
                await self._nc.close()
                
            self._logger.info("Weavelet async components stopped")
        except Exception as e:
            self._logger.exception(f"Error stopping async components: {e}")

    def publish_training_status(self, status: Dict[str, Any]) -> None:
        """Publish training status updates to the weaver."""
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._async_publish_status(status), self._loop)

    async def _async_publish_status(self, status: Dict[str, Any]) -> None:
        """Async method to publish training status."""
        try:
            if not self._nc:
                self._logger.warning("Cannot publish status - not connected to NATS")
                return
                
            # Create a status message (this could be extended with proper protobuf message)
            envelope = EventEnvelope()
            # Add status fields as needed based on your protobuf schema
            
            await self._nc.publish("torchLoom.training.status", envelope.SerializeToString())
            self._logger.debug(f"Published training status: {status}")
        except Exception as e:
            self._logger.exception(f"Failed to publish training status: {e}")


# Backward compatibility - simple function that creates a weavelet
def weavelet_process(queue, addr: str = torchLoomConstants.DEFAULT_ADDR) -> None:
    """Backward compatibility function that mimics the old multiprocessing approach."""
    weavelet = Weavelet(torchLoom_addr=addr)
    
    def handle_optimizer_change(optimizer_type: str):
        queue.put(optimizer_type)
    
    weavelet.register_config_handler("optimizer_type", handle_optimizer_change)
    weavelet.start()
    
    try:
        # Keep the process running
        while True:
            asyncio.sleep(1)
    except KeyboardInterrupt:
        weavelet.stop()

