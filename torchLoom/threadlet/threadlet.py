"""
Core Threadlet class for process-based configuration management.
"""

import asyncio
import logging
import multiprocessing
import threading
import time
import uuid
from multiprocessing.connection import Connection
from typing import Any, Dict, Optional, Tuple, Type

from torchLoom.common.constants import TimeConstants, NatsConstants
from torchLoom.common.handlers import *
from torchLoom.proto import torchLoom_pb2
from torchLoom.common.utils import get_device_uuid

from . import handlers
from .listener import ThreadletListener
from .message import (
    CommandType,
    MessageFactory,
    MessageType,
    deserialize_message,
    serialize_message,
)

class Threadlet:
    """Process-based Threadlet for torchLoom training processes.

    This class manages all communication between training processes and the weaver,
    including receiving configuration updates and sending training status updates.
    It runs in a separate process using multiprocessing.Process and supports
    handler registration for automatic configuration management.

    Note: Heartbeats are sent automatically by the ThreadletListener process
    at regular intervals. The main process cannot control heartbeat timing.
    """

    def __init__(
        self,
        process_id: Optional[str] = None,
        torchLoom_addr: str = NatsConstants.DEFAULT_ADDR,
    ):
        # Core identifiers
        self._process_id = process_id or f"threadlet:{uuid.uuid4()}"
        self._device_uuid = get_device_uuid()
        self._server_id = os.hostname() # TODO: Add tailored method for this

        # NATS connection setup
        self._torchLoom_addr = torchLoom_addr
        self._stop_event = multiprocessing.Event()
        self._pipe_listener_stop_event = threading.Event()

        # Process management
        self._threadlet_listener_process: Optional[multiprocessing.Process] = None
        self._pipe_listener_thread: Optional[threading.Thread] = None
        self._listener_pipe_conn, self._main_pipe_conn = multiprocessing.Pipe(
            duplex=True
        )

        # Configuration
        self._nc_timeout = TimeConstants.PIPE_POLL_INTERVAL

        # Handler registry for configuration updates
        self._handler_registry = HandlerRegistry("threadlet_config")
        self._auto_dispatch = True

        self._target_object = None  # Will be set when using with Lightning wrapper

        # Logger for this class
        self._logger = logging.getLogger(__name__)

    def publish(self, message_type: str, **kwargs: Any) -> None:
        """Publishes a message to the ThreadletListener based on the message type.

        Args:
            message_type: The type of message to send (e.g., "training_status", "device_status").
            **kwargs: Arguments specific to the message type. 
                      For "device_status", 'device_uuid' must be provided in kwargs.
        """
        try:
            message_to_send = None
            log_identifier = ""

            if message_type == "training_status":
                self._logger.debug(f"publish: type='training_status', process_id={self._process_id}, args: {kwargs}")
                message_to_send = MessageFactory.create_training_status(
                    process_id=self._process_id,
                    **kwargs
                )
                log_identifier = f"process_id: {self._process_id}, step: {kwargs.get('current_step')}"
            elif message_type == "device_status":
                self._logger.debug(f"publish: type='device_status', device_uuid={self._device_uuid}, process_id={self._process_id}, args: {kwargs}")
                message_to_send = MessageFactory.create_device_status(
                    device_uuid=self._device_uuid,
                    process_id=self._process_id,
                    server_id=self._server_id,
                    **kwargs
                )
                log_identifier = f"device_uuid: {self._device_uuid} (process: {self._process_id})"
            else:
                self._logger.error(f"publish: Unknown message type '{message_type}'.")
                return

            if message_to_send:
                self._send_message_to_listener(message_to_send)
                self._logger.debug(f"publish: Sent {message_type} message for {log_identifier}")

        except Exception as e:
            self._logger.exception(f"publish: Error for message_type '{message_type}': {e}")

    def publish_training_status(
        self,
        **kwargs: Any
    ) -> None:
        """Send status message to ThreadletListener."""
        self.publish(message_type="training_status", **kwargs)

    def publish_device_status(
        self,
        device_uuid: str,
        **kwargs: Any
    ) -> None:
        """Send device status message to ThreadletListener."""
        self.publish(message_type="device_status", device_uuid=device_uuid, **kwargs)

    def register_handler(self, config_key: str, handler, expected_type=None) -> None:
        """Register a handler for a specific configuration parameter.

        Args:
            config_key: The configuration parameter name (e.g., 'optimizer_type')
            handler: Function to call when this parameter changes
            expected_type: Expected type for the parameter value (ignored - no type checking)
        """
        self._handler_registry.register_handler(config_key, handler, expected_type)

    def _pipe_message_processor_loop(self) -> None:
        """Continuously listens for messages on the pipe from ThreadletListener."""
        self._logger.info("Pipe message processor loop started.")
        try:
            while not self._pipe_listener_stop_event.is_set():
                if self._main_pipe_conn.poll(TimeConstants.PIPE_POLL_INTERVAL):
                    received_data = self._main_pipe_conn.recv()
                    self._logger.debug(
                        f"Received raw data from listener process: {type(received_data)}"
                    )

                    message = None
                    if isinstance(received_data, tuple) and len(received_data) == 2:
                        message_type_str, serialized_bytes = received_data
                        if isinstance(serialized_bytes, bytes) and isinstance(message_type_str, str):
                            message = deserialize_message(serialized_bytes, message_type_str)
                        else:
                            self._logger.warning(f"Received malformed tuple from pipe: types were {type(message_type_str)}, {type(serialized_bytes)}")
                    else:
                        self._logger.warning(f"Received unexpected data type from pipe: {type(received_data)}")

                    if message:
                        if message.message_type == MessageType.COMMAND.value:
                            handlers.handle_command_message(
                                message=message,
                                handler_registry=self._handler_registry,
                                auto_dispatch=self._auto_dispatch,
                                stop_callback=self.stop,
                            )
                        else:
                            self._logger.warning(f"Received unexpected message type {message.message_type} from listener. Expected COMMAND.")
        except EOFError:
            self._logger.info("Pipe closed, listener process likely terminated.")
        except Exception as e:
            if (
                not self._pipe_listener_stop_event.is_set()
            ):  # Log only if not intentionally stopping
                self._logger.exception(f"Error in pipe message processor loop: {e}")
        finally:
            self._logger.info("Pipe message processor loop stopped.")

    def _send_message_to_listener(self, message) -> None:
        """Send a structured message to the ThreadletListener process."""
        try:
            if self._main_pipe_conn and not self._main_pipe_conn.closed:
                # Determine the message_type_str for the listener's deserialize_message
                message_type_str = ""
                # The 'message' object here is a protobuf message instance
                if isinstance(message, torchLoom_pb2.PipeTrainingStatusMessage):
                    message_type_str = "TRAINING_STATUS"
                elif isinstance(message, torchLoom_pb2.PipeDeviceStatusMessage):
                    message_type_str = "DEVICE_STATUS"
                elif isinstance(message, torchLoom_pb2.PipeCommandMessage):
                    message_type_str = "COMMAND"
                else:
                    self._logger.error(f"Attempting to send unknown protobuf message type: {type(message)}")
                    return

                serialized_bytes = message.SerializeToString()
                payload = (message_type_str, serialized_bytes)
                self._main_pipe_conn.send(payload)
                self._logger.debug(
                    f"Sent message to ThreadletListener: type_str='{message_type_str}', content_bytes_len={len(serialized_bytes)}"
                )
        except (BrokenPipeError, OSError):
            self._logger.warning("Pipe to listener process is broken, dropping message")
        except Exception as e:
            self._logger.warning(f"Error sending message via pipe: {e}")

    def start(self) -> None:
        """Start the threadlet in a separate process."""
        try:
            # Start the pipe listener thread in this main process
            self._pipe_listener_stop_event.clear()
            self._pipe_listener_thread = threading.Thread(
                target=self._pipe_message_processor_loop,
                name=f"threadlet-pipe-listener-{self._process_id}",
                daemon=True,
            )
            self._pipe_listener_thread.start()
            self._logger.info("Threadlet pipe listener thread started.")

            self._threadlet_listener_process = multiprocessing.Process(
                target=self._run_threadlet_listener_process,
                args=(
                    self._process_id,
                    self._torchLoom_addr,
                    self._listener_pipe_conn,  # Pass one end of the duplex pipe
                    self._stop_event,
                ),
                name=f"threadlet-listener-proc-{self._process_id}",
            )
            self._threadlet_listener_process.start()

            # Give the process a moment to start
            time.sleep(TimeConstants.BRIEF_PAUSE)

            self._logger.info(
                f"ThreadletListener process started with PID: {self._threadlet_listener_process.pid}"
            )

            # Log registered handlers
            handlers = self.get_registered_handlers()
            self._logger.info(
                f"Threadlet has {len(handlers)} registered configuration handlers"
            )
        except Exception as e:
            self._logger.exception(f"Failed to start threadlet process: {e}")
            self._cleanup()

    def get_registered_handlers(self) -> Dict[str, Type]:
        """Get all currently registered handlers.

        Returns:
            Dictionary mapping config keys to their expected types
        """
        return self._handler_registry.list_handlers()

    def stop(self) -> None:
        """Stop the threadlet process and clean up resources."""
        self._logger.info("Stopping threadlet...")
        self._cleanup()

    def _cleanup(self) -> None:
        # Ensure cleanup if start fails partially
        if self._pipe_listener_thread and self._pipe_listener_thread.is_alive():
            self._pipe_listener_stop_event.set()
            self._pipe_listener_thread.join(
                timeout=TimeConstants.PIPE_LISTENER_TIMEOUT
            )
        
        # Signal the listener process to stop first
        if (
            self._threadlet_listener_process
            and self._threadlet_listener_process.is_alive()
        ):
            self._stop_event.set()

        # Now, manage the listener process
        if (
            self._threadlet_listener_process
            and self._threadlet_listener_process.is_alive()
        ):
            self._threadlet_listener_process.join(
                timeout=TimeConstants.PIPE_LISTENER_TIMEOUT
            )
            if self._threadlet_listener_process.is_alive():
                self._threadlet_listener_process.terminate()
        
        # Close pipe connections after signaling/stopping processes
        try:
            if self._main_pipe_conn:
                self._main_pipe_conn.close()
                self._logger.info("Main pipe connection closed.")
        except Exception as e:
            self._logger.error(f"Error closing main pipe connection: {e}")
        
        try:
            if self._listener_pipe_conn:  # Check if it exists
                # Check if it's a real connection object and has a close method
                if hasattr(self._listener_pipe_conn, "close") and callable(
                    getattr(self._listener_pipe_conn, "close")
                ):
                    self._listener_pipe_conn.close()
                    self._logger.info(
                        "Listener pipe connection closed from main process side (best effort)."
                    )
        except Exception as e:
            self._logger.warning(
                f"Error attempting to close listener pipe connection from main: {e}"
            )
        
        self._logger.info("Threadlet stopped and resources cleaned up.")

    @staticmethod
    def _run_threadlet_listener_process(
        process_id: str,
        torchLoom_addr: str,
        pipe_to_main_process: Connection,
        stop_event: multiprocessing.Event,
    ) -> None:
        """Main function that runs in the separate threadlet listener process."""
        try:
            # Create event loop for this process
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Create the async threadlet listener instance
            threadlet_listener = ThreadletListener(
                process_id=process_id,
                torchLoom_addr=torchLoom_addr,
                pipe_to_main_process=pipe_to_main_process,
                stop_event=stop_event,
            )

            # Run the async main loop
            loop.run_until_complete(threadlet_listener.run())
        except Exception as e:
            print(f"Error in threadlet listener process: {e}")
        finally:
            # Clean up
            try:
                loop.close()
            except:
                pass
