"""
Core Threadlet class for process-based configuration management.
"""

import asyncio
import logging
import multiprocessing
import os
import platform
import threading
import time
import uuid
from multiprocessing.connection import Connection
from typing import Any, Dict, Optional, Tuple, Type

from torchLoom.common.constants import NatsConstants, TimeConstants
from torchLoom.common.handlers import *
from torchLoom.common.utils import get_device_uuid
from torchLoom.proto import torchLoom_pb2

from .listener import ThreadletListener
from .publishers import ThreadletEventPublisher


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
        device_uuid: Optional[str] = None,
        torchLoom_addr: str = NatsConstants.DEFAULT_ADDR,
    ):
        # Core identifiers
        self._process_id = process_id or f"threadlet:{uuid.uuid4()}"
        self._device_uuid = device_uuid or get_device_uuid()
        self._server_id = platform.node()  # TODO: Add tailored method for this

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

        # Logger for this class
        self._logger = logging.getLogger(__name__)

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
                    self._device_uuid,
                    self._server_id,
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

        except Exception as e:
            self._logger.exception(f"Failed to start threadlet process: {e}")
            self._cleanup()

    def publish(self, message_type: str, **kwargs: Any) -> None:
        """Queues a message to be published by the ThreadletListener via NATS.

        Args:
            message_type: The type of event to publish (e.g., "training_status", "device_status", "heartbeat").
            **kwargs: Arguments specific to the message type.
                      For "device_status", 'device_uuid' (if not self._device_uuid) and 'status_data' are needed.
                      For "training_status", 'status_data' is needed.
                      process_id and device_uuid (for heartbeat) are added automatically if not overridden.
        """
        try:
            kwargs["process_id"] = self._process_id
            kwargs["device_uuid"] = self._device_uuid

            # Prepare a dictionary payload for the listener
            # The listener will use ThreadletEventPublisher with this data
            payload_for_listener = {
                "action": "publish_event",
                "event_type": message_type,
                "event_data": kwargs,
            }

            self._send_internal_message_to_listener(payload_for_listener)
            self._logger.debug(
                f"Queued '{message_type}' event for publishing by listener. Data: {kwargs}"
            )

        except Exception as e:
            self._logger.exception(
                f"publish: Error queueing event '{message_type}': {e}"
            )

    def _send_internal_message_to_listener(self, payload: Dict[str, Any]) -> None:
        """Sends a dictionary payload to the ThreadletListener process.

        All pipe communication now uses dictionary format for consistency.
        """
        try:
            if self._main_pipe_conn and not self._main_pipe_conn.closed:
                self._main_pipe_conn.send(payload)
                self._logger.debug(
                    f"Sent internal message to ThreadletListener: {payload.get('action', 'N/A')}"
                )
        except (BrokenPipeError, OSError):
            self._logger.warning(
                "Pipe to listener process is broken, dropping internal message."
            )
        except Exception as e:
            self._logger.exception(f"Error sending internal message via pipe: {e}")

    def publish_device_registration(self) -> None:
        """Requests publishing of device registration event."""
        self.publish(message_type="device_registration")

    def publish_heartbeat(
        self, status: str = "active", metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """Requests publishing of heartbeat event."""
        # process_id and device_uuid will be added by self.publish()
        kwargs = {"status": status}
        if metadata:
            kwargs["metadata"] = metadata
        self.publish(message_type="heartbeat", **kwargs)

    def publish_training_status(
        self, **kwargs: Any  # Should contain status_data dictionary
    ) -> None:
        """Send status message to ThreadletListener to be published."""
        self.publish(message_type="training_status", **kwargs)

    def publish_device_status(self, **kwargs: Any) -> None:
        """Send device status message to ThreadletListener to be published."""
        self.publish(message_type="device_status", **kwargs)

    def _pipe_message_processor_loop(self) -> None:
        """Continuously listens for dictionary messages on the bidirectional pipe.

        All communication through the pipe uses dictionary format for consistency.
        Messages from the listener process are expected to have a 'message_type' field
        to indicate what kind of message it is (e.g., 'command', 'status', etc.).
        """
        self._logger.info("Pipe message processor loop started.")
        try:
            while not self._pipe_listener_stop_event.is_set():
                if self._main_pipe_conn.poll(TimeConstants.PIPE_POLL_INTERVAL):
                    received_data = self._main_pipe_conn.recv()
                    self._logger.debug(
                        f"Received data from listener process: {type(received_data)}"
                    )

                    if isinstance(received_data, dict):
                        message_type = received_data.get("message_type")
                        if message_type == "command":
                            self._handle_command_dict(received_data)
                    else:
                        self._logger.warning(
                            f"Received unexpected data type from pipe: {type(received_data)}. Expected dict."
                        )

        except EOFError:
            self._logger.info("Pipe closed, listener process likely terminated.")
        except Exception as e:
            if not self._pipe_listener_stop_event.is_set():
                self._logger.exception(f"Error in pipe message processor loop: {e}")
        finally:
            self._logger.info("Pipe message processor loop stopped.")

    def _handle_command_dict(self, command_dict: Dict[str, Any]) -> None:
        """Handle a command message received as a dictionary from the listener.

        Args:
            command_dict: Dictionary containing command information with structure:
                - message_type: 'command'
                - command_type: The type of command (e.g., 'update_config', 'pause', 'resume', 'stop')
                - payload: The command data
        """
        try:
            command_type = command_dict.get("command_type")
            payload = command_dict.get("payload", {})

            if command_type == "update_config":
                for config_key, config_value in payload.items():
                    if self._handler_registry.has_handler(config_key):
                        handler = self._handler_registry.get_handler(config_key)
                        handler(config_value)
                    else:
                        self._logger.warning(
                            f"No handler registered for config key: {config_key}. Available handlers: {list(self._handler_registry._handlers.keys())}"
                        )

            elif command_type == "pause":
                self._logger.info("Received pause command from weaver")
                if self._handler_registry.has_handler("pause_training"):
                    handler = self._handler_registry.get_handler("pause_training")
                    handler()
                else:
                    self._logger.warning("No handler registered for pause_training")

            elif command_type == "resume":
                self._logger.info("Received resume command from weaver")
                if self._handler_registry.has_handler("resume_training"):
                    handler = self._handler_registry.get_handler("resume_training")
                    handler()
                else:
                    self._logger.warning("No handler registered for resume_training")

            elif command_type == "stop":
                self._logger.info("Received stop command from weaver")
                self.stop()
            else:
                self._logger.warning(f"Unknown command type: {command_type}")

        except Exception as e:
            self._logger.exception(f"Error handling command dict: {e}")

    @staticmethod
    def _run_threadlet_listener_process(
        process_id: str,
        device_uuid: str,
        server_id: str,
        torchLoom_addr: str,
        pipe_to_main_process: Connection,  # This pipe is bidirectional
        stop_event: multiprocessing.Event,
    ) -> None:
        """Main function that runs in the separate threadlet listener process."""
        # This is static method context. Logger here will be for the listener process.
        listener_logger = logging.getLogger(
            __name__ + ".listener_process"
        )  # Separate logger
        try:
            # Create event loop for this process
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # NATS connection will be established by ThreadletListener's init or run.
            # ThreadletEventPublisher will be instantiated within ThreadletListener
            # or passed the NATS clients from it.

            # Create the async threadlet listener instance
            # It will need to instantiate ThreadletEventPublisher internally
            threadlet_listener = ThreadletListener(
                process_id=process_id,
                device_uuid=device_uuid,
                server_id=server_id,
                torchLoom_addr=torchLoom_addr,
                pipe_to_main_process=pipe_to_main_process,
                stop_event=stop_event,
            )

            loop.run_until_complete(threadlet_listener.run())
        except Exception as e:
            listener_logger.exception(f"Error in threadlet listener process: {e}")
        finally:
            try:
                if loop and not loop.is_closed():
                    loop.close()
            except Exception as loop_close_exc:
                listener_logger.error(
                    f"Error closing event loop in listener process: {loop_close_exc}"
                )

    def stop(self) -> None:
        """Stop the threadlet process and clean up resources."""
        self._logger.info("Stopping threadlet...")
        self._cleanup()

    def _cleanup(self) -> None:
        # Ensure cleanup if start fails partially
        if self._pipe_listener_thread and self._pipe_listener_thread.is_alive():
            self._pipe_listener_stop_event.set()
            self._pipe_listener_thread.join(timeout=TimeConstants.PIPE_LISTENER_TIMEOUT)

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

    def register_handler(self, config_key: str, handler, expected_type=None) -> None:
        """Register a handler for a specific configuration parameter.

        Args:
            config_key: The configuration parameter name (e.g., 'optimizer_type')
            handler: Function to call when this parameter changes
            expected_type: Expected type for the parameter value (ignored - no type checking)
        """
        self._handler_registry.register_handler(config_key, handler, expected_type)
