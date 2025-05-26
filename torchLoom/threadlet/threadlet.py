"""
Core Threadlet class for process-based configuration management.
"""

import asyncio
import logging
import multiprocessing
import time
import uuid
import threading
from multiprocessing.connection import Connection
from typing import Any, Dict, Optional, Tuple, Type

from torchLoom.common.config import Config
from torchLoom.common.constants import torchLoomConstants

from .handlers import HandlerRegistry
from .listener import ThreadletListener
from .message import (
    MessageType, 
    MessageFactory, 
    serialize_message, 
    deserialize_message,
    CommandType
)


class Threadlet:
    """Process-based Threadlet for torchLoom training processes.

    This class manages all communication between training processes and the weaver,
    including receiving configuration updates and sending training status updates.
    It runs in a separate process using multiprocessing.Process and supports
    decorator-based handler registration for automatic configuration management.
    """

    def __init__(
        self,
        replica_id: Optional[str] = None,
        torchLoom_addr: str = torchLoomConstants.DEFAULT_ADDR,
    ):
        # Core identifiers
        self._replica_id = replica_id or f"threadlet:{uuid.uuid4()}"
        self._device_uuid: Optional[str] = None

        # NATS connection setup
        self._torchLoom_addr = torchLoom_addr
        self._stop_event = multiprocessing.Event()
        self._pipe_listener_stop_event = threading.Event()

        # Inter-process communication with the listener process
        self._listener_pipe_conn, self._main_pipe_conn = multiprocessing.Pipe(duplex=True)

        # Process management
        self._threadlet_listener_process: Optional[multiprocessing.Process] = None
        self._pipe_listener_thread: Optional[threading.Thread] = None

        # Configuration
        self._nc_timeout = Config.NC_TIMEOUT or 1
        self._exception_sleep = Config.EXCEPTION_RETRY_TIME or 1

        # Can register handler using the handler decorator in the registry
        self._handler_registry = HandlerRegistry()
        self._auto_dispatch = True

        # Default handlers setup
        self._default_handlers_enabled = True
        self._target_object = None  # Will be set when using with Lightning wrapper

        # Logger for this class
        self._logger = logging.getLogger(__name__)

    def register_handler(self, config_key: str, handler, expected_type=None) -> None:
        """Register a handler for a specific configuration parameter.

        Args:
            config_key: The configuration parameter name (e.g., 'optimizer_type')
            handler: Function to call when this parameter changes
            expected_type: Expected type for the parameter value (ignored - no type checking)
        """
        self._handler_registry.register_handler(config_key, handler, expected_type)

    def handler(self, config_key: str, expected_type=None):
        """Decorator for registering configuration handlers.

        Args:
            config_key: The configuration parameter name
            expected_type: Expected type for the parameter value (ignored - no type checking)

        Usage:
            @threadlet.handler("optimizer_type")
            def update_optimizer(self, new_type: str):
                # Implementation here
                pass
        """

        def decorator(func):
            self.register_handler(config_key, func, expected_type)
            return func

        return decorator

    def _dispatch_handlers(self, config_updates: Dict[str, Any]) -> None:
        """Automatically dispatch handlers for configuration updates."""
        self._handler_registry.dispatch_handlers(config_updates)

    def _pipe_message_processor_loop(self) -> None:
        """Continuously listens for messages on the pipe from ThreadletListener."""
        self._logger.info("Pipe message processor loop started.")
        try:
            while not self._pipe_listener_stop_event.is_set():
                if self._main_pipe_conn.poll(0.1): # Poll with a timeout
                    received_data = self._main_pipe_conn.recv()
                    self._logger.debug(f"Received raw data from listener process: {type(received_data)}")
                    
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
                        # Only COMMAND messages are expected from Listener to Threadlet now
                        if message.message_type == MessageType.COMMAND.value: # Compare with protobuf enum value
                            self._handle_command_message(message)
                        else:
                            self._logger.warning(f"Received unexpected message type {message.message_type} from listener. Expected COMMAND.")
        except EOFError:
            self._logger.info("Pipe closed, listener process likely terminated.")
        except Exception as e:
            if not self._pipe_listener_stop_event.is_set(): # Log only if not intentionally stopping
                self._logger.exception(f"Error in pipe message processor loop: {e}")
        finally:
            self._logger.info("Pipe message processor loop stopped.")

    def _handle_command_message(self, message) -> None:
        """Handle command message from ThreadletListener (includes config updates)."""
        try:
            command_type = message.command_type.value if hasattr(message.command_type, 'value') else message.command_type
            params = dict(message.params) if message.params else {}
            
            # Check for custom command type in params
            actual_command_type = params.pop("_command_type", None) or command_type
            
            self._logger.info(f"Received command: {actual_command_type} with params: {params}")
            
            # Handle specific commands
            if message.command_type == CommandType.KILL or actual_command_type == "KILL":
                self._logger.warning("Received KILL command from weaver")
                self.stop()
            elif message.command_type == CommandType.PAUSE or actual_command_type == "PAUSE":
                self._logger.info("Received PAUSE command from weaver")
                if self._auto_dispatch:
                    self._dispatch_handlers({"pause_training": True})
            elif message.command_type == CommandType.RESUME or actual_command_type == "RESUME":
                self._logger.info("Received RESUME command from weaver")
                if self._auto_dispatch:
                    self._dispatch_handlers({"resume_training": True})
            elif message.command_type == CommandType.UPDATE_CONFIG or actual_command_type in ["UPDATE_CONFIG", "CONFIG"]:
                self._logger.info(f"Received config update command with params: {params}")
                if self._auto_dispatch and params:
                    self._dispatch_handlers(params)
            elif actual_command_type == "STATUS":
                self._logger.info(f"Received status command: {params}")
                # Handle status updates if needed
            else:
                self._logger.warning(f"Unknown command type: {actual_command_type}")
        except Exception as e:
            self._logger.exception(f"Error handling command message: {e}")

    def _send_current_status(self) -> None:
        """Send current status to ThreadletListener."""
        try:
            status_message = MessageFactory.create_status(
                replica_id=self._replica_id,
                status="active",
                message=f"handlers_registered: {len(self._handler_registry._handlers)}"
            )
            self._send_message_to_listener(status_message)
        except Exception as e:
            self._logger.exception(f"Error sending current status: {e}")

    def _send_message_to_listener(self, message) -> None:
        """Send a structured message to the ThreadletListener process."""
        try:
            if self._main_pipe_conn and not self._main_pipe_conn.closed:
                serialized_message = serialize_message(message)
                self._main_pipe_conn.send(serialized_message)
                self._logger.debug(f"Sent message to ThreadletListener: {message.message_type}")
        except (BrokenPipeError, OSError):
            self._logger.warning("Pipe to listener process is broken, dropping message")
        except Exception as e:
            self._logger.warning(f"Error sending message via pipe: {e}")

    def enable_auto_dispatch(self) -> None:
        """Enable automatic handler dispatch."""
        self._auto_dispatch = True

    def disable_auto_dispatch(self) -> None:
        """Disable automatic handler dispatch."""
        self._auto_dispatch = False

    def enable_default_handlers(self, target_object: Optional[Any] = None) -> None:
        """Enable default handlers for common configuration parameters.

        Args:
            target_object: The object that will receive configuration updates (e.g., Lightning module)
                          If None, only logging handlers are registered.
        """
        self._default_handlers_enabled = True
        self._target_object = target_object
        self._handler_registry.register_default_handlers(target_object)

    def disable_default_handlers(self) -> None:
        """Disable default handlers (users must register all handlers manually)."""
        self._default_handlers_enabled = False
        # Note: This doesn't remove already registered handlers, just prevents auto-registration

    def set_target_object(self, target_object: Any) -> None:
        """Set the target object for configuration updates.

        Args:
            target_object: The object that will receive configuration updates (e.g., Lightning module)
        """
        self._target_object = target_object
        # Re-register default handlers with the new target object
        if self._default_handlers_enabled:
            self._handler_registry.register_default_handlers(target_object)

    def get_registered_handlers(self) -> Dict[str, Type]:
        """Get all currently registered handlers.

        Returns:
            Dictionary mapping config keys to their expected types
        """
        return self._handler_registry.list_handlers()

    def get_supported_config_parameters(self) -> Dict[str, str]:
        """Get all supported configuration parameters and their descriptions.

        Returns:
            Dictionary mapping config keys to their descriptions
        """
        descriptions = {
            # Training parameters
            "learning_rate": "Learning rate for optimizer",
            "lr": "Learning rate (alias for learning_rate)",
            "batch_size": "Training batch size",
            "momentum": "Optimizer momentum",
            "weight_decay": "Weight decay regularization",
            # Optimizer parameters
            "optimizer_type": "Type of optimizer (adam, sgd, etc.)",
            "optimizer": "Optimizer type (alias)",
            # Training control
            "training_enabled": "Enable/disable training",
            "pause_training": "Pause training execution",
            "resume_training": "Resume paused training",
            # Model parameters
            "dropout_rate": "Dropout rate for regularization",
            "dropout": "Dropout rate (alias)",
            # Logging and debugging
            "log_level": "Logging level (DEBUG, INFO, WARNING, ERROR)",
            "logging_interval": "Interval for logging updates",
            "verbose": "Enable verbose output",
            # Advanced parameters
            "gradient_clip_val": "Gradient clipping value",
            "accumulate_grad_batches": "Number of batches to accumulate gradients",
        }

        return descriptions

    def start(self) -> None:
        """Start the threadlet in a separate process."""
        try:
            # Register default handlers if enabled
            if self._default_handlers_enabled:
                self._handler_registry.register_default_handlers(self._target_object)

            # Start the pipe listener thread in this main process
            self._pipe_listener_stop_event.clear()
            self._pipe_listener_thread = threading.Thread(
                target=self._pipe_message_processor_loop,
                name=f"threadlet-pipe-listener-{self._replica_id}",
                daemon=True,
            )
            self._pipe_listener_thread.start()
            self._logger.info("Threadlet pipe listener thread started.")

            self._threadlet_listener_process = multiprocessing.Process(
                target=self._run_threadlet_listener_process,
                args=(
                    self._replica_id,
                    self._torchLoom_addr,
                    self._listener_pipe_conn, # Pass one end of the duplex pipe
                    self._stop_event,
                ),
                name=f"threadlet-listener-proc-{self._replica_id}",
            )
            self._threadlet_listener_process.start()

            # Give the process a moment to start
            time.sleep(0.1)

            self._logger.info(f"ThreadletListener process started with PID: {self._threadlet_listener_process.pid}")

            # Log registered handlers
            handlers = self.get_registered_handlers()
            self._logger.info(f"Threadlet has {len(handlers)} registered configuration handlers")
        except Exception as e:
            self._logger.exception(f"Failed to start threadlet process: {e}")
            # Ensure cleanup if start fails partially
            if self._pipe_listener_thread and self._pipe_listener_thread.is_alive():
                self._pipe_listener_stop_event.set()
                self._pipe_listener_thread.join(timeout=1)
            if self._threadlet_listener_process and self._threadlet_listener_process.is_alive():
                self._stop_event.set()
                self._threadlet_listener_process.join(timeout=1)
                if self._threadlet_listener_process.is_alive(): self._threadlet_listener_process.terminate()
            raise

    def stop(self) -> None:
        """Stop the threadlet process and clean up resources."""
        self._logger.info("Stopping threadlet...")
        try:
            # Stop the pipe listener thread in this process
            if self._pipe_listener_thread and self._pipe_listener_thread.is_alive():
                self._logger.info("Stopping pipe listener thread...")
                self._pipe_listener_stop_event.set()
                self._pipe_listener_thread.join(timeout=2)
                if self._pipe_listener_thread.is_alive():
                    self._logger.warning("Pipe listener thread did not stop in time.")
                else:
                    self._logger.info("Pipe listener thread stopped.")
            
            # Stop the ThreadletListener process
            if self._threadlet_listener_process and self._threadlet_listener_process.is_alive():
                self._logger.info("Stopping ThreadletListener process...")
                self._stop_event.set()
                self._threadlet_listener_process.join(timeout=5)

                if self._threadlet_listener_process.is_alive():
                    self._logger.warning("ThreadletListener process did not stop gracefully, terminating.")
                    self._threadlet_listener_process.terminate()
                    self._threadlet_listener_process.join(timeout=2)

                    if self._threadlet_listener_process.is_alive():
                        self._logger.error("Failed to terminate ThreadletListener process, killing.")
                        self._threadlet_listener_process.kill()
                        self._threadlet_listener_process.join()
                else:
                    self._logger.info("ThreadletListener process stopped successfully.")

            # Clean up pipe resources
            self._logger.info("Closing pipes...")
            try:
                if self._main_pipe_conn:
                    self._main_pipe_conn.close()
                    self._logger.info("Main pipe connection closed.")
            except Exception as e:
                self._logger.error(f"Error closing main pipe connection: {e}")
            
            # self._listener_pipe_conn is closed by the listener process or when self._main_pipe_conn is closed.
            # If it was passed to another process, that process is responsible for closing its end.
            # However, if the listener process died abruptly, it might be good to try closing it here too,
            # though it might already be closed or raise an error.
            try:
                if self._listener_pipe_conn: # Check if it exists
                     # Check if it's a real connection object and has a close method
                    if hasattr(self._listener_pipe_conn, 'close') and callable(getattr(self._listener_pipe_conn, 'close')):
                        self._listener_pipe_conn.close()
                        self._logger.info("Listener pipe connection closed from main process side (best effort).")
            except Exception as e:
                self._logger.warning(f"Error attempting to close listener pipe connection from main: {e}")


            self._logger.info("Threadlet stopped and resources cleaned up.")

        except Exception as e:
            self._logger.exception(f"Error stopping threadlet: {e}")

    def publish_heartbeat(self, status: str = "active") -> None:
        """Send heartbeat message to ThreadletListener."""
        try:
            heartbeat_message = MessageFactory.create_heartbeat(
                replica_id=self._replica_id,
                status=status
            )
            self._send_message_to_listener(heartbeat_message)
            self._logger.debug(f"Sent heartbeat: {status}")
        except Exception as e:
            self._logger.exception(f"Error sending heartbeat: {e}")

    def publish_metrics(
        self, 
        step: int = 0, 
        epoch: int = 0, 
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        **kwargs
    ) -> None:
        """Send metrics message to ThreadletListener."""
        try:
            metrics_message = MessageFactory.create_metrics(
                replica_id=self._replica_id,
                step=step,
                epoch=epoch,
                loss=loss,
                accuracy=accuracy,
                gradient_norm=gradient_norm,
                **kwargs
            )
            self._send_message_to_listener(metrics_message)
            self._logger.debug(f"Sent metrics for step {step}")
        except Exception as e:
            self._logger.exception(f"Error sending metrics: {e}")

    def publish_status(
        self, 
        status: str = "active", 
        current_step: int = 0, 
        epoch: int = 0, 
        message: str = ""
    ) -> None:
        """Send status message to ThreadletListener."""
        try:
            status_message = MessageFactory.create_status(
                replica_id=self._replica_id,
                status=status,
                current_step=current_step,
                epoch=epoch,
                message=message
            )
            self._send_message_to_listener(status_message)
            self._logger.debug(f"Sent status: {status}")
        except Exception as e:
            self._logger.exception(f"Error sending status: {e}")

    @staticmethod
    def _run_threadlet_listener_process(
        replica_id: str,
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
                replica_id=replica_id,
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

    # Backward compatibility properties
    @property
    def _handlers(self):
        """Backward compatibility: access to handler registry."""
        return self._handler_registry._handlers

    @property
    def _handler_types(self):
        """Backward compatibility: access to handler types."""
        return self._handler_registry._handler_types