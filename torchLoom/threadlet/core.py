"""
Core Threadlet class for process-based configuration management.
"""

import asyncio
import logging
import multiprocessing
import time
import uuid
from multiprocessing.connection import Connection
from typing import Any, Dict, Optional, Tuple, Type

from torchLoom.common.config import Config
from torchLoom.common.constants import torchLoomConstants

from .handlers import HandlerRegistry
from .listener import ThreadletListener


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
        config_pipe: Optional[
            Tuple[
                Connection,
                Connection,
            ]
        ] = None,
        status_pipe: Optional[
            Tuple[
                Connection,
                Connection,
            ]
        ] = None,
    ):
        # Core identifiers
        self._replica_id = replica_id or f"threadlet:{uuid.uuid4()}"
        self._device_uuid: Optional[str] = None

        # NATS connection setup
        self._torchLoom_addr = torchLoom_addr
        self._stop_event = multiprocessing.Event()

        # Inter-process communication using pipes
        # Config pipe: listener -> main (listener sends config updates to main process)
        if config_pipe is None:
            self._config_receiver, self._config_sender = multiprocessing.Pipe(
                duplex=False
            )
        else:
            self._config_receiver, self._config_sender = config_pipe

        # Status pipe: main -> listener (main sends status updates to listener)
        if status_pipe is None:
            self._status_receiver, self._status_sender = multiprocessing.Pipe(
                duplex=False
            )
        else:
            self._status_receiver, self._status_sender = status_pipe

        # Process management
        self._process: Optional[multiprocessing.Process] = None

        # Configuration
        self._nc_timeout = Config.NC_TIMEOUT or 1
        self._exception_sleep = Config.EXCEPTION_RETRY_TIME or 1

        # Enhanced handler system using the new registry
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
            expected_type: Expected type for the parameter value (inferred if not provided)
        """
        self._handler_registry.register_handler(config_key, handler, expected_type)

    def handler(self, config_key: str, expected_type=None):
        """Decorator for registering configuration handlers.

        Args:
            config_key: The configuration parameter name
            expected_type: Expected type for the parameter value

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

    def get_config_update(self, timeout: float = 0.1) -> Optional[Dict[str, str]]:
        """Get configuration update from the threadlet process if available.

        If auto_dispatch is enabled (default), this will automatically call
        registered handlers. Otherwise, it returns the config update dict.
        """
        try:
            if self._config_receiver.poll(timeout):
                config_update = self._config_receiver.recv()

                if config_update and self._auto_dispatch:
                    # Automatically dispatch to registered handlers
                    self._dispatch_handlers(config_update)
                    return None  # No need to return since handlers were called

                return config_update
        except EOFError:
            # Pipe has been closed
            return None
        except Exception as e:
            self._logger.warning(f"Error getting config update from pipe: {e}")
            return None

        return None

    def check_and_apply_updates(self) -> bool:
        """Check for configuration updates, consolidate to the latest for each key, and apply them automatically.

        Retrieves all pending updates from the pipe, determines the latest value
        for each configuration key, and dispatches handlers only for these latest values.

        Returns:
            True if any updates were found and applied, False otherwise.
        """
        latest_updates: Dict[str, Any] = {}
        updates_found = False

        while True:
            try:
                # Check if data is available without blocking
                if self._config_receiver.poll(0):
                    config_update = self._config_receiver.recv()
                    updates_found = True

                    # Consolidate updates, keeping the latest value for each key
                    if config_update:
                        for key, value in config_update.items():
                            latest_updates[key] = value
                else:
                    # No more data available
                    break

            except EOFError:
                # Pipe has been closed
                break
            except Exception as e:
                # Log any other errors during pipe processing
                self._logger.warning(f"Error getting config update from pipe: {e}")
                break

        # If any updates were found and consolidated, dispatch handlers for the latest values
        if latest_updates:
            self._dispatch_handlers(latest_updates)
            return True  # Return True if updates were processed

        return False  # Return False if no updates were found

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

            self._process = multiprocessing.Process(
                target=self._run_threadlet_listener_process,
                args=(
                    self._replica_id,
                    self._torchLoom_addr,
                    self._config_sender,  # Send side of config pipe
                    self._status_receiver,  # Receive side of status pipe
                    self._stop_event,
                ),
                name=f"threadlet-{self._replica_id}",
            )
            self._process.start()

            # Give the process a moment to start
            time.sleep(0.1)

            print(f"Threadlet process started with PID: {self._process.pid}")

            # Log registered handlers
            handlers = self.get_registered_handlers()
            print(f"Threadlet has {len(handlers)} registered configuration handlers")
        except Exception as e:
            print(f"Failed to start threadlet process: {e}")
            raise

    def stop(self) -> None:
        """Stop the threadlet process and clean up resources."""
        try:
            if self._process and self._process.is_alive():
                print("Stopping threadlet process")
                self._stop_event.set()

                # Wait for the process to finish gracefully
                self._process.join(timeout=5)

                # If still alive, terminate forcefully
                if self._process.is_alive():
                    print("Force terminating threadlet process")
                    self._process.terminate()
                    self._process.join(timeout=2)

                    # Last resort - kill
                    if self._process.is_alive():
                        self._process.kill()
                        self._process.join()

                print("Threadlet process stopped successfully")

            # Clean up pipe resources
            try:
                print("Closing pipes...")

                # Close config pipe connections
                if hasattr(self, "_config_receiver") and self._config_receiver:
                    try:
                        self._config_receiver.close()
                        print("Config receiver closed")
                    except Exception as e:
                        print(f"Error closing config receiver: {e}")

                if hasattr(self, "_config_sender") and self._config_sender:
                    try:
                        self._config_sender.close()
                        print("Config sender closed")
                    except Exception as e:
                        print(f"Error closing config sender: {e}")

                # Close status pipe connections
                if hasattr(self, "_status_receiver") and self._status_receiver:
                    try:
                        self._status_receiver.close()
                        print("Status receiver closed")
                    except Exception as e:
                        print(f"Error closing status receiver: {e}")

                if hasattr(self, "_status_sender") and self._status_sender:
                    try:
                        self._status_sender.close()
                        print("Status sender closed")
                    except Exception as e:
                        print(f"Error closing status sender: {e}")

                print("Multiprocessing resources cleaned up")

            except Exception as e:
                print(f"Error cleaning up multiprocessing resources: {e}")

        except Exception as e:
            print(f"Error stopping threadlet process: {e}")

    def publish_status(self, status: Dict[str, Any]) -> None:
        """
        Publish status update.

        This method accepts any type of status (TrainingStatus, deviceStatus)
        and sends it to the weaver via NATS messaging.

        Args:
            status_dict: Status data as dictionary, should be a TrainingStatus.to_dict(),
                        deviceStatus.to_dict(), or any compatible dictionary.
        """
        try:
            if self._status_sender and not self._status_sender.closed:
                self._status_sender.send(status)
        except (BrokenPipeError, OSError):
            # Pipe is broken or closed, ignore
            pass
        except Exception as e:
            self._logger.warning(f"Error sending status via pipe: {e}")

    @staticmethod
    def _run_threadlet_listener_process(
        replica_id: str,
        torchLoom_addr: str,
        config_sender: Connection,
        status_receiver: Connection,
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
                config_sender=config_sender,
                status_receiver=status_receiver,
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

    def _validate_and_convert_value(self, config_key: str, value, expected_type=None):
        """Backward compatibility: validate and convert value."""
        if expected_type is None:
            expected_type = self._handler_registry.get_handler_type(config_key)
            if expected_type is None:
                return value
        return self._handler_registry._type_converter.validate_and_convert_value(
            config_key, value, expected_type
        )
