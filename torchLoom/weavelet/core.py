"""
Core Weavelet class for process-based configuration management.
"""

import asyncio
import logging
import multiprocessing
import queue
import time
import uuid
from typing import Any, Dict, Optional

from torchLoom.config import Config
from torchLoom.constants import torchLoomConstants

from .handlers import HandlerRegistry
from .listener import WeaveletListener


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
        config_queue: Optional["multiprocessing.Queue[Any]"] = None,
        status_queue: Optional["multiprocessing.Queue[Any]"] = None,
    ):
        # Core identifiers
        self._replica_id = replica_id or f"weavelet:{uuid.uuid4()}"
        self._device_uuid: Optional[str] = None

        # NATS connection setup
        self._torchLoom_addr = torchLoom_addr
        self._stop_event = multiprocessing.Event()

        # Inter-process communication
        self._config_queue = config_queue or multiprocessing.Queue()
        self._status_queue = status_queue or multiprocessing.Queue()

        # Process management
        self._process: Optional[multiprocessing.Process] = None

        # Configuration
        self._nc_timeout = Config.NC_TIMEOUT or 1
        self._exception_sleep = Config.EXCEPTION_RETRY_TIME or 1

        # Enhanced handler system using the new registry
        self._handler_registry = HandlerRegistry()
        self._auto_dispatch = True
        
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
            @weavelet.handler("optimizer_type")
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
        """Check for configuration updates, consolidate to the latest for each key, and apply them automatically.

        Retrieves all pending updates from the queue, determines the latest value
        for each configuration key, and dispatches handlers only for these latest values.

        Returns:
            True if any updates were found and applied, False otherwise.
        """
        latest_updates: Dict[str, Any] = {}
        updates_found = False

        while True:
            try:
                # Get all pending updates from the queue
                config_update = self._config_queue.get_nowait()
                updates_found = True # Mark that at least one update was found

                # Consolidate updates, keeping the latest value for each key
                if config_update:
                    for key, value in config_update.items():
                        latest_updates[key] = value

            except queue.Empty:
                # Queue is empty, stop processing
                break
            except Exception as e:
                # Log any other errors during queue processing
                self._logger.warning(f"Error getting config update from queue: {e}")
                # Continue processing other items if possible, or break depending on severity
                # For now, let's break to avoid infinite loops on persistent errors
                break


        # If any updates were found and consolidated, dispatch handlers for the latest values
        if latest_updates:
            self._dispatch_handlers(latest_updates)
            return True # Return True if updates were processed

        return False # Return False if no updates were found in the queue

    def enable_auto_dispatch(self) -> None:
        """Enable automatic handler dispatch."""
        self._auto_dispatch = True

    def disable_auto_dispatch(self) -> None:
        """Disable automatic handler dispatch."""
        self._auto_dispatch = False

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
        config_queue: "multiprocessing.Queue[Any]",
        status_queue: "multiprocessing.Queue[Any]",
        stop_event: "multiprocessing.Event",
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