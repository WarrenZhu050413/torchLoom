"""
UI interface for the torchLoom Weaver.

This module handles all UI-specific functionality including status updates,
WebSocket communication, and UI data formatting.
"""

import asyncio
import time
from typing import Any, Callable, Optional

from torchLoom.common.publishers import BasePublisher
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

logger = setup_logger(name="ui_interface")

class UINotificationManager:
    """
    Manages UI notifications and WebSocket broadcasting.

    This class handles all status broadcasting logic using a direct
    send function from the WebSocket server.
    """

    def __init__(self):
        self._websocket_send_func: Optional[Callable[[dict], Any]] = None
        self._status_tracker = None
        self._broadcast_task = None
        self._stop_broadcaster = asyncio.Event()

    def set_websocket_send_func(self, send_func: Callable[[dict], Any]):
        """Set the WebSocket send function for broadcasting messages."""
        try:
            self._websocket_send_func = send_func
            logger.debug("WebSocket send function set successfully")
        except Exception as e:
            logger.error(f"Failed to set WebSocket send function: {e}")

    def set_status_tracker(self, status_tracker):
        """Set the status tracker for data access."""
        try:
            self._status_tracker = status_tracker
            logger.debug("Status tracker set successfully")
        except Exception as e:
            logger.error(f"Failed to set status tracker: {e}")

    def notify_status_change(self):
        """Called by StatusTracker when data changes - triggers immediate broadcast."""
        try:
            # Trigger immediate broadcast to all connected clients
            if self._websocket_send_func:
                # Schedule immediate broadcast (non-blocking)
                asyncio.create_task(self._immediate_broadcast())
        except Exception as e:
            logger.warning(f"Failed to notify status change: {e}")

    async def _immediate_broadcast(self):
        """Immediate status broadcast triggered by status changes."""
        try:
            await self.broadcast_status_update()
        except Exception as e:
            logger.error(f"Failed in immediate broadcast: {e}")

    def format_status_data_for_ui(self, status_tracker=None):
        """
        Centralized method to format protobuf status data for UI JSON serialization.

        This is the single source of truth for converting protobuf data to dict format
        for WebSocket communication and UI display.
        """
        try:
            # Use provided status_tracker or fall back to instance variable
            tracker = status_tracker or self._status_tracker
            if not tracker:
                raise ValueError("No status tracker available")

            # Get raw protobuf data for UI
            ui_snapshot = tracker.get_ui_status_snapshot()

            # Convert protobuf to dict for JSON serialization
            return {
                "devices": [
                    {
                        "device_uuid": device.device_uuid,
                        "process_id": device.process_id,
                        "server_id": device.server_id,
                        "utilization": device.utilization,
                        "temperature": device.temperature,
                        "memory_used": device.memory_used,
                        "memory_total": device.memory_total,
                    }
                    for device in ui_snapshot.devices
                ],
                "training_status": [
                    {
                        "process_id": training.process_id,
                        "current_step": training.current_step,
                        "epoch": training.epoch,
                        "metrics": dict(training.metrics),
                        "training_time": training.training_time,
                        "max_step": training.max_step,
                        "max_epoch": training.max_epoch,
                        "config": dict(training.config),
                    }
                    for training in ui_snapshot.training_status
                ],
                "timestamp": ui_snapshot.timestamp,
            }
        except Exception as e:
            logger.error(f"Failed to format status data for UI: {e}")
            return {"devices": [], "training_status": [], "timestamp": int(time.time())}

    async def broadcast_status_update(self):
        """Broadcast status update to all connected WebSocket clients."""
        try:
            if self._websocket_send_func:
                # Format the status data
                status_data = self.format_status_data_for_ui()

                # Send to all connected clients via the websocket send function
                await self._websocket_send_func(
                    {"type": "status_update", "data": status_data}
                )
                logger.debug("Broadcasted status update to all connected clients")
        except Exception as e:
            logger.error(f"Failed to broadcast status update: {e}")

    async def start_status_broadcaster(self):
        """
        Start the periodic status broadcaster.

        This runs in a background task and periodically broadcasts status updates
        to all connected WebSocket clients.
        """
        from torchLoom.common.constants import TimeConstants

        logger.info("Starting periodic UI status broadcaster.")

        while not self._stop_broadcaster.is_set():
            try:
                if self._websocket_send_func:
                    await self.broadcast_status_update()

                # Wait for the broadcast interval or stop signal
                try:
                    await asyncio.wait_for(
                        self._stop_broadcaster.wait(),
                        timeout=TimeConstants.STATUS_BROADCAST_IN,
                    )
                    break  # Stop signal received
                except asyncio.TimeoutError:
                    pass  # Continue with next broadcast

            except asyncio.CancelledError:
                logger.info("Status broadcaster task cancelled.")
                break
            except Exception as e:
                logger.exception(f"Error in status broadcaster: {e}")
                try:
                    await asyncio.wait_for(
                        self._stop_broadcaster.wait(),
                        timeout=TimeConstants.EXCEPTION_SLEEP,
                    )
                    break
                except asyncio.TimeoutError:
                    pass

        logger.info("UI status broadcaster stopped.")

    def start_broadcaster_task(self):
        """Start the broadcaster as a background task."""
        if self._broadcast_task is None or self._broadcast_task.done():
            self._stop_broadcaster.clear()
            self._broadcast_task = asyncio.create_task(self.start_status_broadcaster())
            logger.info("Started UI status broadcaster task")
        else:
            logger.warning("Broadcaster task already running")

    async def stop_broadcaster(self):
        """Stop the status broadcaster."""
        logger.info("Stopping UI status broadcaster...")
        self._stop_broadcaster.set()

        if self._broadcast_task and not self._broadcast_task.done():
            try:
                await asyncio.wait_for(self._broadcast_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Broadcaster task did not stop gracefully, cancelling..."
                )
                self._broadcast_task.cancel()
                try:
                    await self._broadcast_task
                except asyncio.CancelledError:
                    logger.info("Broadcaster task cancelled successfully")
            except Exception as e:
                logger.exception(f"Error stopping broadcaster task: {e}")

    def get_status_data_for_initial_connection(self):
        """Get formatted status data for initial WebSocket connection."""
        return self.format_status_data_for_ui()
