"""
WebSocket server for torchLoom UI integration.

This module provides real-time bidirectional communication between the
Weaver backend and the Vue.js frontend using WebSockets and REST APIs.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from torchLoom.common.constants import (
    NetworkConstants,
    TimeConstants,
    torchLoomConstants,
)
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope, UICommand

logger = setup_logger(name="websocket_server")


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"WebSocket connected. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(
                f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
            )

    async def send_to_all(self, message: str):
        """Send message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to connection: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

    async def send_json_to_all(self, data: dict):
        """Send JSON data to all connected clients."""
        message = json.dumps(data)
        await self.send_to_all(message)


class WebSocketServer:
    """Simplified WebSocket server for torchLoom UI, using direct broadcast."""

    def __init__(
        self,
        status_tracker,
        weaver,
        host=NetworkConstants.DEFAULT_UI_HOST,
        port=NetworkConstants.DEFAULT_UI_PORT,
    ):
        self.app = FastAPI(title="torchLoom UI WebSocket API", version="1.0.0")
        self.status_tracker = status_tracker
        self.weaver = weaver
        self.host = host
        self.port = port
        self.manager = ConnectionManager()

        # CORS might still be needed for WebSocket connections from a different origin
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=NetworkConstants.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],  # Allow all methods for WebSocket
            allow_headers=["*"],  # Allow all headers
        )

        self.setup_routes()
        logger.info(f"WebSocket server initialized on {host}:{port}")

    def setup_routes(self):
        """Setup FastAPI WebSocket route."""

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.manager.connect(websocket)
            try:
                # Send initial status upon connection
                initial_status = self.get_ui_status_dict()
                await websocket.send_text(
                    json.dumps({"type": "status_update", "data": initial_status})
                )
                logger.info("Sent initial status to newly connected WebSocket client.")

                while True:
                    data = await websocket.receive_text()
                    await self.handle_websocket_message(data, websocket)
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected.")
            except Exception as e:
                logger.exception(f"WebSocket error: {e}")
            finally:
                self.manager.disconnect(websocket)

        # All REST API endpoints are removed.

    def get_ui_status_dict(self) -> dict:
        """Convert status tracker data to UI-friendly format."""
        replica_groups = {}
        for device_id, device in self.status_tracker.devices.items():
            replica_id = device.replica_id
            group_id = replica_id.split("_")[0] if "_" in replica_id else replica_id
            if group_id not in replica_groups:
                replica_groups[group_id] = {
                    "id": group_id,
                    "devices": {},
                    "status": "training",
                    "stepProgress": 0,
                    "fixedStep": None,
                    "lastActiveStep": None,
                }
            replica_groups[group_id]["devices"][device_id] = {
                "id": device_id,
                "server": device.server_id,
                "status": device.status,
                "utilization": round(getattr(device, "utilization", 0), 1),
                "temperature": round(getattr(device, "temperature", 0), 1),
                "memory_used": round(getattr(device, "memory_used", 0), 2),
                "memory_total": round(getattr(device, "memory_total", 8.0), 2),
                "batch": device.config.get("batch_size", "32"),
                "lr": device.config.get("learning_rate", "0.001"),
                "opt": device.config.get("optimizer_type", "Adam"),
                "last_updated": getattr(device, "last_updated", time.time()),
            }
        for replica_id, replica in self.status_tracker.replicas.items():
            group_id = replica_id.split("_")[0] if "_" in replica_id else replica_id
            if group_id in replica_groups:
                replica_groups[group_id]["status"] = replica.status
                replica_groups[group_id]["stepProgress"] = round(
                    replica.step_progress, 1
                )
                replica_groups[group_id]["lastActiveStep"] = replica.last_active_step
                if replica.fixed_step is not None:
                    replica_groups[group_id]["fixedStep"] = replica.fixed_step

        system_summary = (
            self.status_tracker.get_system_summary()
            if hasattr(self.status_tracker, "get_system_summary")
            else {}
        )

        return {
            "replicaGroups": replica_groups,
            "communicationStatus": (
                self.status_tracker.communication_status
                if hasattr(self.status_tracker, "communication_status")
                else "unknown"
            ),
            "systemSummary": system_summary,
            "timestamp": time.time(),
        }

    async def handle_websocket_message(self, message: str, websocket: WebSocket):
        """Handle incoming WebSocket messages from UI."""
        try:
            data = json.loads(message)
            command_type = data.get("type")

            if command_type == "deactivate_device":
                await self.handle_deactivate_device(data.get("device_id"))
            elif command_type == "reactivate_group":
                await self.handle_reactivate_group(data.get("replica_id"))
            elif command_type == "update_config":
                await self.handle_config_update(
                    data.get("replica_id"), data.get("config_params", {})
                )
            elif command_type == "ping":
                await websocket.send_text(
                    json.dumps({"type": "pong", "timestamp": time.time()})
                )
            else:
                logger.warning(f"Unknown command type from WebSocket: {command_type}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received via WebSocket: {message}")
        except Exception as e:
            logger.exception(f"Error handling WebSocket message: {e}")

    async def handle_deactivate_device(self, device_id: str):
        """Handle device deactivation: update local status (optional) and publish UICommand."""
        if not device_id:
            logger.warning("Deactivate device command received with no device_id.")
            return

        # Optional: Update local status_tracker for immediate UI feedback if desired,
        # but the authoritative state comes from the Weaver via UI_UPDATE.
        # self.status_tracker.set_communication_status("rebuilding") # Example
        # self.status_tracker.update_training_progress(replica_id, status="deactivating") # Example
        # self.status_tracker.deactivate_device(device_id) # Example

        logger.info(f"Processing device deactivation command for: {device_id}")
        if (
            self.weaver
            and self.weaver._subscription_manager
            and self.weaver._subscription_manager.js
        ):
            try:
                env = EventEnvelope()
                env.ui_command.command_type = "deactivate_device"
                env.ui_command.target_id = device_id
                await self.weaver._subscription_manager.js.publish(
                    torchLoomConstants.subjects.UI_COMMANDS, env.SerializeToString()
                )
                logger.debug(
                    f"Published UI_COMMAND (deactivate_device) for {device_id} to NATS JetStream."
                )
            except Exception as e:
                logger.exception(
                    f"Failed to publish UI_COMMAND (deactivate_device) to NATS: {e}"
                )
        else:
            logger.warning(
                "Cannot publish deactivate_device: Weaver, SubscriptionManager, or JetStream not available."
            )
        # Optional: self.status_tracker.set_communication_status("stable") # Example

    async def handle_reactivate_group(self, replica_id: str):
        """Handle group reactivation: update local status (optional) and publish UICommand."""
        if not replica_id:
            logger.warning("Reactivate group command received with no replica_id.")
            return

        logger.info(f"Processing replica group reactivation command for: {replica_id}")
        if (
            self.weaver
            and self.weaver._subscription_manager
            and self.weaver._subscription_manager.js
        ):
            try:
                env = EventEnvelope()
                env.ui_command.command_type = "reactivate_group"
                env.ui_command.target_id = replica_id
                await self.weaver._subscription_manager.js.publish(
                    torchLoomConstants.subjects.UI_COMMANDS, env.SerializeToString()
                )
                logger.debug(
                    f"Published UI_COMMAND (reactivate_group) for {replica_id} to NATS JetStream."
                )
            except Exception as e:
                logger.exception(
                    f"Failed to publish UI_COMMAND (reactivate_group) to NATS: {e}"
                )
        else:
            logger.warning(
                "Cannot publish reactivate_group: Weaver, SubscriptionManager, or JetStream not available."
            )

    async def handle_config_update(self, replica_id: str, config_params: dict):
        """Handle config update: update local status (optional) and publish UICommand."""
        if not replica_id:
            logger.warning("Update config command received with no replica_id.")
            return

        logger.info(
            f"Processing config update command for {replica_id}: {config_params}"
        )
        if (
            self.weaver
            and self.weaver._subscription_manager
            and self.weaver._subscription_manager.js
        ):
            try:
                env = EventEnvelope()
                env.ui_command.command_type = "update_config"
                env.ui_command.target_id = replica_id
                for key, value in config_params.items():
                    env.ui_command.params[key] = str(value)
                await self.weaver._subscription_manager.js.publish(
                    torchLoomConstants.subjects.UI_COMMANDS, env.SerializeToString()
                )
                logger.debug(
                    f"Published UI_COMMAND (update_config) for {replica_id} to NATS JetStream."
                )
            except Exception as e:
                logger.exception(
                    f"Failed to publish UI_COMMAND (update_config) to NATS: {e}"
                )
        else:
            logger.warning(
                "Cannot publish update_config: Weaver, SubscriptionManager, or JetStream not available."
            )

    async def broadcast_status_update(self):
        """Broadcast status update to all connected WebSocket clients."""
        if self.manager.active_connections:
            status_data = self.get_ui_status_dict()
            await self.manager.send_json_to_all(
                {"type": "status_update", "data": status_data}
            )
            # Consider if broadcast_individual_status_updates logic needs to be merged or called here
            # For simplicity, sticking to one main 'status_update' type from this server.

    async def start_status_broadcaster(self):
        """Start periodic status broadcasts to WebSocket clients."""
        logger.info("Starting WebSocket status broadcaster using StatusTracker.")
        while True:  # Should be controlled by an event or server lifecycle
            try:
                if self.manager.active_connections:
                    await self.broadcast_status_update()
                await asyncio.sleep(TimeConstants.STATUS_BROADCAST_IN)
            except asyncio.CancelledError:
                logger.info("Status broadcaster task cancelled.")
                break
            except Exception as e:
                logger.exception(f"Error in status broadcaster: {e}")
                await asyncio.sleep(TimeConstants.ERROR_RETRY_SLEEP)

    async def start_server(self):
        """Start the WebSocket server."""
        # Note: start_status_broadcaster should be started concurrently if this method is blocking.
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level="info"
        )
        server = uvicorn.Server(config)
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        await server.serve()

    async def run_with_status_broadcaster(self):
        """Run server with status broadcaster in parallel."""
        broadcaster_task = None
        try:
            broadcaster_task = asyncio.create_task(self.start_status_broadcaster())
            await self.start_server()  # This will block
        except KeyboardInterrupt:
            logger.info("WebSocket server shutting down due to KeyboardInterrupt...")
        except Exception as e:
            logger.exception(f"Error running WebSocket server with broadcaster: {e}")
        finally:
            if broadcaster_task and not broadcaster_task.done():
                broadcaster_task.cancel()
                try:
                    await broadcaster_task
                except asyncio.CancelledError:
                    logger.info(
                        "Status broadcaster task successfully cancelled on shutdown."
                    )
            logger.info("WebSocket server stopped.")


# Example of how this might be run (typically in weaver.py or a main script):
# async def main():
#     status_tracker = StatusTracker() # Assuming StatusTracker exists
#     weaver_mock = MagicMock() # Mock weaver for example
#     weaver_mock._subscription_manager = MagicMock()
#     weaver_mock._subscription_manager.nc = MagicMock() # Mock NATS client
#     weaver_mock._subscription_manager.js = MagicMock() # Mock JetStream client
#
#     ws_server = WebSocketServer(status_tracker=status_tracker, weaver=weaver_mock)
#     await ws_server.run_with_status_broadcaster() # or await ws_server.start_server()
#
# if __name__ == "__main__":
#     asyncio.run(main())
