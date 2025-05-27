"""
Simple WebSocket server for torchLoom UI integration.

This module provides WebSocket connection management and message handling.
UI commands are forwarded to the Weaver's handler system.
"""

import asyncio
import json
import time
from typing import Any, Callable, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from torchLoom.common.constants import UINetworkConstants
from torchLoom.log.logger import setup_logger

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

    async def send_json_to_all(self, data: dict):
        """Send JSON data to all connected clients."""
        if not self.active_connections:
            return

        message = json.dumps(data)
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


class WebSocketServer:
    """WebSocket server for torchLoom UI communication."""

    def __init__(
        self,
        host=UINetworkConstants.DEFAULT_UI_HOST,
        port=UINetworkConstants.DEFAULT_UI_PORT,
    ):
        self.app = FastAPI(title="torchLoom UI WebSocket API", version="1.0.0")
        self.host = host
        self.port = port
        self.manager = ConnectionManager()

        # Callback function to handle UI commands - will be set by Weaver
        self._ui_command_handler: Optional[Callable[[dict], Any]] = None

        # Callback function to get initial status data - will be set by Weaver
        self._get_initial_status: Optional[Callable[[], dict]] = None

        # CORS for WebSocket connections
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=UINetworkConstants.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.setup_routes()
        logger.info(f"WebSocket server initialized on {host}:{port}")

    def set_ui_command_handler(self, handler: Callable[[dict], Any]):
        """Set the callback function to handle UI commands."""
        self._ui_command_handler = handler
        logger.debug("UI command handler set for WebSocket server")

    def set_initial_status_provider(self, provider: Callable[[], dict]):
        """Set the callback function to get initial status data."""
        self._get_initial_status = provider
        logger.debug("Initial status provider set for WebSocket server")

    async def send_to_all(self, data: dict):
        """Send data to all connected WebSocket clients."""
        await self.manager.send_json_to_all(data)

    def setup_routes(self):
        """Setup FastAPI WebSocket route."""

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.manager.connect(websocket)
            try:
                # Send initial status upon connection
                if self._get_initial_status:
                    initial_status = self._get_initial_status()
                    await websocket.send_text(
                        json.dumps({"type": "status_update", "data": initial_status})
                    )
                    logger.info(
                        "Sent initial status to newly connected WebSocket client."
                    )
                else:
                    logger.warning(
                        "Initial status provider not set, sending empty initial status"
                    )
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "status_update",
                                "data": {
                                    "devices": [],
                                    "training_status": [],
                                    "timestamp": int(time.time()),
                                },
                            }
                        )
                    )

                # Handle incoming messages
                while True:
                    try:
                        data = await websocket.receive_text()
                        await self.handle_ui_message(data, websocket)
                    except WebSocketDisconnect:
                        break

            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected.")
            except Exception as e:
                logger.exception(f"WebSocket error: {e}")
            finally:
                self.manager.disconnect(websocket)

    async def handle_ui_message(self, message: str, websocket: WebSocket):
        """Handle UI messages and forward commands to Weaver's handler system."""
        try:
            data = json.loads(message)
            command_type = data.get("type")

            if command_type == "ping":
                await websocket.send_text(
                    json.dumps({"type": "pong", "timestamp": time.time()})
                )
            elif command_type in ["ui_command"]:
                # Forward UI commands to Weaver's handler system
                if self._ui_command_handler:
                    try:
                        await self._ui_command_handler(data)
                        logger.info(f"Forwarded UI command: {command_type}")
                    except Exception as e:
                        logger.error(f"Error handling UI command {command_type}: {e}")
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "error",
                                    "message": f"Failed to process command: {command_type}",
                                    "error": str(e),
                                }
                            )
                        )
                else:
                    logger.warning(
                        f"No UI command handler set for command: {command_type}"
                    )
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "UI command handler not available",
                            }
                        )
                    )

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON received via WebSocket: {message}")
            await websocket.send_text(
                json.dumps({"type": "error", "message": "Invalid JSON format"})
            )
        except Exception as e:
            logger.exception(f"Error handling WebSocket message: {e}")
            await websocket.send_text(
                json.dumps({"type": "error", "message": f"Server error: {str(e)}"})
            )

    async def start_server(self):
        """Start the WebSocket server."""
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level="info"
        )
        server = uvicorn.Server(config)
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        await server.serve()

    async def run_server(self):
        """Run the WebSocket server."""
        try:
            await self.start_server()  # This will block
        except KeyboardInterrupt:
            logger.info("WebSocket server shutting down due to KeyboardInterrupt...")
        except Exception as e:
            logger.exception(f"Error running WebSocket server: {e}")
        finally:
            logger.info("WebSocket server stopped.")
