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
from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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

        # Setup static files and templates for web GUI
        try:
            self.app.mount(
                "/static",
                StaticFiles(directory="torchloom_web_gui/static"),
                name="static",
            )
            self.templates = Jinja2Templates(directory="torchloom_web_gui/templates")
        except Exception as e:
            logger.warning(
                f"Could not mount static files or templates: {e}. Web GUI may not work properly."
            )
            self.templates = None

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
        """Setup FastAPI WebSocket route and HTTP routes for web GUI."""

        @self.app.get("/", response_class=HTMLResponse)
        async def get_web_gui(request: Request):
            """Serve the web GUI HTML page."""
            if not self.templates:
                return HTMLResponse(
                    "<h1>TorchLoom Control Panel</h1><p>Web GUI templates not available. Please ensure torchloom_web_gui/templates directory exists.</p>"
                )

            return self.templates.TemplateResponse("index.html", {"request": request})

        @self.app.post("/send_command", response_class=HTMLResponse)
        async def send_command_endpoint(
            request: Request,
            command_type: str = Form(...),
            process_id: Optional[str] = Form(None),
            params: Optional[str] = Form(None),
        ):
            """Handle command form submissions from the web GUI."""
            if not self._ui_command_handler:
                logger.error("Cannot send command: UI command handler not set.")
                return HTMLResponse(
                    "<p class='error'>❌ Error: UI command handler not available.</p>",
                    status_code=503,
                )

            command_data = {
                "type": "ui_command",
                "data": {
                    "command_type": command_type,
                },
            }
            if process_id and process_id.strip():
                command_data["data"]["process_id"] = process_id.strip()

            parsed_params = {}
            if params and params.strip():
                try:
                    parsed_params = json.loads(params)
                    if not isinstance(parsed_params, dict):
                        raise ValueError("Params must be a JSON object.")
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in params: {params}")
                    return HTMLResponse(
                        f"<p class='error'>❌ Error: Invalid JSON in parameters: {str(e)}</p>",
                        status_code=400,
                    )
                except ValueError as ve:
                    logger.error(str(ve))
                    return HTMLResponse(
                        f"<p class='error'>❌ Error: {str(ve)}</p>", status_code=400
                    )

            command_data["data"]["params"] = parsed_params

            try:
                # Handle both sync and async command handlers
                import inspect

                if inspect.iscoroutinefunction(self._ui_command_handler):
                    await self._ui_command_handler(command_data)
                else:
                    self._ui_command_handler(command_data)

                logger.info(
                    f"Processed command from web GUI: {command_type} for {process_id if process_id else 'all'}"
                )
                return HTMLResponse(
                    f"<p class='success'>✅ Command '{command_type}' sent successfully.</p>"
                )
            except Exception as e:
                logger.error(f"Failed to process command from web GUI: {e}")
                return HTMLResponse(
                    f"<p class='error'>❌ Error processing command: {str(e)}</p>",
                    status_code=500,
                )

        @self.app.get("/api/processes")
        async def get_processes():
            """Get available process IDs and their configurations for the UI dropdown."""
            try:
                if self._get_initial_status:
                    status_data = self._get_initial_status()
                    processes = []
                    
                    # Extract process information from training status
                    training_status = status_data.get("training_status", [])
                    for status in training_status:
                        process_id = status.get("process_id")
                        if process_id and process_id != "N/A":
                            config = status.get("config", {})
                            processes.append({
                                "process_id": process_id,
                                "config": config,
                                "status": status.get("status", "unknown"),
                                "type": "training"
                            })
                    
                    # Extract process information from devices
                    devices = status_data.get("devices", [])
                    for device in devices:
                        process_id = device.get("process_id")
                        if process_id and process_id != "N/A":
                            # Check if we already have this process from training status
                            existing = next((p for p in processes if p["process_id"] == process_id), None)
                            if not existing:
                                config = device.get("config", {})
                                processes.append({
                                    "process_id": process_id,
                                    "config": config,
                                    "status": "device",
                                    "type": "device",
                                    "device_uuid": device.get("device_uuid", "N/A")
                                })
                    
                    return {"processes": processes}
                else:
                    return {"processes": []}
            except Exception as e:
                logger.error(f"Error fetching processes: {e}")
                return {"error": str(e), "processes": []}

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
                        # Handle both sync and async command handlers
                        import inspect

                        if inspect.iscoroutinefunction(self._ui_command_handler):
                            await self._ui_command_handler(data)
                        else:
                            self._ui_command_handler(data)
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
