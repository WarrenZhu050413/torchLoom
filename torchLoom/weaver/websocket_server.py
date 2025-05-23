"""
WebSocket server for torchLoom UI integration.

This module provides real-time bidirectional communication between the
Weaver backend and the Vue.js frontend using WebSockets and REST APIs.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from torchLoom.common.constants import torchLoomConstants, TimeoutConstants
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
            self.active_connections.remove(conn)

    async def send_json_to_all(self, data: dict):
        """Send JSON data to all connected clients."""
        message = json.dumps(data)
        await self.send_to_all(message)


class WebSocketServer:
    """WebSocket server for torchLoom UI."""

    def __init__(self, status_tracker, weaver=None, nats_client=None, host="0.0.0.0", port=8080):
        self.app = FastAPI(title="torchLoom UI API", version="1.0.0")
        self.status_tracker = status_tracker
        self.weaver = weaver  # Direct reference to weaver for command handling
        self.nats_client = nats_client  # Keep for backward compatibility, but prefer direct calls
        self.host = host
        self.port = port
        self.manager = ConnectionManager()

        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:5173",
                "http://localhost:3000",
                "http://127.0.0.1:5173",
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.setup_routes()
        logger.info(f"WebSocket server initialized on {host}:{port}")

    def setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.manager.connect(websocket)
            try:
                # Send initial status with consistent message format
                initial_status = self.get_ui_status_dict()
                await websocket.send_text(
                    json.dumps({"type": "status_update", "data": initial_status})
                )

                # Listen for client messages
                while True:
                    data = await websocket.receive_text()
                    await self.handle_websocket_message(data, websocket)

            except WebSocketDisconnect:
                self.manager.disconnect(websocket)
            except Exception as e:
                logger.exception(f"WebSocket error: {e}")
                self.manager.disconnect(websocket)

        @self.app.get("/api/status")
        async def get_status():
            """Get current system status."""
            return JSONResponse(self.get_ui_status_dict())

        @self.app.post("/api/commands/deactivate-gpu")
        async def deactivate_gpu(request: dict):
            """Deactivate a specific GPU."""
            gpu_id = request.get("gpu_id")
            if not gpu_id:
                raise HTTPException(status_code=400, detail="gpu_id required")

            await self.handle_deactivate_gpu(gpu_id)
            return {
                "status": "success",
                "message": f"GPU {gpu_id} deactivation initiated",
            }

        @self.app.post("/api/commands/reactivate-group")
        async def reactivate_group(request: dict):
            """Reactivate a replica group."""
            replica_id = request.get("replica_id")
            if not replica_id:
                raise HTTPException(status_code=400, detail="replica_id required")

            await self.handle_reactivate_group(replica_id)
            return {
                "status": "success",
                "message": f"Replica group {replica_id} reactivation initiated",
            }

        @self.app.post("/api/commands/update-config")
        async def update_config(request: dict):
            """Update training configuration."""
            replica_id = request.get("replica_id")
            config_params = request.get("config_params", {})

            if not replica_id:
                raise HTTPException(status_code=400, detail="replica_id required")

            await self.handle_config_update(replica_id, config_params)
            return {
                "status": "success",
                "message": f"Configuration updated for {replica_id}",
            }

        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "connections": len(self.manager.active_connections),
                "gpus": len(self.status_tracker.gpus),
                "replicas": len(self.status_tracker.replicas),
            }

    def get_ui_status_dict(self) -> dict:
        """Convert status tracker data to UI-friendly format."""
        # Organize GPUs by replica groups and servers
        replica_groups = {}

        for gpu_id, gpu in self.status_tracker.gpus.items():
            replica_id = gpu.replica_id
            server_id = (
                gpu.server_id
            )  # This field should match StatusTracker's GPUState

            # Extract group ID from replica_id
            group_id = replica_id.split("_")[0] if "_" in replica_id else replica_id

            if group_id not in replica_groups:
                replica_groups[group_id] = {
                    "id": group_id,
                    "gpus": {},
                    "status": "training",
                    "stepProgress": 0,
                    "fixedStep": None,
                    "lastActiveStep": None,
                }

            # Add GPU data with comprehensive info
            replica_groups[group_id]["gpus"][gpu_id] = {
                "id": gpu_id,
                "server": server_id,
                "status": gpu.status,
                "utilization": round(gpu.utilization, 1),
                "temperature": round(gpu.temperature, 1),
                "memory_used": (
                    round(gpu.memory_used, 2) if hasattr(gpu, "memory_used") else 0.0
                ),
                "memory_total": (
                    round(gpu.memory_total, 2) if hasattr(gpu, "memory_total") else 8.0
                ),
                "batch": gpu.config.get("batch_size", "32"),
                "lr": gpu.config.get("learning_rate", "0.001"),
                "opt": gpu.config.get("optimizer_type", "Adam"),
                "last_updated": (
                    gpu.last_updated if hasattr(gpu, "last_updated") else time.time()
                ),
            }

        # Update replica group status from replica tracker
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

        # Include system summary information
        system_summary = self.status_tracker.get_system_summary()

        return {
            "replicaGroups": replica_groups,
            "communicationStatus": self.status_tracker.communication_status,
            "systemSummary": system_summary,
            "timestamp": time.time(),
        }

    async def handle_websocket_message(self, message: str, websocket: WebSocket):
        """Handle incoming WebSocket messages from UI."""
        try:
            data = json.loads(message)
            command_type = data.get("type")

            if command_type == "deactivate_gpu":
                await self.handle_deactivate_gpu(data.get("gpu_id"))
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
                logger.warning(f"Unknown command type: {command_type}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {message}")
        except Exception as e:
            logger.exception(f"Error handling WebSocket message: {e}")

    async def handle_deactivate_gpu(self, gpu_id: str):
        """Handle GPU deactivation command."""
        if gpu_id not in self.status_tracker.gpus:
            logger.warning(f"GPU {gpu_id} not found")
            return

        # Update local status
        replica_id = self.status_tracker.gpus[gpu_id].replica_id
        self.status_tracker.set_communication_status("rebuilding")
        self.status_tracker.update_training_progress(replica_id, status="deactivating")

        # Simulate delay and then deactivate
        await asyncio.sleep(0.8)
        self.status_tracker.deactivate_gpu(gpu_id)

        # Call weaver handler directly (much more efficient than NATS)
        if self.weaver and hasattr(self.weaver, '_handlers'):
            try:
                ui_handler = self.weaver._handlers.get('ui_commands')
                if ui_handler:
                    # Create a mock envelope for the direct call
                    from torchLoom.proto.torchLoom_pb2 import EventEnvelope
                    env = EventEnvelope()
                    env.ui_command.command_type = "deactivate_gpu"
                    env.ui_command.target_id = gpu_id
                    await ui_handler.handle(env)
                    logger.debug(f"Sent UI command directly to weaver: deactivate_gpu for {gpu_id}")
            except Exception as e:
                logger.exception(f"Failed to send direct UI command: {e}")
                # Fallback to NATS if available
                if self.nats_client:
                    await self.send_ui_command("deactivate_gpu", gpu_id)

        # Reset communication status
        await asyncio.sleep(1.2)
        self.status_tracker.set_communication_status("stable")

        logger.info(f"Processed GPU deactivation: {gpu_id}")

    async def handle_reactivate_group(self, replica_id: str):
        """Handle replica group reactivation command."""
        # Update local status
        self.status_tracker.set_communication_status("rebuilding")
        self.status_tracker.update_training_progress(replica_id, status="activating")

        # Simulate delay and then reactivate
        await asyncio.sleep(0.8)
        self.status_tracker.reactivate_replica_group(replica_id)

        # Call weaver handler directly (much more efficient than NATS)
        if self.weaver and hasattr(self.weaver, '_handlers'):
            try:
                ui_handler = self.weaver._handlers.get('ui_commands')
                if ui_handler:
                    # Create a mock envelope for the direct call
                    from torchLoom.proto.torchLoom_pb2 import EventEnvelope
                    env = EventEnvelope()
                    env.ui_command.command_type = "reactivate_group"
                    env.ui_command.target_id = replica_id
                    await ui_handler.handle(env)
                    logger.debug(f"Sent UI command directly to weaver: reactivate_group for {replica_id}")
            except Exception as e:
                logger.exception(f"Failed to send direct UI command: {e}")
                # Fallback to NATS if available
                if self.nats_client:
                    await self.send_ui_command("reactivate_group", replica_id)

        # Reset communication status
        await asyncio.sleep(1.2)
        self.status_tracker.set_communication_status("stable")

        logger.info(f"Processed replica group reactivation: {replica_id}")

    async def handle_config_update(self, replica_id: str, config_params: dict):
        """Handle configuration update command."""
        # Update local GPU configs
        replica_gpus = [
            g for g in self.status_tracker.gpus.values() if g.replica_id == replica_id
        ]
        for gpu in replica_gpus:
            gpu.config.update(config_params)

        # Call weaver handler directly (much more efficient than NATS)
        if self.weaver and hasattr(self.weaver, '_handlers'):
            try:
                config_handler = self.weaver._handlers.get('configuration')
                if config_handler:
                    # Create a mock envelope for the direct call
                    from torchLoom.proto.torchLoom_pb2 import EventEnvelope
                    env = EventEnvelope()
                    # Set config_info fields
                    for key, value in config_params.items():
                        env.config_info.config_params[key] = str(value)
                    await config_handler.handle(env)
                    logger.debug(f"Sent config change directly to weaver for {replica_id}: {config_params}")
            except Exception as e:
                logger.exception(f"Failed to send direct config change: {e}")
                # Fallback to NATS if available
                if self.nats_client:
                    await self.send_config_change(replica_id, config_params)

        logger.info(f"Processed config update for {replica_id}: {config_params}")

    async def send_ui_command(
        self, command_type: str, target_id: str, params: Optional[dict] = None
    ):
        """Send UI command via NATS."""
        if not self.nats_client:
            return

        try:
            env = EventEnvelope()
            env.ui_command.command_type = command_type
            env.ui_command.target_id = target_id

            if params:
                for key, value in params.items():
                    env.ui_command.params[key] = str(value)

            js = self.nats_client.jetstream()
            await js.publish(
                torchLoomConstants.subjects.UI_COMMANDS, env.SerializeToString()
            )

            logger.debug(f"Sent UI command: {command_type} for {target_id}")

        except Exception as e:
            logger.exception(f"Failed to send UI command: {e}")

    async def send_config_change(self, replica_id: str, config_params: dict):
        """Send configuration change via NATS."""
        if not self.nats_client:
            return

        try:
            env = EventEnvelope()
            for key, value in config_params.items():
                env.config_info.config_params[key] = str(value)

            js = self.nats_client.jetstream()
            await js.publish(
                torchLoomConstants.subjects.CONFIG_INFO, env.SerializeToString()
            )

            logger.debug(f"Sent config change for {replica_id}: {config_params}")

        except Exception as e:
            logger.exception(f"Failed to send config change: {e}")

    async def broadcast_status_update(self):
        """Broadcast status update to all connected WebSocket clients."""
        if self.manager.active_connections:
            # Send legacy status update format for backward compatibility
            status_data = self.get_ui_status_dict()
            await self.manager.send_json_to_all(
                {"type": "status_update", "data": status_data}
            )
            
            # Send individual protobuf-aligned message types for enhanced reactivity
            await self.broadcast_individual_status_updates()

    async def broadcast_individual_status_updates(self):
        """Broadcast individual status message types aligned with protobuf definitions."""
        if not self.manager.active_connections:
            return
            
        try:
            # Broadcast training status updates
            for replica_id, replica in self.status_tracker.replicas.items():
                training_status_data = {
                    "replica_id": replica.replica_id,
                    "status_type": "training_update", 
                    "current_step": replica.current_step,
                    "epoch": 0,  # Add epoch tracking to StatusTracker if needed
                    "step_progress": replica.step_progress,
                    "epoch_progress": 0.0,  # Add epoch progress tracking if needed
                    "status": replica.status,
                    "metrics": {},  # Add metrics tracking to StatusTracker if needed
                    "training_time": 0.0,  # Add training time tracking if needed
                    "batch_idx": replica.last_active_step,
                    "timestamp": int(time.time())
                }
                
                await self.manager.send_json_to_all({
                    "type": "training_status",
                    "data": training_status_data
                })
            
            # Broadcast GPU status updates
            for gpu_id, gpu in self.status_tracker.gpus.items():
                gpu_status_data = {
                    "gpu_id": gpu.gpu_id,
                    "replica_id": gpu.replica_id,
                    "server_id": gpu.server_id,
                    "status": gpu.status,
                    "utilization": gpu.utilization,
                    "temperature": gpu.temperature,
                    "memory_used": gpu.memory_used,
                    "memory_total": gpu.memory_total,
                    "config": dict(gpu.config),
                    "timestamp": int(time.time())
                }
                
                await self.manager.send_json_to_all({
                    "type": "gpu_status", 
                    "data": gpu_status_data
                })
                
        except Exception as e:
            logger.exception(f"Error broadcasting individual status updates: {e}")

    async def start_status_broadcaster(self):
        """Start periodic status broadcasts to WebSocket clients."""
        logger.info("Starting WebSocket status broadcaster")

        while True:
            try:
                if self.manager.active_connections:
                    await self.broadcast_status_update()
                await asyncio.sleep(TimeoutConstants.STATUS_BROADCAST_IN)  # Broadcast every second

            except Exception as e:
                logger.exception(f"Error in status broadcaster: {e}")
                await asyncio.sleep(2.0)

    async def start_server(self):
        """Start the WebSocket server."""
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level="info"
        )
        server = uvicorn.Server(config)

        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        await server.serve()

    async def run_with_status_broadcaster(self):
        """Run server with status broadcaster in parallel."""
        try:
            # Start both the server and status broadcaster
            await asyncio.gather(self.start_server(), self.start_status_broadcaster())
        except Exception as e:
            logger.exception(f"Error running WebSocket server: {e}")
            raise
