"""
WebSocket server for torchLoom UI integration.

This module provides real-time bidirectional communication between the
Weaver backend and the Vue.js frontend using WebSockets and REST APIs.
"""

import asyncio
import json
import time
from typing import List, Dict, Set, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from torchLoom.log.logger import setup_logger
from torchLoom.constants import torchLoomConstants
from torchLoom.proto.torchLoom_pb2 import UICommand, EventEnvelope

logger = setup_logger(name="websocket_server")


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
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
    
    def __init__(self, status_tracker, nats_client=None, host="0.0.0.0", port=8080):
        self.app = FastAPI(title="torchLoom UI API", version="1.0.0")
        self.status_tracker = status_tracker
        self.nats_client = nats_client
        self.host = host
        self.port = port
        self.manager = ConnectionManager()
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
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
                await websocket.send_text(json.dumps({
                    "type": "status_update",
                    "data": initial_status
                }))
                
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
            return {"status": "success", "message": f"GPU {gpu_id} deactivation initiated"}
        
        @self.app.post("/api/commands/reactivate-group")
        async def reactivate_group(request: dict):
            """Reactivate a replica group."""
            replica_id = request.get("replica_id")
            if not replica_id:
                raise HTTPException(status_code=400, detail="replica_id required")
            
            await self.handle_reactivate_group(replica_id)
            return {"status": "success", "message": f"Replica group {replica_id} reactivation initiated"}
        
        @self.app.post("/api/commands/update-config")
        async def update_config(request: dict):
            """Update training configuration."""
            replica_id = request.get("replica_id")
            config_params = request.get("config_params", {})
            
            if not replica_id:
                raise HTTPException(status_code=400, detail="replica_id required")
            
            await self.handle_config_update(replica_id, config_params)
            return {"status": "success", "message": f"Configuration updated for {replica_id}"}
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "connections": len(self.manager.active_connections),
                "gpus": len(self.status_tracker.gpus),
                "replicas": len(self.status_tracker.replicas)
            }
    
    def get_ui_status_dict(self) -> dict:
        """Convert status tracker data to UI-friendly format."""
        # Organize GPUs by replica groups and servers
        replica_groups = {}
        
        for gpu_id, gpu in self.status_tracker.gpus.items():
            replica_id = gpu.replica_id
            server_id = gpu.server_id
            
            # Extract group ID from replica_id
            group_id = replica_id.split('_')[0] if '_' in replica_id else replica_id
            
            if group_id not in replica_groups:
                replica_groups[group_id] = {
                    "id": group_id,
                    "gpus": {},
                    "status": "training",
                    "stepProgress": 0,
                    "fixedStep": None,
                    "lastActiveStep": None
                }
            
            # Add GPU data
            replica_groups[group_id]["gpus"][gpu_id] = {
                "id": gpu_id,
                "server": server_id,
                "status": gpu.status,
                "utilization": gpu.utilization,
                "temperature": gpu.temperature,
                "batch": gpu.config.get("batch_size", "32"),
                "lr": gpu.config.get("learning_rate", "0.001"),
                "opt": gpu.config.get("optimizer_type", "Adam")
            }
        
        # Update replica group status from replica tracker
        for replica_id, replica in self.status_tracker.replicas.items():
            group_id = replica_id.split('_')[0] if '_' in replica_id else replica_id
            
            if group_id in replica_groups:
                replica_groups[group_id]["status"] = replica.status
                replica_groups[group_id]["stepProgress"] = replica.step_progress
                replica_groups[group_id]["lastActiveStep"] = replica.last_active_step
                if replica.fixed_step is not None:
                    replica_groups[group_id]["fixedStep"] = replica.fixed_step
        
        return {
            "step": self.status_tracker.global_step,
            "replicaGroups": replica_groups,
            "communicationStatus": self.status_tracker.communication_status,
            "timestamp": time.time()
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
                await self.handle_config_update(data.get("replica_id"), data.get("config_params", {}))
            elif command_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
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
        
        # Send UI command via NATS if available
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
        
        # Send UI command via NATS if available
        if self.nats_client:
            await self.send_ui_command("reactivate_group", replica_id)
        
        # Reset communication status
        await asyncio.sleep(1.2)
        self.status_tracker.set_communication_status("stable")
        
        logger.info(f"Processed replica group reactivation: {replica_id}")
    
    async def handle_config_update(self, replica_id: str, config_params: dict):
        """Handle configuration update command."""
        # Update local GPU configs
        replica_gpus = [g for g in self.status_tracker.gpus.values() if g.replica_id == replica_id]
        for gpu in replica_gpus:
            gpu.config.update(config_params)
        
        # Send config change via NATS if available
        if self.nats_client:
            await self.send_config_change(replica_id, config_params)
        
        logger.info(f"Processed config update for {replica_id}: {config_params}")
    
    async def send_ui_command(self, command_type: str, target_id: str, params: Optional[dict] = None):
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
            await js.publish(torchLoomConstants.subjects.UI_COMMAND, env.SerializeToString())
            
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
            await js.publish(torchLoomConstants.subjects.CONFIG_INFO, env.SerializeToString())
            
            logger.debug(f"Sent config change for {replica_id}: {config_params}")
            
        except Exception as e:
            logger.exception(f"Failed to send config change: {e}")
    
    async def broadcast_status_update(self):
        """Broadcast status update to all connected WebSocket clients."""
        if self.manager.active_connections:
            status_data = self.get_ui_status_dict()
            await self.manager.send_json_to_all({
                "type": "status_update",
                "data": status_data
            })
    
    async def start_status_broadcaster(self):
        """Start periodic status broadcasts to WebSocket clients."""
        logger.info("Starting WebSocket status broadcaster")
        
        while True:
            try:
                if self.manager.active_connections:
                    await self.broadcast_status_update()
                await asyncio.sleep(1.0)  # Broadcast every second
                
            except Exception as e:
                logger.exception(f"Error in status broadcaster: {e}")
                await asyncio.sleep(2.0)
    
    async def start_server(self):
        """Start the WebSocket server."""
        config = uvicorn.Config(
            self.app, 
            host=self.host, 
            port=self.port, 
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        await server.serve()
    
    async def run_with_status_broadcaster(self):
        """Run server with status broadcaster in parallel."""
        try:
            # Start both the server and status broadcaster
            await asyncio.gather(
                self.start_server(),
                self.start_status_broadcaster()
            )
        except Exception as e:
            logger.exception(f"Error running WebSocket server: {e}")
            raise 