import asyncio
import json
import time
from typing import Optional
import websockets
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("websocket_cli")


class TorchLoomWebSocketCLI:
    """WebSocket client for interacting with torchLoom weaver."""
    
    def __init__(self, ws_url: str = "ws://localhost:8080/ws"):
        self.ws_url = ws_url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.running = False
        self.command_queue = asyncio.Queue()
        logger.info(f"WebSocket CLI initialized with URL: {ws_url}")
        
    async def connect(self):
        """Connect to the WebSocket server."""
        try:
            self.websocket = await websockets.connect(self.ws_url)
            self.running = True
            logger.info(f"Connected to WebSocket server at {self.ws_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket server: {e}")
            return False
            
    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from WebSocket server")
            
    async def send_command(self, command_data: dict):
        """Send a command to the WebSocket server."""
        if self.websocket:
            try:
                await self.websocket.send(json.dumps(command_data))
                logger.info(f"Sent command: {command_data.get('type', 'unknown')}")
            except Exception as e:
                logger.error(f"Failed to send command: {e}")
                
    async def receive_messages(self):
        """Continuously receive and process messages from the WebSocket server."""
        while self.running:
            try:
                if self.websocket:
                    message = await self.websocket.recv()
                    data = json.loads(message)
                    await self.process_message(data)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.running = False
                break
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON message: {e}")
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                
    async def process_message(self, data: dict):
        """Process received messages and display them."""
        msg_type = data.get("type", "unknown")
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if msg_type == "status_update":
            status_data = data.get("data", {})
            print(f"\n[{timestamp}] === STATUS UPDATE ===")
            
            # Display device information
            devices = status_data.get("devices", [])
            if devices:
                print(f"Devices ({len(devices)}):")
                for device in devices:
                    device_uuid = device.get('device_uuid') or device.get('device_id', 'unknown')
                    # Status is removed from deviceStatus, log other available info
                    replica_id = device.get('replica_id', 'unknown')
                    utilization = device.get('utilization', 'N/A')
                    temperature = device.get('temperature', 'N/A')
                    print(f"  - {device_uuid} | Replica: {replica_id} | Util: {utilization}% | Temp: {temperature}°C")
            
            # Display training status
            training_status = status_data.get("training_status", [])
            if training_status:
                print(f"\nTraining Status ({len(training_status)} replicas):")
                for status in training_status:
                    replica_id = status.get("replica_id", "unknown")
                    state = status.get("status", "unknown")
                    step = status.get("current_step", 0)
                    epoch = status.get("epoch", 0)
                    print(f"  - Replica: {replica_id}")
                    print(f"    Status: {state} | Step: {step} | Epoch: {epoch}")
                    
                    # Show metrics if available
                    metrics = status.get("metrics", {})
                    if metrics:
                        loss = metrics.get("loss", "N/A")
                        accuracy = metrics.get("accuracy", "N/A")
                        print(f"    Metrics: Loss={loss}, Accuracy={accuracy}")
                    
                    # Show config if available
                    config = status.get("config", {})
                    if config:
                        print(f"    Config: {config}")
                    
                    # Show additional status info if available
                    if status.get("training_time"):
                        print(f"    Training Time: {status['training_time']:.2f}s")
                    if status.get("status_type"):
                        print(f"    Type: {status['status_type']}")
                        
        elif msg_type == "pong":
            logger.debug(f"Received pong at {timestamp}")
            
        elif msg_type == "error":
            print(f"\n[{timestamp}] ERROR: {data.get('message', 'Unknown error')}")
            if 'error' in data:
                print(f"  Details: {data['error']}")
                
        else:
            print(f"\n[{timestamp}] Received {msg_type}: {json.dumps(data, indent=2)}")
            
    async def send_ping(self):
        """Send periodic ping messages to keep connection alive."""
        while self.running:
            await self.send_command({"type": "ping"})
            await asyncio.sleep(30)  # Ping every 30 seconds
            
    async def process_command_queue(self):
        """Process commands from the queue."""
        while self.running:
            try:
                # Wait for command with timeout
                command = await asyncio.wait_for(self.command_queue.get(), timeout=1.0)
                await self.send_command(command)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing command queue: {e}")
                
    async def demo_commands(self):
        """Send a sequence of demo commands, then continue running."""
        
        await asyncio.sleep(2)  # Wait for initial connection
        while self.running:
            
            logger.info("Starting demo command sequence...")
            
            # Demo sequence of commands
            commands = [
                {
                    "type": "ui_command",
                    "data": {
                        "command_type": "update_config",
                        "target_id": "demo-replica-1",
                        "params": {
                            "learning_rate": "0.001",
                            "batch_size": "32"
                        }
                    }
                },
                {
                    "type": "pause_training",
                    "replica_id": "demo-replica-1"
                },
                {
                    "type": "resume_training", 
                    "replica_id": "demo-replica-1"
                },
                {
                    "type": "ui_command",
                    "data": {
                        "command_type": "deactivate_device",
                        "target_id": "device-123",
                        "params": {}
                    }
                }
            ]
            
            for i, cmd in enumerate(commands):
                await asyncio.sleep(3)  # Wait between commands
                print(f"\n[DEMO] Sending command {i+1}/{len(commands)}: {cmd.get('type', cmd.get('data', {}).get('command_type', 'unknown'))}")
                await self.command_queue.put(cmd)
                
            logger.info("Demo command sequence completed - continuing to run...")
            await asyncio.sleep(5)
        
    async def run(self):
        """Main run loop."""
        if not await self.connect():
            return
            
        try:
            # Create tasks
            tasks = [
                asyncio.create_task(self.receive_messages(), name="receive"),
                asyncio.create_task(self.send_ping(), name="ping"),
                asyncio.create_task(self.process_command_queue(), name="command_queue"),
                asyncio.create_task(self.demo_commands(), name="demo")
            ]
            
            # Wait for any task to complete (likely receive_messages on disconnect)
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await self.disconnect()
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.exception(f"Error in main run loop: {e}")
        finally:
            await self.disconnect()


async def main():
    """Main entry point."""
    print("=== torchLoom WebSocket CLI ===")
    print("Connecting to weaver WebSocket server...")
    print("This demo will:")
    print("1. Connect to the WebSocket server")
    print("2. Receive and display status updates")
    print("3. Send a sequence of demo commands")
    print("\nPress Ctrl+C to exit\n")
    
    cli = TorchLoomWebSocketCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main()) 