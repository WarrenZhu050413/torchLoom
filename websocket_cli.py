import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Optional

import websockets

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("websocket_cli")


class TorchLoomWebSocketCLI:
    """WebSocket client for interacting with torchLoom weaver."""

    def __init__(
        self,
        ws_url: str = "ws://localhost:8080/ws",
        test_process_id: Optional[str] = None,
    ):
        self.ws_url = ws_url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.running = False
        self.command_queue = asyncio.Queue()
        self.active_process_ids = set()  # Track active process_ids
        self.active_device_uuids = set()  # Track active device_uuids
        self.test_process_id = (
            test_process_id or "test-process"
        )  # Fixed ID for E2E testing
        logger.info(
            f"WebSocket CLI initialized with URL: {ws_url}. Test Process ID: {self.test_process_id}"
        )

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

            # Track active participants for demo commands
            current_process_ids = set()
            current_device_uuids = set()

            # Display device information with all fields and IDs
            devices = status_data.get("devices", [])
            if devices:
                print(f"\n🖥️  DEVICES ({len(devices)}):")
                for device in devices:
                    current_device_uuids.add(device.get("device_uuid", "unknown"))
                    current_process_ids.add(device.get("process_id", "unknown"))
                    device_uuid = device.get("device_uuid") or device.get(
                        "device_uuid", "unknown"
                    )
                    process_id = device.get("process_id", "unknown")
                    server_id = device.get("server_id", "unknown")
                    utilization = device.get("utilization", "N/A")
                    temperature = device.get("temperature", "N/A")
                    memory_used = device.get("memory_used", "N/A")
                    memory_total = device.get("memory_total", "N/A")

                    print(f"  🔸 Device UUID: {device_uuid}")
                    print(f"     Process ID: {process_id}")
                    print(f"     Server: {server_id}")
                    print(f"     Utilization: {utilization}%")
                    print(f"     Temperature: {temperature}°C")
                    print(f"     Memory: {memory_used}/{memory_total} GB")

                    # Display device config if available
                    config = device.get("config", {})
                    if config:
                        print(f"     Config: {config}")
                    print()

            # Display training status with all fields and clear IDs
            training_status = status_data.get("training_status", [])
            if training_status:
                print(f"🚀 TRAINING STATUS ({len(training_status)} replicas):")
                for status in training_status:
                    process_id = status.get("process_id", "unknown")
                    current_process_ids.add(process_id)
                    state = status.get("status", "unknown")
                    current_step = status.get("current_step", 0)
                    epoch = status.get("epoch", 0)
                    max_step = status.get("max_step", 0)
                    max_epoch = status.get("max_epoch", 0)
                    training_time = status.get("training_time", 0.0)

                    print(f"  🔸 Process ID: {process_id}")
                    print(f"     Status: {state}")
                    print(
                        f"     Progress: Step {current_step}/{max_step} | Epoch {epoch}/{max_epoch}"
                    )

                    # Calculate and display progress percentages
                    if max_step > 0:
                        step_progress = (current_step / max_step) * 100
                        print(f"     Step Progress: {step_progress:.1f}%")
                    if max_epoch > 0:
                        epoch_progress = (epoch / max_epoch) * 100
                        print(f"     Epoch Progress: {epoch_progress:.1f}%")

                    if training_time > 0:
                        hours = int(training_time // 3600)
                        minutes = int((training_time % 3600) // 60)
                        seconds = int(training_time % 60)
                        print(
                            f"     Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}"
                        )

                    # Show detailed metrics if available
                    metrics = status.get("metrics", {})
                    if metrics:
                        print(f"     📊 Metrics:")
                        # Primary metrics
                        if "loss" in metrics:
                            print(f"        Loss: {metrics['loss']}")
                        if "accuracy" in metrics:
                            print(f"        Accuracy: {metrics['accuracy']}")
                        if "learning_rate" in metrics:
                            print(f"        Learning Rate: {metrics['learning_rate']}")
                        if "batch_size" in metrics:
                            print(f"        Batch Size: {metrics['batch_size']}")
                        if "gradient_norm" in metrics:
                            print(f"        Gradient Norm: {metrics['gradient_norm']}")
                        if "throughput" in metrics:
                            print(
                                f"        Throughput: {metrics['throughput']} samples/sec"
                            )

                        # Additional metrics
                        additional_metrics = {
                            k: v
                            for k, v in metrics.items()
                            if k
                            not in [
                                "loss",
                                "accuracy",
                                "learning_rate",
                                "batch_size",
                                "gradient_norm",
                                "throughput",
                                "message",
                            ]
                        }
                        if additional_metrics:
                            print(f"        Other: {additional_metrics}")

                    # Show config if available
                    config = status.get("config", {})
                    if config:
                        print(f"     ⚙️  Config: {config}")

                    # Show message if available
                    if status.get("message"):
                        print(f"     💬 Message: {status['message']}")

                    # Show additional status info if available
                    if status.get("status_type"):
                        print(f"     Type: {status['status_type']}")

                    print()

            # Update active participants tracking and display summary
            self.active_process_ids.update(current_process_ids)
            self.active_device_uuids.update(current_device_uuids)

            # Display participants summary
            if self.active_process_ids or self.active_device_uuids:
                print(f"\n📋 ACTIVE PARTICIPANTS:")
                if self.active_process_ids:
                    print(f"   Process IDs: {sorted(list(self.active_process_ids))}")
                if self.active_device_uuids:
                    print(f"   Device UUIDs: {sorted(list(self.active_device_uuids))}")
                print()

        elif msg_type == "pong":
            logger.debug(f"Received pong at {timestamp}")

        elif msg_type == "error":
            print(f"\n[{timestamp}] ❌ ERROR: {data.get('message', 'Unknown error')}")
            if "error" in data:
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

            logger.info("Starting enhanced demo command sequence...")

            # Demo sequence of commands showcasing the comprehensive data
            # Use an actual active process_id if available, otherwise use placeholder
            self.active_process_ids.discard("")
            if self.active_process_ids:

                demo_process_id = next(iter(self.active_process_ids))
                logger.info(
                    f"[DEMO] 🎯 Using active process_id from received status: {demo_process_id} for commands."
                )
                # If a specific test_process_id is set and observed, prefer it for consistency in testing.
                if (
                    self.test_process_id
                    and self.test_process_id in self.active_process_ids
                ):
                    demo_process_id = self.test_process_id
                    logger.info(
                        f"[DEMO] ✅ Prioritizing fixed test_process_id: {self.test_process_id} as it is active."
                    )
                elif self.test_process_id:
                    logger.info(
                        f"[DEMO] ℹ️ Fixed test_process_id {self.test_process_id} is set, but not yet observed in active processes. Will use {demo_process_id} for now."
                    )
            else:
                # Fallback to the fixed test_process_id if no active ones are found yet.
                demo_process_id = self.test_process_id
                logger.warning(
                    f"[DEMO] ⚠️  No active process_ids found. Using pre-defined test_process_id: {demo_process_id} for commands."
                )

            commands = [
                {
                    "type": "ui_command",
                    "data": {
                        "command_type": "pause_training",
                        "process_id": demo_process_id,
                        "params": {},
                    },
                },
            ]

            for i, cmd in enumerate(commands):
                await asyncio.sleep(4)  # Wait between commands for better visualization
                cmd_desc = cmd.get("type", "unknown")
                if cmd_desc == "ui_command":
                    cmd_desc = f"UI: {cmd['data'].get('command_type', 'unknown')}"

                print(f"\n[DEMO] 🚀 Sending command {i+1}/{len(commands)}: {cmd_desc}")
                if "data" in cmd and "params" in cmd["data"]:
                    print(f"       Parameters: {cmd['data']['params']}")
                await self.command_queue.put(cmd)

            logger.info("Demo command sequence completed - continuing to monitor...")
            await asyncio.sleep(20)  # Wait longer before next sequence

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
                asyncio.create_task(self.demo_commands(), name="demo"),
            ]

            # Wait for any task to complete (likely receive_messages on disconnect)
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

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
    print("3. Send a sequence of demo commands targeting a specific process_id")
    print("\nPress Ctrl+C to exit\n")

    # For testing, we can pass a specific process_id to the CLI
    # This should match the process_id used when spawning the threadlet
    test_pid = (
        "test-process"  # Ensure this matches the one in spawn_threadlet.py for the test
    )
    cli = TorchLoomWebSocketCLI(test_process_id=test_pid)
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
