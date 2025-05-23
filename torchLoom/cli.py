import argparse
import asyncio
import cmd
from typing import Optional

import nats
from torchLoom.common.config import Config
from torchLoom.common.constants import torchLoomConstants
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

logger = setup_logger(
    name="torchLoom_monitor_cli", log_file=Config.torchLoom_MONITOR_CLI_LOG_FILE
)


class TorchLoomClient:
    """Programmatic client for sending torchLoom messages."""

    def __init__(self, nats_url: Optional[str] = None):
        self.nats_url = nats_url or torchLoomConstants.DEFAULT_ADDR
        self._nc: Optional[nats.aio.client.Client] = None
        logger.info(f"TorchLoom client initialized with NATS URL: {self.nats_url}")

    async def connect(self):
        """Connect to NATS server."""
        if self._nc is None:
            self._nc = await nats.connect(self.nats_url)
            logger.debug(f"Connected to NATS server at {self.nats_url}")

    async def disconnect(self):
        """Disconnect from NATS server."""
        if self._nc is not None:
            await self._nc.close()
            self._nc = None
            logger.debug("Disconnected from NATS server")

    async def register_device(self, device_uuid: str, replica_id: str):
        """Register a device with a replica."""
        await self.connect()

        if self._nc is None:
            raise RuntimeError("Failed to connect to NATS server")

        envelope = EventEnvelope()
        envelope.register_device.device_uuid = device_uuid
        envelope.register_device.replica_id = replica_id

        await self._nc.publish(
            torchLoomConstants.subjects.MONITOR, envelope.SerializeToString()
        )
        logger.info(
            f"Published device registration for device {device_uuid} with replica {replica_id}"
        )

    async def fail_device(self, device_uuid: str):
        """Simulate a device failure."""
        await self.connect()

        if self._nc is None:
            raise RuntimeError("Failed to connect to NATS server")

        envelope = EventEnvelope()
        envelope.monitored_fail.device_uuid = device_uuid

        await self._nc.publish(
            torchLoomConstants.subjects.MONITOR, envelope.SerializeToString()
        )
        logger.info(f"Published device failure event for device {device_uuid}")

    async def reset_learning_rate(self, learning_rate: str):
        """Reset the learning rate."""
        await self.connect()

        if self._nc is None:
            raise RuntimeError("Failed to connect to NATS server")

        envelope = EventEnvelope()
        envelope.config_info.config_params["learning_rate"] = learning_rate

        await self._nc.publish(
            torchLoomConstants.subjects.CONFIG_INFO, envelope.SerializeToString()
        )
        logger.info(f"Published reset learning rate event with lr = {learning_rate}")

    async def send_config_info(self, config_params: dict):
        """Send configuration information."""
        await self.connect()

        if self._nc is None:
            raise RuntimeError("Failed to connect to NATS server")

        envelope = EventEnvelope()
        for key, value in config_params.items():
            envelope.config_info.config_params[key] = str(value)

        await self._nc.publish(
            torchLoomConstants.subjects.CONFIG_INFO, envelope.SerializeToString()
        )
        logger.info(f"Published config info event with parameters: {config_params}")

    async def set_batch_size(self, batch_size: int):
        """Set the training batch size."""
        await self.send_config_info({"batch_size": str(batch_size)})

    async def pause_training(self):
        """Pause training."""
        await self.send_config_info({"pause_training": "true"})

    async def resume_training(self):
        """Resume training."""
        await self.send_config_info({"pause_training": "false"})

    async def set_training_params(self, **params):
        """Set multiple training parameters at once."""
        await self.send_config_info(params)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class MyShell(cmd.Cmd):
    """Interactive shell for torchLoom CLI."""

    prompt = "torchLoom> "
    intro = "Welcome to torchLoom CLI"

    def __init__(
        self, nats_url: str = None, completekey="tab", stdin=None, stdout=None
    ):
        super().__init__(completekey, stdin, stdout)
        self.client = TorchLoomClient(nats_url)
        logger.info("torchLoom CLI initialized")

    def do_register_device(self, line):
        """Register a device with a replica

        Usage: register_device <device_uuid> <replica_id>
        """
        parts = line.strip().split()
        if len(parts) != 2:
            print("Invalid format. Usage: register_device <device_uuid> <replica_id>")
            return

        device_uuid, replica_id = parts
        asyncio.run(self.client.register_device(device_uuid, replica_id))

    def do_fail_device(self, line):
        """Simulate a device failure

        Usage: fail_device <device_uuid>
        """
        device_uuid = line.strip()
        if not device_uuid:
            print("Device UUID is required")
            return

        asyncio.run(self.client.fail_device(device_uuid))

    def do_reset_lr(self, line):
        """Reset the learning rate

        Usage: reset_lr <learning_rate>
        """
        lr = line.strip()
        if not lr:
            print("Learning rate is required")
            return

        asyncio.run(self.client.reset_learning_rate(lr))

    def do_config_info(self, line):
        """Send configuration information

        Usage: config_info key1=value1 key2=value2 ...
        """
        parts = line.strip().split()
        if not parts:
            print("At least one config parameter is required")
            print("Usage: config_info key1=value1 key2=value2 ...")
            return

        config_params = {}
        for part in parts:
            if "=" not in part:
                print(f"Invalid parameter format: {part}. Use key=value format.")
                return
            key, value = part.split("=", 1)
            config_params[key] = value

        asyncio.run(self.client.send_config_info(config_params))

    def do_set_batch_size(self, line):
        """Set the training batch size

        Usage: set_batch_size <batch_size>
        """
        try:
            batch_size = int(line.strip())
            if batch_size <= 0:
                print("Batch size must be a positive integer")
                return
            asyncio.run(self.client.set_batch_size(batch_size))
            print(f"Set batch size to {batch_size}")
        except ValueError:
            print("Invalid batch size. Must be an integer.")

    def do_pause_training(self, line):
        """Pause training

        Usage: pause_training
        """
        asyncio.run(self.client.pause_training())
        print("Training paused")

    def do_resume_training(self, line):
        """Resume training

        Usage: resume_training
        """
        asyncio.run(self.client.resume_training())
        print("Training resumed")

    def do_set_lr(self, line):
        """Set learning rate (enhanced version)

        Usage: set_lr <learning_rate>
        """
        try:
            lr = float(line.strip())
            if lr <= 0:
                print("Learning rate must be positive")
                return
            asyncio.run(self.client.reset_learning_rate(str(lr)))
            print(f"Set learning rate to {lr}")
        except ValueError:
            print("Invalid learning rate. Must be a number.")

    def do_training_config(self, line):
        """Set multiple training parameters at once

        Usage: training_config lr=0.01 batch_size=64 pause_training=false
        """
        parts = line.strip().split()
        if not parts:
            print("Usage: training_config param1=value1 param2=value2 ...")
            print("Available parameters: learning_rate, batch_size, pause_training")
            return

        config_params = {}
        for part in parts:
            if "=" not in part:
                print(f"Invalid parameter format: {part}. Use key=value format.")
                return
            key, value = part.split("=", 1)

            # Validate common parameters
            if key == "learning_rate":
                try:
                    float(value)
                except ValueError:
                    print(f"Invalid learning rate: {value}")
                    return
            elif key == "batch_size":
                try:
                    int(value)
                except ValueError:
                    print(f"Invalid batch size: {value}")
                    return
            elif key == "pause_training":
                if value.lower() not in ["true", "false"]:
                    print(
                        f"Invalid pause_training value: {value}. Use 'true' or 'false'."
                    )
                    return

            config_params[key] = value

        asyncio.run(self.client.send_config_info(config_params))
        print(f"Updated training configuration: {config_params}")

    def do_exit(self, line):
        """Exit the CLI"""
        logger.info("Exiting torchLoom CLI")
        asyncio.run(self.client.disconnect())
        return True

    def do_quit(self, line):
        """Exit the CLI and close NATS connection."""
        return self.do_exit(line)

    # Legacy commands for backward compatibility
    def do_test(self, line):
        """Legacy command - use fail_device instead"""
        print("Warning: 'test' command is deprecated. Use 'fail_device' instead.")
        self.do_fail_device(line)

    def do_setlr(self, line):
        """Legacy command - use reset_lr instead"""
        print("Warning: 'setlr' command is deprecated. Use 'reset_lr' instead.")
        self.do_reset_lr(line)

    def default(self, line):
        print(f"Unknown command: {line}")

    def emptyline(self):
        """Do nothing on empty line."""
        return None


def run_cli_command(command: str, nats_url: Optional[str] = None) -> bool:
    """Run a single CLI command programmatically.

    Args:
        command: CLI command to run (e.g., "register_device device1 replica1")
        nats_url: NATS server URL (optional)

    Returns:
        True if command executed successfully, False otherwise
    """
    try:
        shell = MyShell(nats_url)
        result = shell.onecmd(command)
        asyncio.run(shell.client.disconnect())
        return result != True  # onecmd returns True for exit commands
    except Exception as e:
        logger.error(f"Failed to execute command '{command}': {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="torchLoom CLI")
    parser.add_argument(
        "--host", type=str, default="localhost", help="NATS server host"
    )
    parser.add_argument("--port", type=int, default=4222, help="NATS server port")
    parser.add_argument(
        "--command", type=str, help="Single command to execute (non-interactive mode)"
    )
    args = parser.parse_args()

    nats_url = f"nats://{args.host}:{args.port}"

    if args.command:
        # Non-interactive mode - run single command
        logger.info(f"Running command: {args.command}")
        success = run_cli_command(args.command, nats_url)
        exit(0 if success else 1)
    else:
        # Interactive mode
        logger.info(f"Starting torchLoom CLI with NATS server at {nats_url}")
        MyShell(nats_url).cmdloop()
