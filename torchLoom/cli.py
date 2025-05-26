import argparse
import asyncio
import cmd
from typing import Optional

import nats
from torchLoom.common.config import Config
from torchLoom.common.constants import torchLoomConstants
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope, UICommand

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
        if self._nc is None or self._nc.is_closed:
            self._nc = await nats.connect(self.nats_url)
            logger.debug(f"Connected to NATS server at {self.nats_url}")

    async def disconnect(self):
        """Disconnect from NATS server."""
        if self._nc is not None and not self._nc.is_closed:
            await self._nc.close()
            self._nc = None
            logger.debug("Disconnected from NATS server")

    async def _publish_event(self, subject: str, envelope: EventEnvelope):
        """Helper to publish an EventEnvelope."""
        await self.connect()
        if self._nc is None:
            raise RuntimeError("Failed to connect to NATS server")
        await self._nc.publish(subject, envelope.SerializeToString())

    async def _publish_ui_command(self, command_type: str, target_id: Optional[str] = None, params: Optional[dict] = None):
        """Helper to publish a UICommand via EventEnvelope."""
        await self.connect()
        if self._nc is None:
            raise RuntimeError("Failed to connect to NATS server")

        envelope = EventEnvelope()
        envelope.ui_command.command_type = command_type
        if target_id:
            envelope.ui_command.target_id = target_id
        if params:
            for key, value in params.items():
                envelope.ui_command.params[key] = str(value)

        await self._nc.publish(
            torchLoomConstants.subjects.UI_COMMANDS,
            envelope.SerializeToString()
        )
        logger.info(f"Published UI command: {command_type}, Target: {target_id}, Params: {params}")

    async def register_device(self, device_uuid: str, replica_id: str):
        """Register a device with a replica."""
        envelope = EventEnvelope()
        envelope.register_device.device_uuid = device_uuid
        envelope.register_device.replica_id = replica_id
        await self._publish_event(
            torchLoomConstants.subjects.THREADLET_EVENTS, envelope
        )
        logger.info(
            f"Published device registration for device {device_uuid} with replica {replica_id}"
        )

    async def fail_device(self, device_uuid: str):
        """Tell the weaver to mark a device as failed."""
        await self._publish_ui_command(command_type="deactivate_device", target_id=device_uuid)
        logger.info(f"Published UI command to fail device {device_uuid}")

    async def send_config_info(self, config_params: dict):
        """Send configuration information."""
        envelope = EventEnvelope()
        for key, value in config_params.items():
            envelope.config_info.config_params[key] = str(value)

        await self._publish_event(
            torchLoomConstants.subjects.CONFIG_INFO, envelope
        )
        logger.info(f"Published config info event with parameters: {config_params}")

    async def reset_learning_rate(self, learning_rate: str):
        """Reset the learning rate via config update."""
        await self.send_config_info({"learning_rate": learning_rate})

    async def set_batch_size(self, batch_size: int):
        """Set the training batch size via config update."""
        await self.send_config_info({"batch_size": str(batch_size)})

    async def pause_training(self, target_replica_id: Optional[str] = None):
        """Pause training for a specific replica or all if None."""
        await self._publish_ui_command(command_type="pause_training", target_id=target_replica_id)
        logger.info(f"Published UI command to pause training for target: {target_replica_id or 'all'}")

    async def resume_training(self, target_replica_id: Optional[str] = None):
        """Resume training for a specific replica or all if None."""
        await self._publish_ui_command(command_type="resume_training", target_id=target_replica_id)
        logger.info(f"Published UI command to resume training for target: {target_replica_id or 'all'}")

    async def set_training_params(self, **params):
        """Set multiple training parameters at once via config update."""
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
    intro = "Welcome to torchLoom CLI. Type help or ? to list commands.\nNATS URL: Not connected. Connects on first command."

    def __init__(
        self, nats_url_override: Optional[str] = None, completekey="tab", stdin=None, stdout=None
    ):
        super().__init__(completekey, stdin, stdout)
        effective_nats_url = nats_url_override or torchLoomConstants.DEFAULT_ADDR
        self.client = TorchLoomClient(effective_nats_url)
        self.intro = f"Welcome to torchLoom CLI. NATS URL: {self.client.nats_url}"
        logger.info(f"torchLoom CLI initialized with NATS URL: {self.client.nats_url}")

    def do_register_device(self, line):
        """Register a device with a replica group.

        Usage: register_device <device_uuid> <replica_id>
        """
        parts = line.strip().split()
        if len(parts) != 2:
            print("Invalid format. Usage: register_device <device_uuid> <replica_id>")
            return
        device_uuid, replica_id = parts
        asyncio.run(self.client.register_device(device_uuid, replica_id))
        print(f"Sent registration for device {device_uuid} to replica {replica_id}")

    def do_fail_device(self, line):
        """Mark a device as failed (sends UICommand deactivate_device).

        Usage: fail_device <device_uuid>
        """
        device_uuid = line.strip()
        if not device_uuid:
            print("Device UUID is required. Usage: fail_device <device_uuid>")
            return
        asyncio.run(self.client.fail_device(device_uuid))
        print(f"Sent command to mark device {device_uuid} as failed.")

    def do_reset_lr(self, line):
        """Set/Reset the learning rate via config update.

        Usage: reset_lr <learning_rate>
        """
        lr_str = line.strip()
        try:
            lr_val = float(lr_str)
            if lr_val <= 0:
                print("Learning rate must be a positive number.")
                return
            asyncio.run(self.client.reset_learning_rate(lr_str))
            print(f"Sent config update for learning_rate: {lr_str}")
        except ValueError:
            print("Invalid learning rate. Must be a number. Usage: reset_lr <value>")

    def do_config_info(self, line):
        """Send arbitrary configuration key-value pairs.

        Usage: config_info key1=value1 key2=value2 ...
        """
        parts = line.strip().split()
        if not parts:
            print("At least one config parameter is required.")
            print("Usage: config_info key1=value1 key2=value2 ...")
            return
        config_params = {}
        try:
            for part in parts:
                key, value = part.split("=", 1)
                config_params[key] = value
            asyncio.run(self.client.send_config_info(config_params))
            print(f"Sent config_info: {config_params}")
        except ValueError:
            print("Invalid format. Use key=value pairs separated by spaces.")

    def do_set_batch_size(self, line):
        """Set the training batch size via config update.

        Usage: set_batch_size <batch_size>
        """
        try:
            batch_size = int(line.strip())
            if batch_size <= 0:
                print("Batch size must be a positive integer.")
                return
            asyncio.run(self.client.set_batch_size(batch_size))
            print(f"Sent config update for batch_size: {batch_size}")
        except ValueError:
            print("Invalid batch size. Must be an integer. Usage: set_batch_size <value>")

    def do_pause_training(self, line):
        """Pause training (sends UICommand pause_training).

        Usage: pause_training [replica_id] (if no replica_id, attempts to pause all)
        """
        target_replica_id = line.strip() or None
        asyncio.run(self.client.pause_training(target_replica_id))
        print(f"Sent command to pause training for target: {target_replica_id or 'all'}")

    def do_resume_training(self, line):
        """Resume training (sends UICommand resume_training).

        Usage: resume_training [replica_id] (if no replica_id, attempts to resume all)
        """
        target_replica_id = line.strip() or None
        asyncio.run(self.client.resume_training(target_replica_id))
        print(f"Sent command to resume training for target: {target_replica_id or 'all'}")

    def do_training_config(self, line):
        """Set multiple training parameters via config update (e.g., learning_rate, batch_size).

        Usage: training_config lr=0.01 batch_size=64
        """
        parts = line.strip().split()
        if not parts:
            print("Usage: training_config param1=value1 param2=value2 ...")
            return
        config_params = {}
        try:
            for part in parts:
                key, value = part.split("=", 1)
                # Basic validation for known params can be enhanced here if needed
                if key == "learning_rate":
                    float(value)
                elif key == "batch_size":
                    int(value)
                config_params[key] = value
            asyncio.run(self.client.set_training_params(**config_params))
            print(f"Sent training_config update: {config_params}")
        except ValueError:
            print("Invalid format or value. Use key=value. Check types for lr (float) and batch_size (int).")

    def do_exit(self, line):
        """Exit the CLI and close NATS connection."""
        logger.info("Exiting torchLoom CLI via 'exit' command.")
        return True

    def do_quit(self, line):
        """Exit the CLI and close NATS connection."""
        return self.do_exit(line)

    def postloop(self):
        """Hook method executed once when cmdloop() is about to return."""
        print("Shutting down NATS connection...")
        asyncio.run(self.client.disconnect())
        logger.info("torchLoom CLI event loop finished.")

    # Legacy commands - keep for now, but point to new ones
    def do_test(self, line):
        """Legacy: Simulates device failure. Use `fail_device` <device_uuid> instead."""
        print("Warning: 'test' command is deprecated. Use 'fail_device <device_uuid>' instead.")
        self.do_fail_device(line)

    def do_setlr(self, line):
        """Legacy: Sets learning rate. Use `reset_lr` <learning_rate> instead."""
        print("Warning: 'setlr' command is deprecated. Use 'reset_lr <learning_rate>' instead.")
        self.do_reset_lr(line)

    def default(self, line):
        if line == 'EOF':
            print("Exiting...")
            return self.do_exit(line)
        print(f"Unknown command: {line}. Type help or ? for a list of commands.")

    def emptyline(self):
        """Do nothing on empty line."""
        pass


def run_cli_command(command: str, nats_url: Optional[str] = None) -> bool:
    """Run a single CLI command programmatically.

    Args:
        command: CLI command to run (e.g., "register_device device1 replica1")
        nats_url: NATS server URL (optional, defaults to TorchLoomConstants.DEFAULT_ADDR)

    Returns:
        True if command processing indicates success (not an exit command), False otherwise or on error.
    """
    cli_nats_url = nats_url or torchLoomConstants.DEFAULT_ADDR
    shell = MyShell(nats_url_override=cli_nats_url)
    try:
        result = shell.onecmd(command)
        return not result
    except Exception as e:
        logger.error(f"Exception executing command '{command}': {e}")
        return False
    finally:
        asyncio.run(shell.client.disconnect())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="torchLoom CLI - Interactive NATS message publisher")
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help=f"NATS server host. Default is from TorchLoom config if not set."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"NATS server port. Default is from TorchLoom config if not set."
    )
    parser.add_argument(
        "--nats_url",
        type=str,
        default=None,
        help=f"Full NATS server URL (e.g., nats://localhost:4222). Overrides host/port if set. Default: {torchLoomConstants.DEFAULT_ADDR}"
    )
    parser.add_argument(
        "--command",
        type=str,
        help="Single command to execute (non-interactive mode). E.g., \"register_device dev1 rep1\""
    )
    args = parser.parse_args()

    cli_nats_url = torchLoomConstants.DEFAULT_ADDR
    if args.nats_url:
        cli_nats_url = args.nats_url
    elif args.host and args.port:
        cli_nats_url = f"nats://{args.host}:{args.port}"
    elif args.host:
        default_port = 4222
        try:
            default_port = int(torchLoomConstants.DEFAULT_ADDR.split(':')[-1])
        except (ValueError, IndexError):
            pass
        cli_nats_url = f"nats://{args.host}:{default_port}"
    elif args.port:
        default_host = "localhost"
        try:
            default_host = torchLoomConstants.DEFAULT_ADDR.split('//')[-1].split(':')[0]
        except IndexError:
            pass
        cli_nats_url = f"nats://{default_host}:{args.port}"

    if args.command:
        logger.info(f"Running single command: '{args.command}' on NATS: {cli_nats_url}")
        success = run_cli_command(args.command, cli_nats_url)
        exit(0 if success else 1)
    else:
        logger.info(f"Starting torchLoom CLI (interactive mode) with NATS server at {cli_nats_url}")
        try:
            MyShell(nats_url_override=cli_nats_url).cmdloop()
        except KeyboardInterrupt:
            print("\nExiting CLI due to KeyboardInterrupt...")
            logger.info("CLI exited via KeyboardInterrupt.")
        finally:
            pass
