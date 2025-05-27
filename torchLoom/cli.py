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

    async def _publish_ui_command(
        self,
        command_type: str,
        target_id: Optional[str] = None,
        params: Optional[dict] = None,
    ):
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
            torchLoomConstants.subjects.UI_COMMANDS, envelope.SerializeToString()
        )
        logger.info(
            f"Published UI command: {command_type}, Target: {target_id}, Params: {params}"
        )


class MyShell(cmd.Cmd):
    """Interactive shell for torchLoom CLI."""

    prompt = "torchLoom> "
    intro = "Welcome to torchLoom CLI. Type help or ? to list commands.\nNATS URL: Not connected. Connects on first command."

    def __init__(
        self,
        nats_url_override: Optional[str] = None,
        completekey="tab",
        stdin=None,
        stdout=None,
    ):
        super().__init__(completekey, stdin, stdout)
        effective_nats_url = nats_url_override or torchLoomConstants.DEFAULT_ADDR
        self.client = TorchLoomClient(effective_nats_url)
        self.intro = f"Welcome to torchLoom CLI. NATS URL: {self.client.nats_url}"
        logger.info(f"torchLoom CLI initialized with NATS URL: {self.client.nats_url}")
