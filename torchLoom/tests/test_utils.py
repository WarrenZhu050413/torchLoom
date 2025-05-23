"""
Test utilities for torchLoom tests.

This module provides utilities for setting up real NATS servers for integration testing.
"""

import asyncio
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import nats


class NatsTestServer:
    """Manages a NATS server instance for testing."""

    def __init__(self, port: int = 4223, jetstream: bool = True):
        """Initialize NATS test server.

        Args:
            port: Port to run the server on (default 4223 to avoid conflicts)
            jetstream: Whether to enable JetStream
        """
        self.port = port
        self.jetstream = jetstream
        self.process: Optional[subprocess.Popen] = None
        self.nats_server_path = (
            Path(__file__).parent.parent.parent / "nats" / "nats-server"
        )

    async def start(self) -> str:
        """Start the NATS server and return the connection URL."""
        if self.process is not None:
            raise RuntimeError("NATS server is already running")

        # Build command
        cmd = [str(self.nats_server_path), "-p", str(self.port)]
        if self.jetstream:
            cmd.extend(["-js"])

        # Start the server
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Wait for server to start
        connection_url = f"nats://localhost:{self.port}"
        max_attempts = 20
        for attempt in range(max_attempts):
            try:
                nc = await nats.connect(connection_url, connect_timeout=1)
                await nc.close()
                print(f"NATS server started on {connection_url}")
                return connection_url
            except Exception:
                if attempt == max_attempts - 1:
                    self.stop()
                    raise RuntimeError(
                        f"Failed to start NATS server after {max_attempts} attempts"
                    )
                await asyncio.sleep(0.1)

        # Should never reach here, but adding for type safety
        raise RuntimeError("Failed to start NATS server")

    def stop(self):
        """Stop the NATS server."""
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
            print("NATS server stopped")

    async def __aenter__(self):
        """Async context manager entry."""
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.stop()


async def wait_for_messages(subscription, count: int, timeout: float = 5.0):
    """Wait for a specific number of messages on a subscription.

    Args:
        subscription: NATS subscription object
        count: Number of messages to wait for
        timeout: Maximum time to wait in seconds

    Returns:
        List of received messages
    """
    messages = []
    end_time = time.time() + timeout

    while len(messages) < count and time.time() < end_time:
        try:
            msg = await subscription.next_msg(timeout=0.1)
            messages.append(msg)
        except asyncio.TimeoutError:
            continue

    return messages
