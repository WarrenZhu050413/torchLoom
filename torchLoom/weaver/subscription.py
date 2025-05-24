"""
Subscription management for the torchLoom Weaver.

This module handles NATS and JetStream subscription management,
following the single responsibility principle from AGENTS.md.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Tuple

import nats.errors
from nats.aio.client import Client
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

import nats
from torchLoom.common.config import Config
from torchLoom.common.constants import JS, NC, torchLoomConstants
from torchLoom.common.utils import cancel_subscriptions
from torchLoom.log.log_utils import log_and_raise_exception
from torchLoom.log.logger import setup_logger

logger = setup_logger(name="weaver_subscription")


class StreamManager:
    """Manages JetStream stream creation and configuration."""

    def __init__(self, js: JetStreamContext):
        self._js = js
        self._created_streams = set()  # Track created streams

    async def maybe_create_stream(self, stream: str, subjects: List[str]) -> None:
        """Create a stream if it doesn't exist, or update it if it has different subjects."""
        logger.info(
            f"maybe_create_stream called for stream '{stream}' with subjects: {subjects}"
        )

        # Skip if we already created this stream
        if stream in self._created_streams:
            logger.debug(f"Stream {stream} already created, skipping...")
            return

        try:
            logger.info(
                f"Attempting to create stream '{stream}' with subjects: {subjects}"
            )
            await self._js.add_stream(name=stream, subjects=subjects)
            logger.info(f"Created stream: {stream}")
            self._created_streams.add(stream)
        except Exception as e:
            # Handle various stream existence scenarios
            error_msg = str(e).lower()
            logger.warning(f"Error creating stream {stream}: {e}")

            if (
                "already exists" in error_msg
                or "already in use" in error_msg
                or "err_code=10058" in error_msg
            ):
                logger.info(
                    f"Stream {stream} already exists, attempting to update subjects..."
                )

                try:
                    # Try to get the current stream info to see if it needs updating
                    stream_info = await self._js.stream_info(stream)
                    current_subjects = stream_info.config.subjects
                    logger.info(f"Current stream subjects: {current_subjects}")
                    logger.info(f"Desired stream subjects: {subjects}")

                    # Check if subjects match (handle None case)
                    if current_subjects is None:
                        current_subjects = []

                    if set(current_subjects) != set(subjects):
                        logger.info(
                            f"Updating stream {stream} with new subjects: {subjects}"
                        )
                        # Update the stream with new subjects
                        await self._js.update_stream(name=stream, subjects=subjects)
                        logger.info(
                            f"Successfully updated stream {stream} with subjects: {subjects}"
                        )
                    else:
                        logger.info(f"Stream {stream} already has correct subjects")

                    self._created_streams.add(stream)

                except Exception as update_error:
                    logger.exception(
                        f"Failed to update stream {stream}: {update_error}"
                    )
                    # Even if update fails, mark as created to avoid loops
                    self._created_streams.add(stream)
            else:
                logger.exception(f"Error creating stream {stream}: {e}")
                raise


class SubscriptionManager:
    """Manages NATS and JetStream subscriptions."""

    def __init__(self, nc: Client, js: JetStreamContext, stop_event: asyncio.Event):
        self._nc = nc
        self._js = js
        self._stop_event = stop_event
        self._subscriptions: Dict[str, Tuple[Any, asyncio.Task | None]] = {}
        self._stream_manager = StreamManager(js)

        # Configuration from Config
        self._nc_timeout = Config.NC_TIMEOUT or 1
        self._exception_sleep = Config.EXCEPTION_RETRY_TIME or 1

    async def subscribe_js(
        self,
        stream: str,
        subject: str,
        consumer: str,
        message_handler: Callable[[Msg], Awaitable[None]],
    ) -> None:
        """Subscribe to a JetStream subject (assumes stream already exists)."""
        self._validate_js_connection()

        # Don't create streams here - the main weaver should handle stream creation
        # with the proper subject configuration to avoid conflicts
        logger.info(
            f"Subscribing to {subject} on existing stream {stream} with consumer {consumer}"
        )

        psub = await self._js.pull_subscribe(subject, durable=consumer, stream=stream)

        logger.info(
            f"Pull subscription created for {subject} on stream {stream} with consumer {consumer}"
        )

        async def listen_to_js_subscription():
            logger.info(f"Started listening on {subject}")
            while True:
                try:
                    # fetch(1) will wait up to 1 second if no messages are available
                    msgs = await psub.fetch(1, timeout=1)
                    logger.debug(f"Received {len(msgs)} messages on {subject}")
                except TimeoutError:
                    # no new messages this cycle; loop and try again
                    continue
                except Exception as e:
                    logger.exception(f"Error fetching messages from {subject}: {e}")
                    continue

                try:
                    await asyncio.gather(*[message_handler(msg) for msg in msgs])

                    await asyncio.gather(*[msg.ack() for msg in msgs])
                except Exception as e:
                    logger.exception(f"Error processing messages from {subject}: {e}")

        task = asyncio.create_task(listen_to_js_subscription())
        self._subscriptions[subject] = (psub, task)

    async def subscribe_nc(
        self, subject: str, message_handler: Callable[[Msg], Awaitable[None]]
    ) -> None:
        """Subscribe to a regular NATS subject using callback-based approach."""
        self._validate_nc_connection()

        logger.info(f"Subscribing to {subject}")

        # Create a callback wrapper that handles exceptions and stop event
        async def callback_wrapper(msg: Msg) -> None:
            # Check if we should stop processing messages
            if self._stop_event.is_set():
                logger.debug(f"Stop event set, skipping message on {subject}")
                return

            try:
                logger.debug(f"Processing message on {subject}: {msg.data[:50]}...")
                await message_handler(msg)
                logger.debug(f"Successfully processed message on {subject}")
            except Exception as e:
                logger.exception(f"Error processing message from {subject}: {e}")
                # Sleep briefly to avoid tight error loops
                await asyncio.sleep(self._exception_sleep)

        # Use NATS callback-based subscription
        sub = await self._nc.subscribe(subject, cb=callback_wrapper)
        logger.info(f"Subscribed to {subject} with callback")

        # Store subscription without a task since NATS handles message delivery
        self._subscriptions[subject] = (sub, None)

    async def stop_all_subscriptions(self) -> None:
        """Stop all active subscriptions."""
        logger.info("Stopping all subscriptions")
        await cancel_subscriptions(self._subscriptions)
        self._subscriptions.clear()
        logger.info("All subscriptions cleared")

    def _validate_nc_connection(self) -> None:
        """Validate that NATS connection is available."""
        if self._nc is None:
            log_and_raise_exception(
                logger, "NATS connection is not initialized, call initialize() first"
            )

    def _validate_js_connection(self) -> None:
        """Validate that JetStream connection is available."""
        if self._js is None:
            log_and_raise_exception(
                logger, "JetStream is not initialized, call initialize() first"
            )


class ConnectionManager:
    """Manages NATS connection lifecycle."""

    def __init__(self, torchLoom_addr: str = torchLoomConstants.DEFAULT_ADDR):
        self._torchLoom_addr = torchLoom_addr
        self._nc: Client | None = None
        self._js: JetStreamContext | None = None

    async def initialize(self) -> Tuple[Client, JetStreamContext]:
        """Initialize NATS connection and JetStream."""
        self._nc = await nats.connect(self._torchLoom_addr)
        self._js = self._nc.jetstream()
        logger.info(f"Connected to NATS server at {self._torchLoom_addr}")
        return self._nc, self._js

    async def close(self) -> None:
        """Close the NATS connection."""
        if self._nc and not self._nc.is_closed:
            await self._nc.close()
            logger.info("NATS connection closed")

    @property
    def nc(self) -> Client | None:
        """Get the NATS client."""
        return self._nc

    @property
    def js(self) -> JetStreamContext | None:
        """Get the JetStream context."""
        return self._js
