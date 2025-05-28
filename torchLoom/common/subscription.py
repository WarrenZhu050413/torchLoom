"""
Subscription management for the torchLoom Weaver.

This module handles NATS and JetStream subscription management,
following the single responsibility principle from AGENTS.md.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Tuple

import nats.errors
from nats.js import errors as js_errors
from nats.aio.client import Client
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

import nats
from torchLoom.common.constants import NatsConstants, LoggerConstants, TimeConstants
from torchLoom.log_utils.log_utils import log_and_raise_exception
from torchLoom.log_utils.logger import setup_logger

logger = setup_logger(name="subscription_manager")  # Renamed logger


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
    """Manages NATS and JetStream subscriptions, including connection lifecycle."""

    def __init__(
        self,
        torchLoom_addr: str = NatsConstants.DEFAULT_ADDR,
        stop_event: asyncio.Event | None = None,
    ):
        self._torchLoom_addr = torchLoom_addr
        self._nc: Client | None = None
        self._js: JetStreamContext | None = None
        # If a stop_event is provided, use it; otherwise, create a new one.
        # This allows external control over stopping subscriptions if needed.
        self._stop_event = stop_event if stop_event is not None else asyncio.Event()
        self._subscriptions: Dict[str, Tuple[Any, asyncio.Task | None]] = {}
        self._stream_manager: StreamManager | None = None  # Initialize after connection

        # Configuration from Config
        self._nc_timeout = TimeConstants.NC_TIMEOUT
        self._exception_sleep = TimeConstants.EXCEPTION_SLEEP
        # Keep track of tasks created by this manager for robust cleanup
        self._managed_tasks: List[asyncio.Task] = []

    async def initialize(self) -> None:
        """Initialize NATS connection and JetStream."""
        if self._nc and not self._nc.is_closed:
            logger.info("NATS connection already initialized.")
            return

        try:
            self._nc = await nats.connect(self._torchLoom_addr)
            self._js = self._nc.jetstream()
            self._stream_manager = StreamManager(self._js)
            logger.info(f"Connected to NATS server at {self._torchLoom_addr}")
        except Exception as e:
            logger.exception(
                f"Failed to connect to NATS at {self._torchLoom_addr}: {e}"
            )
            # Optionally, re-raise or handle to prevent app from starting without NATS
            raise

    async def close(self) -> None:
        """Gracefully close all subscriptions and the NATS connection."""
        logger.info(
            "Closing SubscriptionManager: stopping subscriptions and NATS connection."
        )
        await self.stop_all_subscriptions()
        if self._nc and not self._nc.is_closed:
            try:
                await self._nc.drain()  # Ensure all buffered messages are sent
                logger.info("NATS connection drained.")
            except Exception as e:
                logger.warning(f"Error draining NATS connection: {e}")
            finally:
                try:
                    await self._nc.close()
                    logger.info("NATS connection closed.")
                except Exception as e:
                    logger.warning(f"Error closing NATS connection: {e}")
        self._nc = None
        self._js = None
        self._stream_manager = None
        logger.info("SubscriptionManager closed.")

    @property
    def nc(self) -> Client:
        """Get the NATS client. Raises RuntimeError if not initialized or closed."""
        if not self._nc or self._nc.is_closed:
            log_and_raise_exception(
                logger,
                "NATS client is not initialized or has been closed. Call initialize() first.",
            )
        return self._nc

    @property
    def js(self) -> JetStreamContext:
        """Get the JetStream context. Raises RuntimeError if not initialized or closed."""
        if (
            not self._js or not self._nc or self._nc.is_closed
        ):  # Check nc as well, as js depends on it
            log_and_raise_exception(
                logger,
                "JetStream context is not initialized or NATS connection is closed. Call initialize() first.",
            )
        return self._js

    @property
    def stream_manager(self) -> StreamManager:
        """Get the StreamManager. Raises RuntimeError if not initialized."""
        if not self._stream_manager:  # Check nc as well
            log_and_raise_exception(
                logger, "StreamManager is not initialized. Call initialize() first."
            )
        return self._stream_manager

    def _create_managed_task(
        self, awaitable: Awaitable, name: str | None = None
    ) -> asyncio.Task:
        """Creates an asyncio.Task and adds it to the managed tasks list for cleanup."""
        task = asyncio.create_task(awaitable, name=name)
        self._managed_tasks.append(task)
        return task

    async def subscribe_js(
        self,
        stream: str,
        subject: str,
        consumer: str,
        message_handler: Callable[[Msg], Awaitable[None]],
        # Add config for pull subscribe if needed, e.g., batch size, expires
    ) -> None:
        """Subscribe to a JetStream subject using pull subscription and a listening task."""
        self._validate_js_connection()

        logger.info(
            f"Subscribing to JS subject '{subject}' on stream '{stream}' with consumer '{consumer}'"
        )

        try:
            psub = await self.js.pull_subscribe(
                subject, durable=consumer, stream=stream
            )
            logger.info(
                f"Pull subscription created for {subject} on stream {stream} with consumer {consumer}"
            )
        except Exception as e:
            log_and_raise_exception(
                logger,
                f"Failed to create pull subscription for {subject} on {stream}: {e}",
            )
            return  # Or raise depending on desired error handling

        async def listen_to_js_subscription():
            logger.info(
                f"Task started: Listening on JS subject '{subject}' (Consumer: '{consumer}')"
            )
            while not self._stop_event.is_set():
                if self.nc.is_closed or self.nc.is_draining:
                    logger.warning(
                        f"NATS connection closed or draining, stopping listener for {subject}."
                    )
                    break
                try:
                    msgs = await psub.fetch(1, timeout=self._nc_timeout)  # Batch size 1
                    if (
                        not msgs and self._stop_event.is_set()
                    ):  # Check stop event after timeout
                        break

                    for msg in msgs:
                        if self._stop_event.is_set():
                            break
                        try:
                            await message_handler(msg)
                            await msg.ack()
                        except Exception as mh_e:
                            logger.exception(
                                f"Error processing message from {subject} (consumer: {consumer}): {mh_e}. Message: {msg.data[:100]}..."
                            )
                            # Decide on nack, term, or let it timeout based on error
                            # For now, just log and continue, message will be redelivered after ack_wait.
                            await asyncio.sleep(
                                self._exception_sleep
                            )  # Brief pause after error

                except nats.errors.TimeoutError:
                    if self._stop_event.is_set():  # Check again after timeout
                        logger.debug(
                            f"Stop event set, listener for {subject} exiting due to timeout."
                        )
                        break
                    continue  # Normal timeout, no messages
                except (
                    js_errors.APIError
                ) as js_e:  # More specific NATS/JS errors
                    logger.error(
                        f"JetStream error on {subject} (consumer: {consumer}): {js_e}"
                    )
                    if self._stop_event.is_set():
                        break
                    await asyncio.sleep(
                        self._exception_sleep * 2
                    )  # Longer sleep for JS errors
                except Exception as e:
                    if self._stop_event.is_set():
                        logger.info(
                            f"Stop event set, listener for {subject} (consumer: {consumer}) exiting: {e}"
                        )
                        break
                    logger.exception(
                        f"Unexpected error fetching/processing messages from {subject} (consumer: {consumer}): {e}"
                    )
                    await asyncio.sleep(self._exception_sleep)
            logger.info(
                f"Task finished: Listening on JS subject '{subject}' (Consumer: '{consumer}')"
            )

        task = self._create_managed_task(
            listen_to_js_subscription(), name=f"js_listener_{subject}_{consumer}"
        )
        self._subscriptions[f"js:{subject}:{consumer}"] = (
            psub,
            task,
        )  # Store with a more unique key

    async def subscribe_nc(
        self, subject: str, message_handler: Callable[[Msg], Awaitable[None]]
    ) -> None:
        """Subscribe to a regular NATS subject using a callback, wrapped in a monitoring task for consistency."""
        self._validate_nc_connection()
        logger.info(f"Subscribing to NATS subject '{subject}'")

        async def callback_wrapper(msg: Msg) -> None:
            if self._stop_event.is_set():
                logger.debug(f"Stop event set, skipping message on {subject}")
                return
            try:
                logger.debug(f"Processing message on {subject}: {msg.data[:50]}...")
                await message_handler(msg)
                logger.debug(f"Successfully processed message on {subject}")
            except Exception as e:
                logger.exception(f"Error processing message from {subject}: {e}")
                await asyncio.sleep(self._exception_sleep)

        try:
            sub = await self.nc.subscribe(subject, cb=callback_wrapper)
            logger.info(
                f"Successfully subscribed to NATS subject '{subject}' with callback."
            )
        except Exception as e:
            log_and_raise_exception(
                logger, f"Failed to subscribe to NATS subject {subject}: {e}"
            )
            return

        # For NATS core subscriptions, the library handles the listening loop.
        # The task here is more for tracking and clean resource management via _managed_tasks
        # and ensuring stop_all_subscriptions can handle it uniformly.
        # It doesn't actively listen itself but represents the active subscription.
        async def monitor_subscription_health():
            logger.debug(f"Task started: Monitoring NATS subscription for '{subject}'")
            while (
                not self._stop_event.is_set()
                and not self.nc.is_closed
                and sub.is_active
            ):
                await asyncio.sleep(1)  # Check periodically
            logger.info(
                f"Task finished: Monitoring NATS subscription for '{subject}' (active: {sub.is_active})"
            )

        task = self._create_managed_task(
            monitor_subscription_health(), name=f"nc_monitor_{subject}"
        )
        self._subscriptions[f"nc:{subject}"] = (
            sub,
            task,
        )  # Store with a more unique key

    async def stop_all_subscriptions(self) -> None:
        """Unsubscribe from all subjects and cancel all listening/monitoring tasks."""
        logger.info(
            f"Stopping all {len(self._subscriptions)} subscriptions and {len(self._managed_tasks)} tasks."
        )
        self._stop_event.set()  # Signal all listening loops to stop

        # First, unsubscribe NATS core subscriptions if they have an unsubscribe method
        for key, (sub_obj, _) in list(
            self._subscriptions.items()
        ):  # Iterate over a copy
            if hasattr(sub_obj, "unsubscribe"):
                try:
                    logger.debug(f"Unsubscribing from {key}...")
                    await sub_obj.unsubscribe()
                    logger.info(f"Unsubscribed from {key}.")
                except Exception as e:
                    logger.warning(f"Error unsubscribing from {key}: {e}")
            # For pull subscriptions, actual "unsubscription" is effectively stopping the fetch loop
            # and consumer deletion if it's ephemeral (which we are not managing here).
            # Durable consumers persist.

        # Cancel all managed tasks (JS listeners, NC monitors)
        # Give tasks a chance to finish cleanly after stop_event is set
        if self._managed_tasks:
            logger.debug(f"Cancelling {len(self._managed_tasks)} managed tasks...")
            for task in self._managed_tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete cancellation
            results = await asyncio.gather(*self._managed_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                task_name = (
                    self._managed_tasks[i].get_name()
                    if hasattr(self._managed_tasks[i], "get_name")
                    else f"task_{i}"
                )
                if isinstance(result, asyncio.CancelledError):
                    logger.debug(f"Task '{task_name}' was cancelled successfully.")
                elif isinstance(result, Exception):
                    logger.warning(
                        f"Task '{task_name}' raised an exception during cancellation or execution: {result}"
                    )
                else:
                    logger.debug(f"Task '{task_name}' completed.")

        self._managed_tasks.clear()
        self._subscriptions.clear()
        logger.info("All subscriptions stopped and tasks cancelled.")

    def _validate_nc_connection(self) -> None:
        """Validate that NATS connection is available and not closed."""
        if not self._nc or self._nc.is_closed:
            log_and_raise_exception(
                logger,
                "NATS connection is not initialized or is closed. Call initialize() first.",
            )

    def _validate_js_connection(self) -> None:
        """Validate that JetStream context is available (implies NATS connection is also up)."""
        self._validate_nc_connection()  # JS relies on NC
        if not self._js:
            log_and_raise_exception(
                logger, "JetStream context is not initialized. Call initialize() first."
            )
