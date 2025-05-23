"""
Subscription management for the torchLoom Weaver.

This module handles NATS and JetStream subscription management,
following the single responsibility principle from AGENTS.md.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Awaitable, Tuple, Any

import nats
import nats.errors
from nats.aio.client import Client
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from torchLoom.config import Config
from torchLoom.constants import torchLoomConstants, NC, JS
from torchLoom.log.logger import setup_logger
from torchLoom.log.log_utils import log_and_raise_exception
from torchLoom.utils import cancel_subscriptions

logger = setup_logger(name="weaver_subscription")


class StreamManager:
    """Manages JetStream stream creation and configuration."""
    
    def __init__(self, js: JetStreamContext):
        self._js = js
    
    async def maybe_create_stream(self, stream: str, subjects: List[str]) -> None:
        """Create a stream if it doesn't exist."""
        try:
            await self._js.add_stream(name=stream, subjects=subjects)
            logger.info(f"Created stream: {stream}")
        except Exception as e:
            # Handle StreamAlreadyExistsError more generically since the exact exception class might not be available
            if "already exists" in str(e).lower():
                logger.info(f"Stream {stream} already exists, continuing...")
            else:
                logger.exception(f"Error creating stream {stream}: {e}")
                raise


class SubscriptionManager:
    """Manages NATS and JetStream subscriptions."""
    
    def __init__(
        self, 
        nc: Client, 
        js: JetStreamContext,
        stop_event: asyncio.Event
    ):
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
        message_handler: Callable[[Msg], Awaitable[None]]
    ) -> None:
        """Subscribe to a JetStream subject."""
        self._validate_js_connection()
        
        await self._stream_manager.maybe_create_stream(stream, [subject])
        
        logger.info(f"Subscribing to {subject} on stream {stream} with consumer {consumer}")
        
        psub = await self._js.pull_subscribe(subject, durable=consumer, stream=stream)
        
        logger.info(f"Pull subscription created for {subject} on stream {stream} with consumer {consumer}")
        
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
                    await asyncio.gather(
                        *[message_handler(msg) for msg in msgs]
                    )
                    
                    await asyncio.gather(
                        *[msg.ack() for msg in msgs]
                    )
                except Exception as e:
                    logger.exception(f"Error processing messages from {subject}: {e}")
        
        task = asyncio.create_task(listen_to_js_subscription())
        self._subscriptions[subject] = (psub, task)
    
    async def subscribe_nc(
        self, 
        subject: str, 
        message_handler: Callable[[Msg], Awaitable[None]]
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
            log_and_raise_exception(logger, "NATS connection is not initialized, call initialize() first")
    
    def _validate_js_connection(self) -> None:
        """Validate that JetStream connection is available."""
        if self._js is None:
            log_and_raise_exception(logger, "JetStream is not initialized, call initialize() first")


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