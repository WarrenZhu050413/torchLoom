#!/usr/bin/env python3
"""
Basic usage example for torchLoom Weaver with default handlers.

This example demonstrates:
1. Zero-configuration usage with all default handlers
2. How to customize specific handlers
3. How to add custom handlers
4. How to list and inspect handlers
"""

import asyncio
from torchLoom.weaver.core import Weaver
from torchLoom.weaver.handlers import MessageHandler
from torchLoom.proto.torchLoom_pb2 import EventEnvelope
from torchLoom.log.logger import setup_logger

logger = setup_logger(name="example")


class CustomHeartbeatHandler(MessageHandler):
    """Example custom heartbeat handler with additional logging."""
    
    def __init__(self, status_tracker, nats_client=None):
        self.status_tracker = status_tracker
        self.nats_client = nats_client
        self.heartbeat_count = 0
    
    async def handle(self, env: EventEnvelope) -> None:
        """Handle heartbeat with custom logic."""
        if not env.HasField("heartbeat"):
            return
            
        self.heartbeat_count += 1
        heartbeat = env.heartbeat
        
        logger.info(f"🔄 Custom heartbeat #{self.heartbeat_count} from {heartbeat.replica_id}")
        
        # You can add any custom logic here
        # For example, custom alerting, metrics collection, etc.


async def basic_usage_example():
    """Demonstrate basic zero-configuration usage."""
    logger.info("=== Basic Usage Example ===")
    
    # Initialize weaver with all default handlers
    weaver = Weaver(enable_ui=False)  # Disable UI for this example
    await weaver.initialize()
    
    # List all default handlers
    handlers = weaver.list_handlers()
    logger.info("Default handlers loaded:")
    for event_type, handler_class in handlers.items():
        logger.info(f"  {event_type}: {handler_class}")
    
    # Show supported events
    events = weaver.get_supported_events()
    logger.info("\nSupported events:")
    for event_type, description in events.items():
        logger.info(f"  {event_type}: {description}")
    
    await weaver.stop()


async def custom_handler_example():
    """Demonstrate handler customization."""
    logger.info("\n=== Custom Handler Example ===")
    
    # Initialize weaver with defaults
    weaver = Weaver(enable_ui=False)
    await weaver.initialize()
    
    # Override the default heartbeat handler with our custom one
    custom_handler = CustomHeartbeatHandler(weaver.status_tracker)
    weaver.override_handler("heartbeat", custom_handler)
    
    # Verify the handler was changed
    current_handler = weaver.get_handler("heartbeat")
    logger.info(f"Heartbeat handler is now: {current_handler.__class__.__name__}")
    
    # List all handlers to see the change
    handlers = weaver.list_handlers()
    logger.info("Updated handlers:")
    for event_type, handler_class in handlers.items():
        if event_type == "heartbeat":
            logger.info(f"  {event_type}: {handler_class} ⭐ (customized)")
        else:
            logger.info(f"  {event_type}: {handler_class}")
    
    await weaver.stop()


async def main():
    """Run all examples."""
    try:
        await basic_usage_example()
        await custom_handler_example()
        
        logger.info("\n✅ All examples completed successfully!")
        logger.info("🎉 torchLoom Weaver is ready to use with minimal configuration!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 