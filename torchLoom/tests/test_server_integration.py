"""
Comprehensive server integration tests for torchLoom.

These tests start actual servers (NATS, Weaver) to test real-world scenarios,
similar to the LighthouseServer tests in the example.
"""

import asyncio
import time
from unittest import TestCase

import pytest
from nats.aio.msg import Msg
from tests.test_utils import NatsTestServer

import nats
from torchLoom.cli import TorchLoomClient
from torchLoom.common.constants import torchLoomConstants
from torchLoom.proto.torchLoom_pb2 import EventEnvelope
from torchLoom.weaver import Weaver


class TestWeaverServerIntegration(TestCase):
    """Integration tests that start actual servers for realistic testing."""

    def test_weaver_startup_and_shutdown_behavior(self):
        """Test that Weaver can start up and shut down properly with real NATS server."""

        async def run_test():
            async with NatsTestServer() as nats_url:
                # Start Weaver server
                weaver = Weaver(nats_url)

                try:
                    start_time = time.time()
                    await weaver.initialize()
                    init_time = time.time() - start_time

                    # Weaver should initialize quickly (under 2 seconds)
                    assert (
                        init_time < 2.0
                    ), f"Weaver initialization took too long: {init_time:.2f}s"

                    # Test that Weaver is properly initialized
                    assert weaver._connection_manager._nc is not None
                    assert weaver._subscription_manager is not None
                    assert weaver._handlers is not None

                    # Test basic subscription setup
                    message_received = []

                    async def test_handler(msg: Msg):
                        message_received.append(msg.data)

                    await weaver.subscribe_nc(
                        torchLoomConstants.subjects.MONITOR, test_handler
                    )

                    # Give subscription time to be ready
                    await asyncio.sleep(0.1)

                    # Test that we can send a message through the system
                    test_nc = await nats.connect(nats_url)
                    test_env = EventEnvelope()
                    test_env.register_device.device_uuid = "test_device"
                    test_env.register_device.replica_id = "test_replica"

                    await test_nc.publish(
                        torchLoomConstants.subjects.MONITOR,
                        test_env.SerializeToString(),
                    )

                    # Wait for message processing
                    await asyncio.sleep(0.5)

                    # Verify message was received
                    assert len(message_received) == 1

                    await test_nc.close()

                finally:
                    # Test graceful shutdown
                    start_time = time.time()
                    await weaver.stop()
                    shutdown_time = time.time() - start_time

                    # Shutdown should be quick (under 1 second)
                    assert (
                        shutdown_time < 1.0
                    ), f"Weaver shutdown took too long: {shutdown_time:.2f}s"

        asyncio.run(run_test())

    def test_weaver_with_cli_client_full_workflow(self):
        """Test a complete workflow with Weaver server and CLI client."""

        async def run_test():
            async with NatsTestServer() as nats_url:
                # Start Weaver server
                weaver = Weaver(nats_url)
                await weaver.initialize()

                # Set up monitoring
                test_nc = await nats.connect(nats_url)
                published_events = []

                async def event_collector(msg: Msg):
                    published_events.append((msg.subject, msg.data))

                # Monitor all relevant subjects
                await test_nc.subscribe(
                    torchLoomConstants.subjects.MONITOR, cb=event_collector
                )
                await test_nc.subscribe(
                    torchLoomConstants.subjects.REPLICA_FAIL, cb=event_collector
                )
                await test_nc.subscribe(
                    torchLoomConstants.subjects.CONFIG_INFO, cb=event_collector
                )

                # Start Weaver subscriptions
                await weaver.subscribe_nc(
                    torchLoomConstants.subjects.MONITOR, weaver.message_handler
                )
                await weaver.subscribe_nc(
                    torchLoomConstants.subjects.CONFIG_INFO, weaver.message_handler
                )

                try:
                    # Test workflow: device registration → failure simulation → recovery
                    start_time = time.time()

                    # Step 1: Register devices using CLI client
                    async with TorchLoomClient(nats_url) as cli_client:
                        await cli_client.register_device("device1", "replica1")
                        await cli_client.register_device("device2", "replica2")
                        await cli_client.register_device(
                            "device1", "replica3"
                        )  # Multi-replica assignment

                    # Wait for registration processing
                    await asyncio.sleep(0.5)

                    # Verify device mappings
                    assert len(weaver.device_to_replicas) == 2
                    assert weaver.get_replicas_for_device("device1") == {
                        "replica1",
                        "replica3",
                    }
                    assert weaver.get_replicas_for_device("device2") == {"replica2"}

                    # Step 2: Simulate device failure
                    async with TorchLoomClient(nats_url) as cli_client:
                        await cli_client.fail_device("device1")

                    # Wait for failure processing
                    await asyncio.sleep(0.5)

                    # Verify replica failure events were generated
                    replica_fails = [
                        data
                        for subject, data in published_events
                        if subject == torchLoomConstants.subjects.REPLICA_FAIL
                    ]
                    assert len(replica_fails) == 2  # Should fail both replicas on device1

                    # Step 3: Adjust learning rate for recovery
                    async with TorchLoomClient(nats_url) as cli_client:
                        await cli_client.reset_learning_rate("0.005")

                    # Wait for learning rate processing
                    await asyncio.sleep(0.5)

                    # Verify config update was published (learning rate is now a config parameter)
                    config_updates = [
                        data
                        for subject, data in published_events
                        if subject == torchLoomConstants.subjects.CONFIG_INFO
                    ]
                    assert len(config_updates) >= 1

                    workflow_time = time.time() - start_time

                    # Entire workflow should complete reasonably quickly (under 5 seconds)
                    assert (
                        workflow_time < 5.0
                    ), f"Workflow took too long: {workflow_time:.2f}s"

                finally:
                    await test_nc.close()
                    await weaver.stop()

        asyncio.run(run_test())

    def test_concurrent_weaver_instances(self):
        """Test that multiple Weaver instances can run concurrently without interference."""

        async def run_test():
            # Start multiple NATS servers on different ports
            async with NatsTestServer(port=4224) as nats_url1:
                async with NatsTestServer(port=4225) as nats_url2:
                    # Start two Weaver instances
                    weaver1 = Weaver(nats_url1)
                    weaver2 = Weaver(nats_url2)

                    await weaver1.initialize()
                    await weaver2.initialize()

                    try:
                        # Set up message collection for both
                        collected_messages1 = []
                        collected_messages2 = []

                        async def collector1(msg: Msg):
                            collected_messages1.append(msg.data)

                        async def collector2(msg: Msg):
                            collected_messages2.append(msg.data)

                        await weaver1.subscribe_nc(
                            torchLoomConstants.subjects.MONITOR, collector1
                        )
                        await weaver2.subscribe_nc(
                            torchLoomConstants.subjects.MONITOR, collector2
                        )

                        # Send messages to each Weaver independently
                        async with TorchLoomClient(nats_url1) as cli1:
                            await cli1.register_device(
                                "weaver1_device", "weaver1_replica"
                            )

                        async with TorchLoomClient(nats_url2) as cli2:
                            await cli2.register_device(
                                "weaver2_device", "weaver2_replica"
                            )

                        # Wait for processing
                        await asyncio.sleep(0.5)

                        # Verify each Weaver only received its own messages
                        assert len(collected_messages1) == 1
                        assert len(collected_messages2) == 1

                        # Verify device mappings are independent
                        assert "weaver1_device" in weaver1.device_to_replicas
                        assert "weaver2_device" in weaver2.device_to_replicas
                        assert "weaver1_device" not in weaver2.device_to_replicas
                        assert "weaver2_device" not in weaver1.device_to_replicas

                    finally:
                        await weaver1.stop()
                        await weaver2.stop()

        asyncio.run(run_test())

    def test_weaver_resilience_to_client_disconnections(self):
        """Test that Weaver handles client connections and disconnections gracefully."""

        async def run_test():
            async with NatsTestServer() as nats_url:
                weaver = Weaver(nats_url)
                await weaver.initialize()

                messages_received = []

                async def message_counter(msg: Msg):
                    messages_received.append(time.time())

                await weaver.subscribe_nc(
                    torchLoomConstants.subjects.MONITOR, message_counter
                )

                try:
                    # Test multiple client connections and disconnections
                    for i in range(3):
                        async with TorchLoomClient(nats_url) as cli_client:
                            await cli_client.register_device(
                                f"device_{i}", f"replica_{i}"
                            )
                            # Client automatically disconnects at end of context

                        # Small delay between connections
                        await asyncio.sleep(0.1)

                    # Wait for all messages to be processed
                    await asyncio.sleep(0.5)

                    # Verify all messages were received despite client disconnections
                    assert len(messages_received) == 3
                    assert len(weaver.device_to_replicas) == 3

                    # Verify Weaver is still responsive after client disconnections
                    async with TorchLoomClient(nats_url) as final_client:
                        await final_client.register_device(
                            "final_device", "final_replica"
                        )

                    await asyncio.sleep(0.2)

                    assert len(messages_received) == 4
                    assert "final_device" in weaver.device_to_replicas

                finally:
                    await weaver.stop()

        asyncio.run(run_test())

    def test_weaver_performance_under_load(self):
        """Test Weaver performance with a high volume of messages."""

        async def run_test():
            async with NatsTestServer() as nats_url:
                weaver = Weaver(nats_url)
                await weaver.initialize()

                processed_count = [0]  # Use list for mutable reference

                async def counting_handler(msg: Msg):
                    processed_count[0] += 1

                await weaver.subscribe_nc(
                    torchLoomConstants.subjects.MONITOR, counting_handler
                )

                try:
                    # Send a burst of messages
                    message_count = 50
                    start_time = time.time()

                    async with TorchLoomClient(nats_url) as cli_client:
                        for i in range(message_count):
                            await cli_client.register_device(
                                f"device_{i}", f"replica_{i}"
                            )

                    # Wait for processing with a reasonable timeout
                    timeout = 10.0  # 10 seconds should be plenty
                    elapsed = 0
                    while processed_count[0] < message_count and elapsed < timeout:
                        await asyncio.sleep(0.1)
                        elapsed = time.time() - start_time

                    processing_time = time.time() - start_time

                    # Verify all messages were processed
                    assert (
                        processed_count[0] == message_count
                    ), f"Only processed {processed_count[0]}/{message_count} messages"

                    # Performance should be reasonable (less than 5 seconds for 50 messages)
                    assert (
                        processing_time < 5.0
                    ), f"Processing took too long: {processing_time:.2f}s"

                    # Verify data integrity
                    assert len(weaver.device_to_replicas) == message_count

                    print(
                        f"Processed {message_count} messages in {processing_time:.2f}s ({message_count/processing_time:.1f} msg/s)"
                    )

                finally:
                    await weaver.stop()

        asyncio.run(run_test())
