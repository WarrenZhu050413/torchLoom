"""
Integration tests using real NATS server.

These tests start an actual NATS server to provide realistic testing scenarios
with minimal mocking.
"""

import asyncio

import pytest
from nats.aio.msg import Msg
from tests.test_utils import NatsTestServer

import nats
from torchLoom.common.constants import torchLoomConstants
from torchLoom.proto.torchLoom_pb2 import EventEnvelope
from torchLoom.weaver import Weaver


@pytest.mark.asyncio
async def test_weaver_real_nats_integration():
    """
    Integration test with real NATS server:
    1. Start NATS server
    2. Initialize Weaver
    3. Register devices and replicas
    4. Simulate a device failure
    5. Verify that replica failure events are sent
    """
    async with NatsTestServer() as nats_url:
        # Initialize the Weaver with real NATS server
        weaver = Weaver(nats_url)
        await weaver.initialize()

        # Create a client to listen for published messages
        test_nc = await nats.connect(nats_url)

        # Collect published messages
        published_messages = []

        async def message_collector(msg: Msg):
            published_messages.append((msg.subject, msg.data))

        # Subscribe to the replica failure subject
        await test_nc.subscribe(
            torchLoomConstants.subjects.REPLICA_FAIL, cb=message_collector
        )

        # Subscribe to learning rate updates using regular NATS
        await test_nc.subscribe("torchLoom.training.reset_lr", cb=message_collector)

        # Set up Weaver subscriptions
        await weaver.subscribe_nc(
            torchLoomConstants.subjects.MONITOR, weaver.message_handler
        )

        try:
            # Step 1: Register devices
            register_env1 = EventEnvelope()
            register_env1.register_device.device_uuid = "device1"
            register_env1.register_device.replica_id = "replica1"

            register_env2 = EventEnvelope()
            register_env2.register_device.device_uuid = "device1"
            register_env2.register_device.replica_id = "replica2"

            # Wait for subscriptions to be ready
            await asyncio.sleep(0.1)

            # Publish registration events
            await test_nc.publish(
                torchLoomConstants.subjects.MONITOR, register_env1.SerializeToString()
            )
            await test_nc.publish(
                torchLoomConstants.subjects.MONITOR, register_env2.SerializeToString()
            )

            # Wait a bit for processing
            await asyncio.sleep(0.5)

            # Verify device-replica mappings
            assert weaver.device_to_replicas.get("device1") == {"replica1", "replica2"}
            assert weaver.replica_to_devices.get("replica1") == {"device1"}
            assert weaver.replica_to_devices.get("replica2") == {"device1"}

            # Step 2: Simulate a device failure
            fail_env = EventEnvelope()
            fail_env.monitored_fail.device_uuid = "device1"

            # Clear previous messages
            published_messages.clear()

            # Publish failure event
            await test_nc.publish(
                torchLoomConstants.subjects.MONITOR, fail_env.SerializeToString()
            )

            # Wait for failure processing
            await asyncio.sleep(0.5)

            # Check that replica failure events were published
            replica_fail_messages = [
                msg
                for subject, msg in published_messages
                if subject == torchLoomConstants.subjects.REPLICA_FAIL
            ]
            print(f"Replica failure messages: {len(replica_fail_messages)}")
            assert len(replica_fail_messages) == 2  # Should have 2 replica failures

            # Step 3: Test learning rate update
            lr_env = EventEnvelope()
            lr_env.learning_rate.lr = "0.001"

            # Clear previous messages
            published_messages.clear()

            # Publish learning rate update
            await test_nc.publish(
                torchLoomConstants.subjects.MONITOR, lr_env.SerializeToString()
            )

            # Wait for learning rate processing
            await asyncio.sleep(0.5)

            # Check that learning rate update was published
            lr_messages = [
                msg
                for subject, msg in published_messages
                if subject == "torchLoom.training.reset_lr"
            ]
            print(f"Learning rate messages: {len(lr_messages)}")
            assert len(lr_messages) == 1
            assert lr_messages[0] == b"0.001"

        finally:
            # Clean up
            await test_nc.close()
            await weaver.stop()


@pytest.mark.asyncio
async def test_config_info_real_nats():
    """
    Test config_info handling with real NATS server.
    """
    async with NatsTestServer() as nats_url:
        # Initialize the Weaver with real NATS server
        weaver = Weaver(nats_url)
        await weaver.initialize()

        # Create a client to listen for published messages
        test_nc = await nats.connect(nats_url)

        # Collect published messages
        published_messages = []

        async def message_collector(msg: Msg):
            published_messages.append((msg.subject, msg.data))

        # Subscribe to learning rate updates using regular NATS
        await test_nc.subscribe("torchLoom.training.reset_lr", cb=message_collector)

        # Subscribe to config info updates
        await test_nc.subscribe(
            torchLoomConstants.subjects.CONFIG_INFO, cb=message_collector
        )

        # Set up Weaver to subscribe to CONFIG_INFO subject
        await weaver.subscribe_nc(
            torchLoomConstants.subjects.CONFIG_INFO, weaver.message_handler
        )

        try:
            # Test config_info with learning_rate
            env = EventEnvelope()
            env.config_info.config_params["learning_rate"] = "0.002"
            env.config_info.config_params["batch_size"] = "32"

            # Wait for subscriptions to be ready
            await asyncio.sleep(0.1)

            # Publish config_info event
            await test_nc.publish(
                torchLoomConstants.subjects.CONFIG_INFO, env.SerializeToString()
            )

            # Wait for processing
            await asyncio.sleep(0.5)

            # Check that both learning rate and config info were published
            lr_messages = [
                msg
                for subject, msg in published_messages
                if subject == "torchLoom.training.reset_lr"
            ]
            config_messages = [
                msg
                for subject, msg in published_messages
                if subject == torchLoomConstants.subjects.CONFIG_INFO
            ]

            print(f"Learning rate messages: {lr_messages}")
            print(f"Config messages: {len(config_messages)}")

            assert len(lr_messages) == 1
            assert lr_messages[0] == b"0.002"
            assert len(config_messages) == 1

            # Test config_info without learning_rate
            published_messages.clear()

            env2 = EventEnvelope()
            env2.config_info.config_params["batch_size"] = "64"
            env2.config_info.config_params["num_workers"] = "4"

            # Publish config_info event
            await test_nc.publish(
                torchLoomConstants.subjects.CONFIG_INFO, env2.SerializeToString()
            )

            # Wait for processing
            await asyncio.sleep(0.5)

            # Check that only config info was published (no learning rate)
            lr_messages = [
                msg
                for subject, msg in published_messages
                if subject == "torchLoom.training.reset_lr"
            ]
            config_messages = [
                msg
                for subject, msg in published_messages
                if subject == torchLoomConstants.subjects.CONFIG_INFO
            ]

            print(f"Learning rate messages (should be 0): {len(lr_messages)}")
            print(f"Config messages (should be 1): {len(config_messages)}")

            assert len(lr_messages) == 0  # No learning rate update
            assert len(config_messages) == 1

        finally:
            # Clean up
            await test_nc.close()
            await weaver.stop()


@pytest.mark.asyncio
async def test_weaver_subscription_management():
    """
    Test that Weaver properly manages subscriptions with real NATS.
    """
    async with NatsTestServer() as nats_url:
        weaver = Weaver(nats_url)
        await weaver.initialize()

        # Create test client
        test_nc = await nats.connect(nats_url)

        try:
            # Start weaver subscriptions (this is normally done in main())
            await weaver.subscribe_nc(
                torchLoomConstants.subjects.MONITOR, weaver.message_handler
            )

            # Wait a bit for subscription to be ready
            await asyncio.sleep(0.1)

            # Send a test message
            test_env = EventEnvelope()
            test_env.register_device.device_uuid = "test_device"
            test_env.register_device.replica_id = "test_replica"

            print(f"Publishing to subject: {torchLoomConstants.subjects.MONITOR}")
            await test_nc.publish(
                torchLoomConstants.subjects.MONITOR, test_env.SerializeToString()
            )

            # Wait longer for processing
            await asyncio.sleep(0.5)

            # Debug: check what mappings we have
            print(f"Device mappings: {weaver.device_to_replicas}")
            print(f"Replica mappings: {weaver.replica_to_devices}")

            # Verify the message was processed
            assert "test_device" in weaver.device_to_replicas
            assert "test_replica" in weaver.replica_to_devices

        finally:
            await test_nc.close()
            await weaver.stop()
