"""
Comprehensive CLI integration tests.

These tests use the torchLoom CLI to send messages through a real NATS server
and verify that the Weaver properly processes them.
"""

import asyncio

import pytest
from nats.aio.msg import Msg
from tests.test_utils import NatsTestServer

import nats
from torchLoom.cli import TorchLoomClient
from torchLoom.common.constants import torchLoomConstants
from torchLoom.proto.torchLoom_pb2 import EventEnvelope
from torchLoom.weaver import Weaver


@pytest.mark.asyncio
async def test_cli_device_registration():
    """Test device registration through CLI with real NATS."""
    async with NatsTestServer() as nats_url:
        # Initialize Weaver with fresh state
        weaver = Weaver(nats_url)
        await weaver.initialize()

        # Ensure clean state
        weaver.device_to_replicas.clear()
        weaver.replica_to_devices.clear()

        # Set up monitoring
        test_nc = await nats.connect(nats_url)
        published_messages = []

        async def message_collector(msg: Msg):
            published_messages.append((msg.subject, msg.data))

        # Subscribe to external messages
        await test_nc.subscribe(
            torchLoomConstants.subjects.MONITOR, cb=message_collector
        )

        # Start Weaver subscriptions
        await weaver.subscribe_nc(
            torchLoomConstants.subjects.MONITOR, weaver.message_handler
        )

        try:
            # Use CLI client to register devices
            async with TorchLoomClient(nats_url) as cli_client:
                await cli_client.register_device("device1", "replica1")
                await cli_client.register_device("device1", "replica2")
                await cli_client.register_device("device2", "replica1")

            # Wait for processing
            await asyncio.sleep(0.5)

            # Verify device mappings
            assert weaver.device_to_replicas.get("device1") == {"replica1", "replica2"}
            assert weaver.device_to_replicas.get("device2") == {"replica1"}
            assert weaver.replica_to_devices.get("replica1") == {"device1", "device2"}
            assert weaver.replica_to_devices.get("replica2") == {"device1"}

            # Verify messages were published
            assert len(published_messages) == 3

        finally:
            await test_nc.close()
            await weaver.stop()


@pytest.mark.asyncio
async def test_cli_device_failure():
    """Test device failure simulation through CLI with real NATS."""
    async with NatsTestServer() as nats_url:
        # Initialize Weaver with fresh state
        weaver = Weaver(nats_url)
        await weaver.initialize()

        # Ensure clean state
        weaver.device_to_replicas.clear()
        weaver.replica_to_devices.clear()

        # Set up monitoring
        test_nc = await nats.connect(nats_url)
        published_messages = []

        async def message_collector(msg: Msg):
            published_messages.append((msg.subject, msg.data))

        # Subscribe to both external and replica fail messages
        await test_nc.subscribe(
            torchLoomConstants.subjects.MONITOR, cb=message_collector
        )
        await test_nc.subscribe(
            torchLoomConstants.subjects.REPLICA_FAIL, cb=message_collector
        )

        # Start Weaver subscriptions
        await weaver.subscribe_nc(
            torchLoomConstants.subjects.MONITOR, weaver.message_handler
        )

        try:
            # Use CLI client to register and fail devices
            async with TorchLoomClient(nats_url) as cli_client:
                # First register devices
                await cli_client.register_device("device1", "replica1")
                await cli_client.register_device("device1", "replica2")

                # Wait for registration processing
                await asyncio.sleep(0.5)

                # Clear messages before failure test
                published_messages.clear()

                # Simulate device failure
                await cli_client.fail_device("device1")

            # Wait for failure processing
            await asyncio.sleep(0.5)

            # Check that replica failure events were published
            replica_fail_messages = [
                msg
                for subject, msg in published_messages
                if subject == torchLoomConstants.subjects.REPLICA_FAIL
            ]
            assert len(replica_fail_messages) == 2  # Should have 2 replica failures

        finally:
            await test_nc.close()
            await weaver.stop()


@pytest.mark.asyncio
async def test_cli_learning_rate_update():
    """Test learning rate updates through CLI with real NATS."""
    async with NatsTestServer() as nats_url:
        # Initialize Weaver
        weaver = Weaver(nats_url)
        await weaver.initialize()

        # Set up monitoring
        test_nc = await nats.connect(nats_url)
        published_messages = []

        async def message_collector(msg: Msg):
            published_messages.append((msg.subject, msg.data))

        # Subscribe to config messages (learning rate is now just a config parameter)
        await test_nc.subscribe(
            torchLoomConstants.subjects.CONFIG_INFO, cb=message_collector
        )

        # Start Weaver subscriptions - subscribe to CONFIG_INFO for learning rate updates
        await weaver.subscribe_nc(
            torchLoomConstants.subjects.CONFIG_INFO, weaver.message_handler
        )

        try:
            # Use CLI client to update learning rate
            async with TorchLoomClient(nats_url) as cli_client:
                await cli_client.reset_learning_rate("0.001")
                await asyncio.sleep(0.2)  # Small delay between messages
                await cli_client.reset_learning_rate("0.0005")

            # Wait for processing
            await asyncio.sleep(1.0)  # Increased wait time

            # Debug: print all received messages
            print(f"All published messages: {published_messages}")

            # Check that config messages were published by the Weaver
            config_messages = [
                msg
                for subject, msg in published_messages
                if subject == torchLoomConstants.subjects.CONFIG_INFO
            ]
            print(f"Config messages: {len(config_messages)}")
            assert len(config_messages) >= 1  # Expecting at least 1 message for now

        finally:
            await test_nc.close()
            await weaver.stop()


@pytest.mark.asyncio
async def test_cli_config_info():
    """Test configuration info through CLI with real NATS."""
    async with NatsTestServer() as nats_url:
        # Initialize Weaver
        weaver = Weaver(nats_url)
        await weaver.initialize()

        # Set up monitoring
        test_nc = await nats.connect(nats_url)
        published_messages = []

        async def message_collector(msg: Msg):
            published_messages.append((msg.subject, msg.data))

        # Subscribe to config info messages
        await test_nc.subscribe(
            torchLoomConstants.subjects.CONFIG_INFO, cb=message_collector
        )

        # Start Weaver subscriptions
        await weaver.subscribe_nc(
            torchLoomConstants.subjects.CONFIG_INFO, weaver.message_handler
        )

        try:
            # Use CLI client to send config info
            async with TorchLoomClient(nats_url) as cli_client:
                # Config with learning rate
                await cli_client.send_config_info(
                    {"learning_rate": "0.002", "batch_size": "32"}
                )

                # Config without learning rate
                await cli_client.send_config_info(
                    {"batch_size": "64", "num_workers": "4"}
                )

            # Wait for processing
            await asyncio.sleep(0.5)

            # Check messages
            config_messages = [
                msg
                for subject, msg in published_messages
                if subject == torchLoomConstants.subjects.CONFIG_INFO
            ]

            # Should have at least two config messages (may have more due to Weaver republishing)
            assert len(config_messages) >= 2

        finally:
            await test_nc.close()
            await weaver.stop()


@pytest.mark.asyncio
async def test_cli_full_workflow():
    """Test a complete workflow using CLI with device registration, failure, and recovery."""
    async with NatsTestServer() as nats_url:
        # Initialize Weaver
        weaver = Weaver(nats_url)
        await weaver.initialize()

        # Set up monitoring
        test_nc = await nats.connect(nats_url)
        published_messages = []

        async def message_collector(msg: Msg):
            published_messages.append((msg.subject, msg.data))

        # Subscribe to all relevant subjects
        await test_nc.subscribe(
            torchLoomConstants.subjects.MONITOR, cb=message_collector
        )
        await test_nc.subscribe(
            torchLoomConstants.subjects.REPLICA_FAIL, cb=message_collector
        )
        await test_nc.subscribe(
            torchLoomConstants.subjects.CONFIG_INFO, cb=message_collector
        )

        # Start Weaver subscriptions
        await weaver.subscribe_nc(
            torchLoomConstants.subjects.MONITOR, weaver.message_handler
        )
        await weaver.subscribe_nc(
            torchLoomConstants.subjects.CONFIG_INFO, weaver.message_handler
        )

        try:
            async with TorchLoomClient(nats_url) as cli_client:
                # Step 1: Register initial training setup
                await cli_client.register_device("device1", "replica1")
                await cli_client.register_device("device2", "replica2")
                await cli_client.register_device("device3", "replica3")

                # Set initial learning rate
                await cli_client.reset_learning_rate("0.01")

                # Wait for setup
                await asyncio.sleep(0.5)

                # Verify initial setup
                assert len(weaver.device_to_replicas) == 3
                assert weaver.get_replicas_for_device("device1") == {"replica1"}

                # Clear messages for failure test
                published_messages.clear()

                # Step 2: Simulate device failure
                await cli_client.fail_device("device1")

                # Wait for failure processing
                await asyncio.sleep(0.5)

                # Check replica failure was handled
                replica_fail_messages = [
                    msg
                    for subject, msg in published_messages
                    if subject == torchLoomConstants.subjects.REPLICA_FAIL
                ]
                assert len(replica_fail_messages) == 1

                # Step 3: Register replacement device
                await cli_client.register_device(
                    "device4", "replica1"
                )  # Replace failed device1

                # Adjust learning rate for recovery
                await cli_client.reset_learning_rate("0.005")

                # Wait for recovery processing
                await asyncio.sleep(0.5)

                # Verify recovery
                assert "device4" in weaver.device_to_replicas
                assert weaver.get_replicas_for_device("device4") == {"replica1"}

                # Check config updates (learning rate is now a config parameter)
                config_messages = [
                    msg
                    for subject, msg in published_messages
                    if subject == torchLoomConstants.subjects.CONFIG_INFO
                ]
                assert len(config_messages) >= 1  # At least the recovery config update

        finally:
            await test_nc.close()
            await weaver.stop()


@pytest.mark.asyncio
async def test_consecutive_learning_rate_messages():
    """Test multiple learning rate messages published consecutively are processed correctly."""
    async with NatsTestServer() as nats_url:
        # Initialize Weaver
        weaver = Weaver(nats_url)
        await weaver.initialize()

        # Set up monitoring
        test_nc = await nats.connect(nats_url)
        published_messages = []
        config_messages_received = []

        async def message_collector(msg: Msg):
            published_messages.append((msg.subject, msg.data))
            print(f"Collected message: {msg.subject} -> {msg.data}")

        async def config_monitor(msg: Msg):
            config_messages_received.append(msg.data)
            print(f"Config message received: {msg.data}")

        # Subscribe to config messages (learning rate is now just a config parameter)
        await test_nc.subscribe(
            torchLoomConstants.subjects.CONFIG_INFO, cb=message_collector
        )

        # Start Weaver subscriptions for CONFIG_INFO
        await weaver.subscribe_nc(
            torchLoomConstants.subjects.CONFIG_INFO, weaver.message_handler
        )

        try:
            # Test rapid consecutive learning rate updates
            learning_rates = ["0.01", "0.005", "0.002", "0.001", "0.0005"]

            async with TorchLoomClient(nats_url) as cli_client:
                # Send all learning rate updates quickly
                for i, lr in enumerate(learning_rates):
                    print(f"Sending learning rate: {lr}")
                    await cli_client.reset_learning_rate(lr)
                    # Add small delay to prevent message ordering issues
                    if i < len(learning_rates) - 1:
                        await asyncio.sleep(0.1)

            # Wait for all messages to be processed
            await asyncio.sleep(2.0)  # Longer wait for consecutive messages

            # Debug output
            print(f"Total published messages: {len(published_messages)}")
            print(f"All messages: {published_messages}")
            print(f"Config messages received: {len(config_messages_received)}")

            # Check that all config messages were published by the Weaver
            config_messages = [
                msg
                for subject, msg in published_messages
                if subject == torchLoomConstants.subjects.CONFIG_INFO
            ]

            print(f"Config messages count: {len(config_messages)}")

            # Check if all config messages arrived at NATS level
            print(f"Config messages arriving at NATS: {len(config_messages_received)}")

            # Verify all learning rates were processed as config messages
            # Note: we may see extra messages due to Weaver republishing, so check for at least 5
            assert len(config_messages) >= len(
                learning_rates
            ), f"Expected at least {len(learning_rates)} config messages, got {len(config_messages)}"

            # Verify all original learning rate values are present in the messages
            message_data = [
                msg
                for subject, msg in published_messages
                if subject == torchLoomConstants.subjects.CONFIG_INFO
            ]
            # Convert protobuf messages to check for learning rate values
            found_rates = set()
            for msg_data in message_data:
                try:
                    env = EventEnvelope()
                    env.ParseFromString(msg_data)
                    if (
                        env.HasField("config_info")
                        and "learning_rate" in env.config_info.config_params
                    ):
                        found_rates.add(env.config_info.config_params["learning_rate"])
                except:
                    pass

            print(f"Found learning rates in messages: {found_rates}")

            # Verify all learning rates were processed
            expected_rates = set(learning_rates)
            assert (
                found_rates >= expected_rates
            ), f"Missing learning rates: {expected_rates - found_rates}"

            print(
                "SUCCESS: All consecutive learning rate messages processed correctly as config parameters!"
            )

        finally:
            await test_nc.close()
            await weaver.stop()
