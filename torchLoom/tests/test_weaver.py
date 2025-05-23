"""
Essential unit tests for Weaver core functionality.

These tests focus on the core Weaver functionality with minimal mocking,
focusing on initialization, device mapping, and basic error handling.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nats.aio.client import Client
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

import nats
from torchLoom.common.config import Config
from torchLoom.common.constants import JS, NC, torchLoomConstants
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope
from torchLoom.weaver import Weaver

logger = setup_logger(name="test_weaver", log_file="/tmp/test_weaver.log")


class TestWeaverEssentials:
    """Essential tests for Weaver core functionality."""

    def test_init(self, weaver):
        """Test that Weaver initializes with correct default values."""
        assert (
            weaver._connection_manager._torchLoom_addr
            == torchLoomConstants.DEFAULT_ADDR
        )
        assert weaver._device_mapper.device_to_replicas == {}
        assert weaver._device_mapper.replica_to_devices == {}
        assert weaver.seq == 0
        assert weaver._connection_manager._nc is None
        assert weaver._connection_manager._js is None

    @pytest.mark.asyncio
    async def test_initialize(self, weaver):
        """Test that initialize sets up NATS connection and JetStream."""
        # Mock nats.connect to return our mock client
        mock_nc = AsyncMock(spec=Client)
        mock_nc.is_closed = False
        mock_js = AsyncMock(spec=JetStreamContext)
        mock_nc.jetstream.return_value = mock_js

        # Patch nats.connect in the scope of this test
        with patch("nats.connect", return_value=mock_nc) as mock_connect:
            await weaver.initialize()

            # Verify the method was called with the expected arguments
            mock_connect.assert_called_once_with(torchLoomConstants.DEFAULT_ADDR)

            # Verify the client and jetstream were properly set
            assert weaver._connection_manager._nc is mock_nc
            assert weaver._connection_manager._js is mock_js
            assert weaver._subscription_manager is not None

    def test_device_replica_mapping(self, weaver):
        """Test that device-to-replica and replica-to-device mappings work correctly."""
        # Setup test data
        weaver._device_mapper.device_to_replicas = {
            "device1": {"replica1", "replica2"},
            "device2": {"replica2", "replica3"},
        }
        weaver._device_mapper.replica_to_devices = {
            "replica1": {"device1"},
            "replica2": {"device1", "device2"},
            "replica3": {"device2"},
        }

        # Test get_replicas_for_device
        assert weaver.get_replicas_for_device("device1") == {"replica1", "replica2"}
        assert weaver.get_replicas_for_device("device2") == {"replica2", "replica3"}
        assert weaver.get_replicas_for_device("unknown") == set()

        # Test get_devices_for_replica
        assert weaver.get_devices_for_replica("replica1") == {"device1"}
        assert weaver.get_devices_for_replica("replica2") == {"device1", "device2"}
        assert weaver.get_devices_for_replica("replica3") == {"device2"}
        assert weaver.get_devices_for_replica("unknown") == set()

    @pytest.mark.asyncio
    async def test_register_device_message_handler(
        self, initialized_weaver, mock_nats_msg
    ):
        """Test that register_device message handler updates mappings correctly."""
        # Create a register device event
        env = EventEnvelope()
        env.register_device.device_uuid = "test_device"
        env.register_device.replica_id = "test_replica"
        mock_nats_msg.data = env.SerializeToString()

        # Call the message handler
        await initialized_weaver.message_handler(mock_nats_msg)

        # Check that mappings were updated
        assert "test_device" in initialized_weaver.device_to_replicas
        assert "test_replica" in initialized_weaver.replica_to_devices
        assert "test_replica" in initialized_weaver.device_to_replicas["test_device"]
        assert "test_device" in initialized_weaver.replica_to_devices["test_replica"]

    @pytest.mark.asyncio
    async def test_runtime_errors(self, weaver):
        """Test that methods raise appropriate errors when called before initialize."""

        async def mock_handler(msg):
            pass

        with pytest.raises(RuntimeError, match="Weaver not initialized"):
            await weaver.subscribe_js(
                "test_stream", "test.subject", "test_consumer", mock_handler
            )

        with pytest.raises(RuntimeError, match="Weaver not initialized"):
            await weaver.subscribe_nc("test.subject", mock_handler)

    @pytest.mark.asyncio
    async def test_stop_functionality(self, initialized_weaver):
        """Test that stop properly cleans up resources."""
        # Mock the subscription manager stop method
        initialized_weaver._subscription_manager.stop_all_subscriptions = AsyncMock()

        # Call the function
        await initialized_weaver.stop()

        # Check that event was set
        assert initialized_weaver._stop_nats.is_set()

        # Check that subscription manager stop was called
        initialized_weaver._subscription_manager.stop_all_subscriptions.assert_called_once()

        # Check that connection was closed
        initialized_weaver._connection_manager._nc.close.assert_called_once()
