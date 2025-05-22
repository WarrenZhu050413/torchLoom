import pytest
import asyncio
from unittest.mock import patch, AsyncMock
import nats

from torchLoom.weaver import Weaver
from torchLoom.torchLoom_pb2 import EventEnvelope

@pytest.mark.asyncio
async def test_weaver_integration(monkeypatch_nats_connect, mock_nats_client, mock_nats_msg):
    """
    Integration test that simulates a full workflow:
    1. Initialize Weaver
    2. Register devices and replicas
    3. Simulate a GPU failure
    4. Verify that replica failure events are sent
    """
    # Initialize the Weaver
    weaver = Weaver()
    await weaver.initialize()
    
    # Setup mock subscriptions
    mock_sub = AsyncMock()
    mock_task = AsyncMock()
    weaver._subscriptions = {}
    
    # Mock the publish method to verify calls
    publish_calls = []
    
    async def mock_publish(subject, data):
        publish_calls.append((subject, data))
    
    weaver._nc.publish = mock_publish
    
    try:
        # Step 1: Register devices
        register_env1 = EventEnvelope()
        register_env1.register_device.device_uuid = "device1"
        register_env1.register_device.replica_id = "replica1"
        mock_nats_msg.data = register_env1.SerializeToString()
        await weaver.message_handler(mock_nats_msg)
        
        register_env2 = EventEnvelope()
        register_env2.register_device.device_uuid = "device1"
        register_env2.register_device.replica_id = "replica2"
        mock_nats_msg.data = register_env2.SerializeToString()
        await weaver.message_handler(mock_nats_msg)
        
        register_env3 = EventEnvelope()
        register_env3.register_device.device_uuid = "device2"
        register_env3.register_device.replica_id = "replica2"
        mock_nats_msg.data = register_env3.SerializeToString()
        await weaver.message_handler(mock_nats_msg)
        
        # Verify device-replica mappings
        assert weaver.device_to_replicas == {
            "device1": {"replica1", "replica2"},
            "device2": {"replica2"}
        }
        assert weaver.replica_to_devices == {
            "replica1": {"device1"},
            "replica2": {"device1", "device2"}
        }
        
        # Step 2: Simulate a GPU failure
        fail_env = EventEnvelope()
        fail_env.monitored_fail.device_uuid = "device1"
        mock_nats_msg.data = fail_env.SerializeToString()
        
        # Mock send_replica_fail_event since we're testing integration, not just the message handler
        weaver.send_replica_fail_event = AsyncMock()
        
        # Clear publish calls before testing
        publish_calls.clear()
        
        await weaver.message_handler(mock_nats_msg)
        
        # Check that replica failure events were published via the mocked method
        assert weaver.send_replica_fail_event.call_count == 2
        weaver.send_replica_fail_event.assert_any_call("replica1")
        weaver.send_replica_fail_event.assert_any_call("replica2")
        
        # Step 3: Test learning rate update
        lr_env = EventEnvelope()
        lr_env.learning_rate.lr = "0.0001"
        mock_nats_msg.data = lr_env.SerializeToString()
        
        # Replace method to test JetStream publish
        js_publish_called = False
        
        async def mock_js_publish(subject, data):
            nonlocal js_publish_called
            js_publish_called = True
            assert subject == "torchLoom.training.reset_lr"
            assert data == b"0.0001"
        
        weaver._nc.jetstream().publish = mock_js_publish
        
        await weaver.message_handler(mock_nats_msg)
        
        assert js_publish_called
    
    finally:
        # Clean up
        await weaver.stop()

@pytest.mark.asyncio
async def test_failure_recovery_workflow(monkeypatch_nats_connect, mock_nats_client, mock_nats_msg):
    """
    Integration test that simulates a failure recovery workflow:
    1. Initialize Weaver
    2. Register devices and replicas
    3. Simulate a GPU failure
    4. Verify that replica failure events are sent
    5. Register a new device to replace the failed one
    6. Verify that mappings are updated
    """
    # Initialize the Weaver
    weaver = Weaver()
    await weaver.initialize()
    
    # Mock the publish method to verify calls
    publish_calls = []
    
    async def mock_publish(subject, data):
        publish_calls.append((subject, data))
    
    weaver._nc.publish = mock_publish
    
    # Mock send_replica_fail_event to make testing easier
    weaver.send_replica_fail_event = AsyncMock()
    
    try:
        # Step 1: Register initial devices
        register_env1 = EventEnvelope()
        register_env1.register_device.device_uuid = "device1"
        register_env1.register_device.replica_id = "replica1"
        mock_nats_msg.data = register_env1.SerializeToString()
        await weaver.message_handler(mock_nats_msg)
        
        # Step 2: Simulate a GPU failure
        fail_env = EventEnvelope()
        fail_env.monitored_fail.device_uuid = "device1"
        mock_nats_msg.data = fail_env.SerializeToString()
        
        # Clear publish calls before testing
        publish_calls.clear()
        
        await weaver.message_handler(mock_nats_msg)
        
        # Check that replica failure event was sent
        assert weaver.send_replica_fail_event.call_count == 1
        weaver.send_replica_fail_event.assert_called_once_with("replica1")
        
        # Step 3: Register a new device to replace the failed one
        register_env2 = EventEnvelope()
        register_env2.register_device.device_uuid = "device2"
        register_env2.register_device.replica_id = "replica1"
        mock_nats_msg.data = register_env2.SerializeToString()
        await weaver.message_handler(mock_nats_msg)
        
        # Verify updated device-replica mappings
        assert weaver.device_to_replicas == {
            "device1": {"replica1"},
            "device2": {"replica1"}
        }
        assert weaver.replica_to_devices == {
            "replica1": {"device1", "device2"}
        }
        
        # Step 4: Simulate learning rate adjustment after recovery
        lr_env = EventEnvelope()
        lr_env.learning_rate.lr = "0.0005"
        mock_nats_msg.data = lr_env.SerializeToString()
        
        # Replace method to test JetStream publish
        js_publish_called = False
        
        async def mock_js_publish(subject, data):
            nonlocal js_publish_called
            js_publish_called = True
            assert subject == "torchLoom.training.reset_lr"
            assert data == b"0.0005"
        
        weaver._nc.jetstream().publish = mock_js_publish
        
        await weaver.message_handler(mock_nats_msg)
        
        assert js_publish_called
    
    finally:
        # Clean up
        await weaver.stop() 