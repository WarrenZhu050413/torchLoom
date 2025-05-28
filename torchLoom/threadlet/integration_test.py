"""
Integration tests for Weaver and ThreadletListener communication.

This module tests the end-to-end communication between the Weaver and 
ThreadletListener components to ensure they can properly exchange messages
through NATS streams.
"""

import asyncio
import multiprocessing
import pytest
import pytest_asyncio
import time
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock, patch, AsyncMock, call

from torchLoom.weaver.weaver import Weaver
from torchLoom.threadlet.listener import ThreadletListener
from torchLoom.proto.torchLoom_pb2 import EventEnvelope, WeaverCommand


class MessageInterceptor:
    """Helper class to intercept and track messages between Weaver and ThreadletListener."""
    
    def __init__(self):
        self.weaver_to_listener_messages: List[Dict[str, Any]] = []
        self.listener_to_weaver_messages: List[Dict[str, Any]] = []
        self.weaver_handlers: Dict[str, Any] = {}
        self.listener_handlers: Dict[str, Any] = {}
    
    def setup_weaver_intercept(self, weaver):
        """Set up message interception for weaver."""
        # Store the original weaver message handler
        self.weaver_handlers["torchLoom.threadlet.events"] = weaver.message_handler
        self.weaver_handlers["torchLoom.external.events"] = weaver.message_handler
        
        # Mock the subscription manager to capture message handlers without actually subscribing
        original_subscribe_js = weaver._subscription_manager.subscribe_js
        
        async def mock_subscribe_js(stream, subject, consumer, message_handler, **kwargs):
            self.weaver_handlers[subject] = message_handler
            # Don't actually subscribe, just capture the handler
            return Mock()
        
        weaver._subscription_manager.subscribe_js = mock_subscribe_js
        
        # Mock the threadlet command publisher to capture outgoing commands
        if weaver.threadlet_command_handler:
            original_publish = weaver.threadlet_command_handler.publish_weaver_command
            
            async def mock_publish_command(command_type, target_process_id, params=None):
                message = {
                    "type": "weaver_command",
                    "command_type": command_type,
                    "target_process_id": target_process_id,
                    "params": params or {}
                }
                self.weaver_to_listener_messages.append(message)
                
                # If we have a listener handler, deliver the message
                if "torchLoom.weaver.commands" in self.listener_handlers:
                    await self._deliver_to_listener(message)
            
            weaver.threadlet_command_handler.publish_weaver_command = mock_publish_command
    
    def setup_listener_intercept(self, listener):
        """Set up message interception for listener."""
        # Mock the subscription manager to capture message handlers
        original_subscribe_js = listener._subscription_manager.subscribe_js
        
        async def mock_subscribe_js(stream, subject, consumer, message_handler, **kwargs):
            self.listener_handlers[subject] = message_handler
            return Mock()
        
        listener._subscription_manager.subscribe_js = mock_subscribe_js
        
        # Mock the threadlet publisher to capture outgoing events
        if hasattr(listener, '_threadlet_publisher') and listener._threadlet_publisher:
            original_publish_heartbeat = listener._threadlet_publisher.publish_heartbeat
            original_publish_training_status = listener._threadlet_publisher.publish_training_status
            original_publish_device_status = listener._threadlet_publisher.publish_device_status
            original_publish_device_registration = listener._threadlet_publisher.publish_device_registration
            
            async def mock_publish_heartbeat(status, metadata=None):
                message = {
                    "type": "heartbeat",
                    "process_id": listener._process_id,
                    "device_uuid": listener._device_uuid,
                    "status": status,
                    "metadata": metadata
                }
                self.listener_to_weaver_messages.append(message)
                
                # If we have a weaver handler, deliver the message
                if "torchLoom.threadlet.events" in self.weaver_handlers:
                    await self._deliver_to_weaver(message)
            
            async def mock_publish_training_status(status_data):
                message = {
                    "type": "training_status",
                    "process_id": listener._process_id,
                    "device_uuid": listener._device_uuid,
                    "status_data": status_data
                }
                self.listener_to_weaver_messages.append(message)
                
                if "torchLoom.threadlet.events" in self.weaver_handlers:
                    await self._deliver_to_weaver(message)
            
            async def mock_publish_device_status(status_data):
                message = {
                    "type": "device_status",
                    "process_id": listener._process_id,
                    "device_uuid": listener._device_uuid,
                    "status_data": status_data
                }
                self.listener_to_weaver_messages.append(message)
                
                if "torchLoom.threadlet.events" in self.weaver_handlers:
                    await self._deliver_to_weaver(message)
            
            async def mock_publish_device_registration():
                message = {
                    "type": "register_device",
                    "process_id": listener._process_id,
                    "device_uuid": listener._device_uuid
                }
                self.listener_to_weaver_messages.append(message)
                
                if "torchLoom.threadlet.events" in self.weaver_handlers:
                    await self._deliver_to_weaver(message)
            
            listener._threadlet_publisher.publish_heartbeat = mock_publish_heartbeat
            listener._threadlet_publisher.publish_training_status = mock_publish_training_status
            listener._threadlet_publisher.publish_device_status = mock_publish_device_status
            listener._threadlet_publisher.publish_device_registration = mock_publish_device_registration
    
    async def _deliver_to_listener(self, message):
        """Deliver a weaver command to the listener."""
        handler = self.listener_handlers.get("torchLoom.weaver.commands")
        if handler:
            # Create protobuf message
            envelope = EventEnvelope()
            weaver_command = envelope.weaver_command
            weaver_command.command_type = message["command_type"]
            weaver_command.target_process_id = message["target_process_id"]
            
            for key, value in message["params"].items():
                weaver_command.params[key] = str(value)
            
            # Create mock NATS message
            mock_msg = Mock()
            mock_msg.data = envelope.SerializeToString()
            mock_msg.subject = "torchLoom.weaver.commands"
            
            # Deliver to handler
            await handler(mock_msg)
    
    async def _deliver_to_weaver(self, message):
        """Deliver a threadlet event to the weaver."""
        handler = self.weaver_handlers.get("torchLoom.threadlet.events")
        if handler:
            # Create protobuf message
            envelope = EventEnvelope()
            
            if message["type"] == "heartbeat":
                heartbeat = envelope.heartbeat
                heartbeat.process_id = message["process_id"]
                heartbeat.device_uuid = message["device_uuid"]
                heartbeat.status = message["status"]
                if message.get("metadata"):
                    for key, value in message["metadata"].items():
                        heartbeat.metadata[key] = str(value)
            
            elif message["type"] == "training_status":
                training_status = envelope.training_status
                training_status.process_id = message["process_id"]
                # Add status_data fields based on what's available
                status_data = message.get("status_data", {})
                if "step" in status_data:
                    training_status.current_step = status_data["step"]
                if "loss" in status_data:
                    training_status.metrics["loss"] = str(status_data["loss"])
            
            elif message["type"] == "device_status":
                device_status = envelope.device_status
                device_status.process_id = message["process_id"]
                device_status.device_uuid = message["device_uuid"]
                status_data = message.get("status_data", {})
                if "utilization" in status_data:
                    device_status.utilization = status_data["utilization"]
            
            elif message["type"] == "register_device":
                register_device = envelope.register_device
                register_device.process_id = message["process_id"]
                register_device.device_uuid = message["device_uuid"]
            
            # Create mock NATS message
            mock_msg = Mock()
            mock_msg.data = envelope.SerializeToString()
            mock_msg.subject = "torchLoom.threadlet.events"
            
            # Deliver to handler
            await handler(mock_msg)


class TestWeaverListenerIntegration:
    """Integration tests for Weaver and ThreadletListener communication."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.process_id = "integration-test-123"
        self.device_uuid = "integration-device-456"
        self.server_id = "integration-server-789"
        self.torchLoom_addr = "nats://localhost:4222"
        
        # Create message interceptor
        self.interceptor = MessageInterceptor()
        
        # Create mock pipe and stop event for listener
        self.mock_pipe = Mock()
        self.mock_pipe.closed = False
        self.stop_event = multiprocessing.Event()
    
    @pytest_asyncio.fixture
    async def weaver(self):
        """Create a Weaver instance for testing."""
        with patch('torchLoom.weaver.weaver.SubscriptionManager') as mock_sub_mgr:
            mock_sub_mgr_instance = AsyncMock()
            mock_sub_mgr_instance.nc = AsyncMock()
            mock_sub_mgr_instance.js = AsyncMock()
            mock_sub_mgr_instance.stream_manager = AsyncMock()
            mock_sub_mgr.return_value = mock_sub_mgr_instance
            
            weaver = Weaver(torchLoom_addr=self.torchLoom_addr, enable_ui=False)
            weaver._subscription_manager = mock_sub_mgr_instance
            
            # Mock ThreadletCommandPublisher
            with patch('torchLoom.weaver.weaver.ThreadletCommandPublisher') as mock_publisher_class:
                mock_publisher = AsyncMock()
                mock_publisher_class.return_value = mock_publisher
                
                await weaver.initialize()
                
                # Set up interception
                self.interceptor.setup_weaver_intercept(weaver)
                
                return weaver
    
    @pytest_asyncio.fixture
    async def listener(self):
        """Create a ThreadletListener instance for testing."""
        with patch('torchLoom.threadlet.listener.SubscriptionManager') as mock_sub_mgr:
            mock_sub_mgr_instance = AsyncMock()
            mock_sub_mgr_instance.nc = AsyncMock()
            mock_sub_mgr_instance.js = AsyncMock()
            mock_sub_mgr.return_value = mock_sub_mgr_instance
            
            listener = ThreadletListener(
                process_id=self.process_id,
                device_uuid=self.device_uuid,
                server_id=self.server_id,
                torchLoom_addr=self.torchLoom_addr,
                pipe_to_main_process=self.mock_pipe,
                stop_event=self.stop_event,
            )
            listener._subscription_manager = mock_sub_mgr_instance
            
            # Mock ThreadletEventPublisher
            with patch('torchLoom.threadlet.listener.ThreadletEventPublisher') as mock_publisher_class:
                mock_publisher = AsyncMock()
                mock_publisher_class.return_value = mock_publisher
                listener._threadlet_publisher = mock_publisher
                
                # Set up interception
                self.interceptor.setup_listener_intercept(listener)
                
                return listener
    
    @pytest.mark.asyncio
    async def test_weaver_sends_update_config_to_listener(self, weaver, listener):
        """Test that weaver can send update_config commands to listener."""
        # Set up listener to capture commands via pipe
        received_commands = []
        
        def mock_send_to_pipe(message_dict):
            received_commands.append(message_dict)
        
        listener._send_dict_to_threadlet = mock_send_to_pipe
        
        # Set up subscriptions for both components
        await weaver._setup_all_streams()
        await listener._setup_subscriptions_with_manager()
        
        # Send update_config command from weaver
        params = {"learning_rate": "0.01", "batch_size": "32"}
        await weaver.threadlet_command_handler.publish_weaver_command(
            command_type="update_config",
            target_process_id=self.process_id,
            params=params
        )
        
        # Give time for message processing
        await asyncio.sleep(0.1)
        
        # Verify message was sent from weaver
        assert len(self.interceptor.weaver_to_listener_messages) == 1
        sent_message = self.interceptor.weaver_to_listener_messages[0]
        assert sent_message["command_type"] == "update_config"
        assert sent_message["target_process_id"] == self.process_id
        assert sent_message["params"] == params
        
        # Verify message was received by listener and forwarded to pipe
        assert len(received_commands) == 1
        received_command = received_commands[0]
        assert received_command["message_type"] == "command"
        assert received_command["command_type"] == "update_config"
        assert received_command["payload"] == params
        assert received_command["process_id"] == self.process_id
    
    @pytest.mark.asyncio
    async def test_weaver_sends_pause_command_to_listener(self, weaver, listener):
        """Test that weaver can send pause commands to listener."""
        received_commands = []
        listener._send_dict_to_threadlet = lambda msg: received_commands.append(msg)
        
        await weaver._setup_all_streams()
        await listener._setup_subscriptions_with_manager()
        
        # Send pause command
        await weaver.threadlet_command_handler.publish_weaver_command(
            command_type="pause",
            target_process_id=self.process_id
        )
        
        await asyncio.sleep(0.1)
        
        # Verify command was sent and received
        assert len(self.interceptor.weaver_to_listener_messages) == 1
        assert self.interceptor.weaver_to_listener_messages[0]["command_type"] == "pause"
        
        assert len(received_commands) == 1
        assert received_commands[0]["command_type"] == "pause"
    
    @pytest.mark.asyncio
    async def test_weaver_ignores_commands_for_different_process(self, weaver, listener):
        """Test that listener ignores commands targeted at different processes."""
        received_commands = []
        listener._send_dict_to_threadlet = lambda msg: received_commands.append(msg)
        
        await weaver._setup_all_streams()
        await listener._setup_subscriptions_with_manager()
        
        # Send command to different process
        await weaver.threadlet_command_handler.publish_weaver_command(
            command_type="update_config",
            target_process_id="different-process-id",
            params={"learning_rate": "0.02"}
        )
        
        await asyncio.sleep(0.1)
        
        # Verify command was sent but not processed by our listener
        assert len(self.interceptor.weaver_to_listener_messages) == 1
        assert len(received_commands) == 0  # Should be ignored
    
    @pytest.mark.asyncio
    async def test_listener_sends_heartbeat_to_weaver(self, weaver, listener):
        """Test that listener can send heartbeat events to weaver."""
        # Mock weaver's status tracker to capture heartbeats
        heartbeats_received = []
        
        async def mock_handle_heartbeat(env, status_tracker, heartbeat_tracker, weaver_publish_command_func):
            heartbeat = env.heartbeat
            heartbeats_received.append({
                "process_id": heartbeat.process_id,
                "device_uuid": heartbeat.device_uuid,
                "status": heartbeat.status
            })
        
        # Replace the heartbeat handler
        weaver._handler_registry.register_handler("heartbeat", mock_handle_heartbeat)
        
        await weaver._setup_all_streams()
        await listener._setup_subscriptions_with_manager()
        
        # Send heartbeat from listener
        await listener._threadlet_publisher.publish_heartbeat(
            status="active",
            metadata={"test": "data"}
        )
        
        await asyncio.sleep(0.1)
        
        # Verify heartbeat was sent and received
        assert len(self.interceptor.listener_to_weaver_messages) == 1
        sent_message = self.interceptor.listener_to_weaver_messages[0]
        assert sent_message["type"] == "heartbeat"
        assert sent_message["status"] == "active"
        assert sent_message["process_id"] == self.process_id
        
        assert len(heartbeats_received) == 1
        received_heartbeat = heartbeats_received[0]
        assert received_heartbeat["process_id"] == self.process_id
        assert received_heartbeat["status"] == "active"
    
    @pytest.mark.asyncio
    async def test_listener_sends_training_status_to_weaver(self, weaver, listener):
        """Test that listener can send training status events to weaver."""
        training_statuses_received = []
        
        async def mock_handle_training_status(env, status_tracker, heartbeat_tracker, weaver_publish_command_func):
            training_status = env.training_status
            training_statuses_received.append({
                "process_id": training_status.process_id,
                "current_step": training_status.current_step,
                "loss": float(training_status.metrics.get("loss", "0"))
            })
        
        weaver._handler_registry.register_handler("training_status", mock_handle_training_status)
        
        await weaver._setup_all_streams()
        await listener._setup_subscriptions_with_manager()
        
        # Send training status from listener
        status_data = {"step": 100, "loss": 0.5}
        await listener._threadlet_publisher.publish_training_status(status_data)
        
        await asyncio.sleep(0.1)
        
        # Verify training status was sent and received
        assert len(self.interceptor.listener_to_weaver_messages) == 1
        sent_message = self.interceptor.listener_to_weaver_messages[0]
        assert sent_message["type"] == "training_status"
        assert sent_message["status_data"] == status_data
        
        assert len(training_statuses_received) == 1
        received_status = training_statuses_received[0]
        assert received_status["process_id"] == self.process_id
        assert received_status["current_step"] == 100
        assert received_status["loss"] == 0.5
    
    @pytest.mark.asyncio
    async def test_listener_sends_device_registration_to_weaver(self, weaver, listener):
        """Test that listener can send device registration events to weaver."""
        registrations_received = []
        
        async def mock_handle_device_registration(env, status_tracker, heartbeat_tracker, weaver_publish_command_func):
            register_device = env.register_device
            registrations_received.append({
                "process_id": register_device.process_id,
                "device_uuid": register_device.device_uuid
            })
        
        weaver._handler_registry.register_handler("register_device", mock_handle_device_registration)
        
        await weaver._setup_all_streams()
        await listener._setup_subscriptions_with_manager()
        
        # Send device registration from listener
        await listener._threadlet_publisher.publish_device_registration()
        
        await asyncio.sleep(0.1)
        
        # Verify registration was sent and received
        assert len(self.interceptor.listener_to_weaver_messages) == 1
        sent_message = self.interceptor.listener_to_weaver_messages[0]
        assert sent_message["type"] == "register_device"
        assert sent_message["process_id"] == self.process_id
        
        assert len(registrations_received) == 1
        received_registration = registrations_received[0]
        assert received_registration["process_id"] == self.process_id
        assert received_registration["device_uuid"] == self.device_uuid
    
    @pytest.mark.asyncio
    async def test_bidirectional_communication_flow(self, weaver, listener):
        """Test complete bidirectional communication flow."""
        # Set up tracking
        received_commands = []
        received_events = []
        
        listener._send_dict_to_threadlet = lambda msg: received_commands.append(msg)
        
        async def mock_handle_event(env, status_tracker, heartbeat_tracker, weaver_publish_command_func):
            payload_type = env.WhichOneof("body")
            if payload_type == "heartbeat":
                received_events.append({"type": "heartbeat", "process_id": env.heartbeat.process_id})
            elif payload_type == "training_status":
                received_events.append({"type": "training_status", "process_id": env.training_status.process_id})
        
        weaver._handler_registry.register_handler("heartbeat", mock_handle_event)
        weaver._handler_registry.register_handler("training_status", mock_handle_event)
        
        await weaver._setup_all_streams()
        await listener._setup_subscriptions_with_manager()
        
        # 1. Weaver sends command to listener
        await weaver.threadlet_command_handler.publish_weaver_command(
            command_type="update_config",
            target_process_id=self.process_id,
            params={"learning_rate": "0.01"}
        )
        
        # 2. Listener sends heartbeat to weaver
        await listener._threadlet_publisher.publish_heartbeat(status="active")
        
        # 3. Listener sends training status to weaver
        await listener._threadlet_publisher.publish_training_status({"step": 50})
        
        # 4. Weaver sends another command
        await weaver.threadlet_command_handler.publish_weaver_command(
            command_type="pause",
            target_process_id=self.process_id
        )
        
        await asyncio.sleep(0.1)
        
        # Verify all messages were exchanged
        assert len(received_commands) == 2
        assert received_commands[0]["command_type"] == "update_config"
        assert received_commands[1]["command_type"] == "pause"
        
        assert len(received_events) == 2
        assert received_events[0]["type"] == "heartbeat"
        assert received_events[1]["type"] == "training_status"
        
        # Verify message counts
        assert len(self.interceptor.weaver_to_listener_messages) == 2
        assert len(self.interceptor.listener_to_weaver_messages) == 2 