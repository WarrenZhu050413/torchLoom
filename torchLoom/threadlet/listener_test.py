"""
Unit tests for the ThreadletListener class.

Tests cover:
1. Pipe message processing and communication
2. Stream subscription and weaver command handling
3. Publishing event requests
4. Cleanup functionality

Test Coverage:
- Bidirectional pipe communication with main process
- NATS stream subscriptions and message handling
- Weaver command processing (update_config, pause, resume, stop)
- Event publishing for heartbeats, training status, device status
- Automatic heartbeat loop functionality
- Multiprocessing event monitoring
- Resource cleanup and error handling
"""

import asyncio
import multiprocessing
import pytest
import time
from unittest.mock import Mock, MagicMock, patch, AsyncMock, call
from multiprocessing.connection import Connection

from torchLoom.threadlet.listener import ThreadletListener
from torchLoom.proto.torchLoom_pb2 import EventEnvelope, WeaverCommand


class TestThreadletListener:
    """Test cases for the ThreadletListener class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.process_id = "test-listener-123"
        self.device_uuid = "test-device-456"
        self.server_id = "test-server-789"
        self.torchLoom_addr = "nats://localhost:4222"
        
        # Create mock pipe and stop event
        self.mock_pipe = Mock()
        self.mock_pipe.closed = False
        self.stop_event = multiprocessing.Event()

    @pytest.fixture
    def listener(self):
        """Create a ThreadletListener instance for testing."""
        with patch('torchLoom.threadlet.listener.SubscriptionManager') as mock_sub_mgr:
            mock_sub_mgr_instance = Mock()
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
            return listener

    @pytest.mark.asyncio
    async def test_async_pipe_message_processor_receives_message(self, listener):
        """Test that _async_pipe_message_processor can receive and process messages."""
        # Setup mock pipe behavior
        publish_request = {
            "action": "publish_event",
            "event_type": "heartbeat",
            "event_data": {"status": "active"}
        }
        
        # Mock poll to return True once, then False to exit loop
        self.mock_pipe.poll.side_effect = [True, False]
        
        # Use asyncio.to_thread mock to return our test message
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = publish_request
            
            # Mock the message processing method
            with patch.object(listener, '_process_pipe_message') as mock_process:
                mock_process.return_value = asyncio.Future()
                mock_process.return_value.set_result(None)
                
                # Run the processor for a short time
                task = asyncio.create_task(listener._async_pipe_message_processor())
                
                # Give it time to process
                await asyncio.sleep(0.1)
                
                # Stop the processor
                listener._async_stop_event.set()
                
                # Wait for completion
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except asyncio.TimeoutError:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Verify message was processed
                mock_process.assert_called_once_with(publish_request)

    @pytest.mark.asyncio
    async def test_process_pipe_message_publish_event(self, listener):
        """Test that _process_pipe_message correctly handles publish event requests."""
        publish_request = {
            "action": "publish_event",
            "event_type": "training_status",
            "event_data": {"status_data": {"step": 100}}
        }
        
        # Mock the handle publish event request method
        with patch.object(listener, '_handle_publish_event_request') as mock_handle:
            mock_handle.return_value = asyncio.Future()
            mock_handle.return_value.set_result(None)
            
            await listener._process_pipe_message(publish_request)
            
            mock_handle.assert_called_once_with(publish_request)

    @pytest.mark.asyncio
    async def test_process_pipe_message_unknown_action(self, listener):
        """Test that _process_pipe_message handles unknown actions gracefully."""
        unknown_request = {
            "action": "unknown_action",
            "data": "test"
        }
        
        # This should not raise an exception
        await listener._process_pipe_message(unknown_request)

    @pytest.mark.asyncio
    async def test_process_pipe_message_invalid_data_type(self, listener):
        """Test that _process_pipe_message handles invalid data types gracefully."""
        # This should not raise an exception
        await listener._process_pipe_message("invalid_string_data")
        await listener._process_pipe_message(123)
        await listener._process_pipe_message(None)

    @pytest.mark.asyncio
    async def test_handle_publish_event_request_heartbeat(self, listener):
        """Test handling of heartbeat publish requests."""
        # Setup mock publisher
        mock_publisher = AsyncMock()
        listener._threadlet_publisher = mock_publisher
        
        heartbeat_request = {
            "event_type": "heartbeat",
            "event_data": {
                "status": "active",
                "metadata": {"test": "data"}
            }
        }
        
        await listener._handle_publish_event_request(heartbeat_request)
        
        mock_publisher.publish_heartbeat.assert_called_once_with("active", {"test": "data"})

    @pytest.mark.asyncio
    async def test_handle_publish_event_request_training_status(self, listener):
        """Test handling of training status publish requests."""
        # Setup mock publisher
        mock_publisher = AsyncMock()
        listener._threadlet_publisher = mock_publisher
        
        training_request = {
            "event_type": "training_status",
            "event_data": {
                "status_data": {"step": 100, "loss": 0.5}
            }
        }
        
        await listener._handle_publish_event_request(training_request)
        
        mock_publisher.publish_training_status.assert_called_once_with({"step": 100, "loss": 0.5})

    @pytest.mark.asyncio
    async def test_handle_publish_event_request_device_status(self, listener):
        """Test handling of device status publish requests."""
        # Setup mock publisher
        mock_publisher = AsyncMock()
        listener._threadlet_publisher = mock_publisher
        
        device_request = {
            "event_type": "device_status",
            "event_data": {
                "status_data": {"utilization": 85.0}
            }
        }
        
        await listener._handle_publish_event_request(device_request)
        
        mock_publisher.publish_device_status.assert_called_once_with({"utilization": 85.0})

    @pytest.mark.asyncio
    async def test_handle_publish_event_request_device_registration(self, listener):
        """Test handling of device registration publish requests."""
        # Setup mock publisher
        mock_publisher = AsyncMock()
        listener._threadlet_publisher = mock_publisher
        
        registration_request = {
            "event_type": "device_registration",
            "event_data": {}
        }
        
        await listener._handle_publish_event_request(registration_request)
        
        mock_publisher.publish_device_registration.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_publish_event_request_no_publisher(self, listener):
        """Test handling of publish requests when publisher is not initialized."""
        listener._threadlet_publisher = None
        
        request = {
            "event_type": "heartbeat",
            "event_data": {"status": "active"}
        }
        
        # This should not raise an exception
        await listener._handle_publish_event_request(request)

    @pytest.mark.asyncio
    async def test_handle_publish_event_request_unknown_event_type(self, listener):
        """Test handling of unknown event types in publish requests."""
        # Setup mock publisher
        mock_publisher = AsyncMock()
        listener._threadlet_publisher = mock_publisher
        
        unknown_request = {
            "event_type": "unknown_event",
            "event_data": {"test": "data"}
        }
        
        # This should not raise an exception
        await listener._handle_publish_event_request(unknown_request)

    @pytest.mark.asyncio
    async def test_setup_subscriptions_with_manager(self, listener):
        """Test that stream subscriptions are set up correctly."""
        # Mock subscription manager
        mock_sub_mgr = AsyncMock()
        listener._subscription_manager = mock_sub_mgr
        
        await listener._setup_subscriptions_with_manager()
        
        # Verify subscription was set up
        mock_sub_mgr.subscribe_js.assert_called_once()
        call_args = mock_sub_mgr.subscribe_js.call_args
        
        # Check the subscription parameters
        assert call_args[1]['stream'] == 'WEAVER_COMMANDS_STREAM'
        assert call_args[1]['subject'] == 'torchLoom.weaver.commands'
        assert call_args[1]['consumer'] == f'threadlet-{self.process_id}'
        assert call_args[1]['message_handler'] == listener._handle_weaver_command

    @pytest.mark.asyncio
    async def test_handle_weaver_command_update_config(self, listener):
        """Test handling of weaver update_config commands."""
        # Create a mock NATS message with weaver command
        mock_msg = Mock()
        
        # Create the protobuf message
        envelope = EventEnvelope()
        weaver_command = envelope.weaver_command
        weaver_command.command_type = "update_config"
        weaver_command.target_process_id = self.process_id
        weaver_command.params["learning_rate"] = "0.01"
        weaver_command.params["batch_size"] = "32"
        
        mock_msg.data = envelope.SerializeToString()
        
        # Mock the send method to track calls
        with patch.object(listener, '_send_dict_to_threadlet') as mock_send:
            await listener._handle_weaver_command(mock_msg)
            
            # Verify command was sent to main process
            mock_send.assert_called_once()
            call_args = mock_send.call_args[0][0]
            
            assert call_args['message_type'] == 'command'
            assert call_args['command_type'] == 'update_config'
            assert call_args['payload'] == {'learning_rate': '0.01', 'batch_size': '32'}
            assert call_args['process_id'] == self.process_id

    @pytest.mark.asyncio
    async def test_handle_weaver_command_pause(self, listener):
        """Test handling of weaver pause commands."""
        mock_msg = Mock()
        
        envelope = EventEnvelope()
        weaver_command = envelope.weaver_command
        weaver_command.command_type = "pause"
        weaver_command.target_process_id = self.process_id
        
        mock_msg.data = envelope.SerializeToString()
        
        with patch.object(listener, '_send_dict_to_threadlet') as mock_send:
            await listener._handle_weaver_command(mock_msg)
            
            mock_send.assert_called_once()
            call_args = mock_send.call_args[0][0]
            
            assert call_args['message_type'] == 'command'
            assert call_args['command_type'] == 'pause'

    @pytest.mark.asyncio
    async def test_handle_weaver_command_resume(self, listener):
        """Test handling of weaver resume commands."""
        mock_msg = Mock()
        
        envelope = EventEnvelope()
        weaver_command = envelope.weaver_command
        weaver_command.command_type = "resume"
        weaver_command.target_process_id = self.process_id
        
        mock_msg.data = envelope.SerializeToString()
        
        with patch.object(listener, '_send_dict_to_threadlet') as mock_send:
            await listener._handle_weaver_command(mock_msg)
            
            mock_send.assert_called_once()
            call_args = mock_send.call_args[0][0]
            
            assert call_args['message_type'] == 'command'
            assert call_args['command_type'] == 'resume'

    @pytest.mark.asyncio
    async def test_handle_weaver_command_stop(self, listener):
        """Test handling of weaver stop commands."""
        mock_msg = Mock()
        
        envelope = EventEnvelope()
        weaver_command = envelope.weaver_command
        weaver_command.command_type = "stop"
        weaver_command.target_process_id = self.process_id
        
        mock_msg.data = envelope.SerializeToString()
        
        with patch.object(listener, '_send_dict_to_threadlet') as mock_send:
            await listener._handle_weaver_command(mock_msg)
            
            mock_send.assert_called_once()
            call_args = mock_send.call_args[0][0]
            
            assert call_args['message_type'] == 'command'
            assert call_args['command_type'] == 'stop'

    @pytest.mark.asyncio
    async def test_handle_weaver_command_wrong_target(self, listener):
        """Test that commands for different targets are ignored."""
        mock_msg = Mock()
        
        envelope = EventEnvelope()
        weaver_command = envelope.weaver_command
        weaver_command.command_type = "update_config"
        weaver_command.target_process_id = "different-process-id"
        weaver_command.params["learning_rate"] = "0.01"
        
        mock_msg.data = envelope.SerializeToString()
        
        with patch.object(listener, '_send_dict_to_threadlet') as mock_send:
            await listener._handle_weaver_command(mock_msg)
            
            # Should not send anything since target is different
            mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_weaver_command_invalid_message(self, listener):
        """Test handling of invalid weaver command messages."""
        mock_msg = Mock()
        mock_msg.data = b"invalid_protobuf_data"
        
        # This should not raise an exception
        await listener._handle_weaver_command(mock_msg)

    def test_send_dict_to_threadlet(self, listener):
        """Test sending dictionary messages to the main threadlet process."""
        test_dict = {
            "message_type": "command",
            "command_type": "test",
            "payload": {"test": "data"}
        }
        
        listener._send_dict_to_threadlet(test_dict)
        
        # Verify message was sent via pipe
        self.mock_pipe.send.assert_called_once_with(test_dict)

    def test_send_dict_to_threadlet_broken_pipe(self, listener):
        """Test handling of broken pipe errors when sending messages."""
        self.mock_pipe.send.side_effect = BrokenPipeError("Pipe broken")
        
        test_dict = {"test": "data"}
        
        # This should not raise an exception
        listener._send_dict_to_threadlet(test_dict)

    def test_send_dict_to_threadlet_closed_pipe(self, listener):
        """Test handling of closed pipe when sending messages."""
        self.mock_pipe.closed = True
        
        test_dict = {"test": "data"}
        
        # Should not attempt to send when pipe is closed
        listener._send_dict_to_threadlet(test_dict)
        self.mock_pipe.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_heartbeat_loop(self, listener):
        """Test that the heartbeat loop sends periodic heartbeats."""
        # Setup mock publisher
        mock_publisher = AsyncMock()
        listener._threadlet_publisher = mock_publisher
        
        # Start heartbeat loop
        task = asyncio.create_task(listener._heartbeat_loop())
        
        # Let it run for a short time
        await asyncio.sleep(0.2)
        
        # Stop the loop
        listener._async_stop_event.set()
        
        # Wait for completion
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Should have sent at least one heartbeat
        mock_publisher.publish_heartbeat.assert_called()

    @pytest.mark.asyncio
    async def test_monitor_mp_stop_event(self, listener):
        """Test that the multiprocessing stop event monitoring works."""
        # Start monitoring task
        task = asyncio.create_task(listener._monitor_mp_stop_event())
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        # Set the multiprocessing stop event
        self.stop_event.set()
        
        # Wait for the task to complete
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Verify the asyncio stop event was set
        assert listener._async_stop_event.is_set()

    @pytest.mark.asyncio
    async def test_register_device(self, listener):
        """Test device registration functionality."""
        # Setup mock publisher
        mock_publisher = AsyncMock()
        listener._threadlet_publisher = mock_publisher
        
        await listener._register_device()
        
        mock_publisher.publish_device_registration.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self, listener):
        """Test cleanup functionality."""
        # Setup mock subscription manager
        mock_sub_mgr = AsyncMock()
        listener._subscription_manager = mock_sub_mgr
        
        await listener._cleanup()
        
        # Verify cleanup was called
        mock_sub_mgr.close.assert_called_once()
        assert listener._async_stop_event.is_set()

    @pytest.mark.asyncio
    async def test_cleanup_pipes(self, listener):
        """Test pipe cleanup functionality."""
        await listener._cleanup_pipes()
        
        # Verify pipe was closed
        self.mock_pipe.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_pipes_already_closed(self, listener):
        """Test pipe cleanup when pipe is already closed."""
        self.mock_pipe.closed = True
        
        # Should not attempt to close again
        await listener._cleanup_pipes()
        self.mock_pipe.close.assert_not_called() 