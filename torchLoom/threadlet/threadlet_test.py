"""
Unit tests for the Threadlet class.

Tests cover:
1. Listener process spawning
2. Pipe message processing
3. Command handling with different command types
4. Cleanup functionality

Test Coverage:
- Process management (start, stop, cleanup)
- Inter-process communication via pipes
- Handler registration and command processing
- Error handling (broken pipes, missing handlers)
- Graceful termination of processes and threads
"""

import asyncio
import multiprocessing
import pytest
import threading
import time
from unittest.mock import Mock, MagicMock, patch, call
from multiprocessing.connection import Connection

from torchLoom.threadlet.threadlet import Threadlet
from torchLoom.common.handlers import HandlerRegistry


class TestThreadlet:
    """Test cases for the Threadlet class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.process_id = "test-threadlet-123"
        self.device_uuid = "test-device-456"
        self.torchLoom_addr = "nats://localhost:4222"

    @patch('torchLoom.threadlet.threadlet.multiprocessing.Process')
    @patch('torchLoom.threadlet.threadlet.multiprocessing.Pipe')
    @patch('torchLoom.threadlet.threadlet.threading.Thread')
    def test_start_spawns_listener_process(self, mock_thread, mock_pipe, mock_process):
        """Test that start() successfully spawns the listener process."""
        # Setup mocks
        mock_listener_conn = Mock()
        mock_main_conn = Mock()
        mock_pipe.return_value = (mock_listener_conn, mock_main_conn)
        
        mock_process_instance = Mock()
        mock_process.return_value = mock_process_instance
        mock_process_instance.pid = 12345
        
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance

        # Create threadlet and start
        threadlet = Threadlet(
            process_id=self.process_id,
            device_uuid=self.device_uuid,
            torchLoom_addr=self.torchLoom_addr
        )
        
        threadlet.start()

        # Verify process was created and started
        mock_process.assert_called_once()
        mock_process_instance.start.assert_called_once()
        
        # Verify pipe was created
        mock_pipe.assert_called_once_with(duplex=True)
        
        # Verify thread was created and started for pipe listener
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()
        
        # Verify process attributes are set correctly
        assert threadlet._threadlet_listener_process == mock_process_instance
        assert threadlet._pipe_listener_thread == mock_thread_instance
        
        # Cleanup
        threadlet._cleanup()

    def test_pipe_message_processor_receives_command(self):
        """Test that _pipe_message_processor_loop receives and processes commands from listener."""
        threadlet = Threadlet(
            process_id=self.process_id,
            device_uuid=self.device_uuid,
            torchLoom_addr=self.torchLoom_addr
        )
        
        # Setup mock pipe connection
        mock_main_conn = Mock()
        threadlet._main_pipe_conn = mock_main_conn
        
        # Mock the poll and recv behavior
        command_dict = {
            'message_type': 'command',
            'command_type': 'update_config',
            'payload': {'learning_rate': 0.01}
        }
        
        # First poll returns True (message available), subsequent calls return False
        mock_main_conn.poll.side_effect = [True] + [False] * 10
        mock_main_conn.recv.return_value = command_dict
        
        # Mock the handler method to track calls
        with patch.object(threadlet, '_handle_command_dict') as mock_handler:
            # Set up the stop event
            threadlet._pipe_listener_stop_event = threading.Event()
            
            # Run the pipe processor for a short time
            processor_thread = threading.Thread(
                target=threadlet._pipe_message_processor_loop,
                daemon=True
            )
            processor_thread.start()
            
            # Give it time to process the message
            time.sleep(0.2)
            
            # Stop the processor
            threadlet._pipe_listener_stop_event.set()
            processor_thread.join(timeout=1.0)
            
            # Verify the command was handled
            mock_handler.assert_called_once_with(command_dict)

    def test_handle_command_dict_update_config(self):
        """Test that _handle_command_dict properly handles update_config commands."""
        threadlet = Threadlet(
            process_id=self.process_id,
            device_uuid=self.device_uuid,
            torchLoom_addr=self.torchLoom_addr
        )
        
        # Register test handlers
        learning_rate_handler = Mock()
        batch_size_handler = Mock()
        threadlet.register_handler('learning_rate', learning_rate_handler)
        threadlet.register_handler('batch_size', batch_size_handler)
        
        # Test update_config command
        command_dict = {
            'message_type': 'command',
            'command_type': 'update_config',
            'payload': {
                'learning_rate': 0.01,
                'batch_size': 32,
                'unknown_param': 'value'  # This should be ignored (no handler)
            }
        }
        
        threadlet._handle_command_dict(command_dict)
        
        # Verify handlers were called with correct values
        learning_rate_handler.assert_called_once_with(0.01)
        batch_size_handler.assert_called_once_with(32)

    def test_handle_command_dict_pause_resume_stop(self):
        """Test that _handle_command_dict properly handles pause, resume, and stop commands."""
        threadlet = Threadlet(
            process_id=self.process_id,
            device_uuid=self.device_uuid,
            torchLoom_addr=self.torchLoom_addr
        )
        
        # Register test handlers
        pause_handler = Mock()
        resume_handler = Mock()
        threadlet.register_handler('pause_training', pause_handler)
        threadlet.register_handler('resume_training', resume_handler)
        
        # Test pause command
        pause_command = {
            'message_type': 'command',
            'command_type': 'pause',
            'payload': {}
        }
        threadlet._handle_command_dict(pause_command)
        pause_handler.assert_called_once()
        
        # Test resume command
        resume_command = {
            'message_type': 'command',
            'command_type': 'resume',
            'payload': {}
        }
        threadlet._handle_command_dict(resume_command)
        resume_handler.assert_called_once()
        
        # Test stop command
        with patch.object(threadlet, 'stop') as mock_stop:
            stop_command = {
                'message_type': 'command',
                'command_type': 'stop',
                'payload': {}
            }
            threadlet._handle_command_dict(stop_command)
            mock_stop.assert_called_once()

    def test_handle_command_dict_no_handler_registered(self):
        """Test that _handle_command_dict handles missing handlers gracefully."""
        threadlet = Threadlet(
            process_id=self.process_id,
            device_uuid=self.device_uuid,
            torchLoom_addr=self.torchLoom_addr
        )
        
        # No handlers registered
        command_dict = {
            'message_type': 'command',
            'command_type': 'update_config',
            'payload': {
                'unknown_param': 'value'
            }
        }
        
        # This should not raise an exception
        threadlet._handle_command_dict(command_dict)

    def test_handle_command_dict_unknown_command_type(self):
        """Test that _handle_command_dict handles unknown command types gracefully."""
        threadlet = Threadlet(
            process_id=self.process_id,
            device_uuid=self.device_uuid,
            torchLoom_addr=self.torchLoom_addr
        )
        
        command_dict = {
            'message_type': 'command',
            'command_type': 'unknown_command',
            'payload': {}
        }
        
        # This should not raise an exception
        threadlet._handle_command_dict(command_dict)

    @patch('torchLoom.threadlet.threadlet.multiprocessing.Process')
    @patch('torchLoom.threadlet.threadlet.multiprocessing.Pipe')
    @patch('torchLoom.threadlet.threadlet.threading.Thread')
    def test_cleanup_resources(self, mock_thread, mock_pipe, mock_process):
        """Test that cleanup properly cleans up all resources."""
        # Setup mocks
        mock_listener_conn = Mock()
        mock_main_conn = Mock()
        mock_pipe.return_value = (mock_listener_conn, mock_main_conn)
        
        mock_process_instance = Mock()
        mock_process.return_value = mock_process_instance
        mock_process_instance.is_alive.return_value = True
        
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        mock_thread_instance.is_alive.return_value = True
        
        threadlet = Threadlet(
            process_id=self.process_id,
            device_uuid=self.device_uuid,
            torchLoom_addr=self.torchLoom_addr
        )
        
        # Start the threadlet to initialize resources
        threadlet.start()
        
        # Call cleanup
        threadlet._cleanup()
        
        # Verify stop event was set
        assert threadlet._stop_event.is_set()
        
        # Verify pipe listener thread was joined and stopped
        threadlet._pipe_listener_stop_event.is_set()
        mock_thread_instance.join.assert_called()
        
        # Verify process was joined
        mock_process_instance.join.assert_called()
        
        # Verify pipe connections were closed
        mock_main_conn.close.assert_called_once()

    @patch('torchLoom.threadlet.threadlet.multiprocessing.Process')
    @patch('torchLoom.threadlet.threadlet.multiprocessing.Pipe')
    @patch('torchLoom.threadlet.threadlet.threading.Thread')
    def test_cleanup_with_terminated_process(self, mock_thread, mock_pipe, mock_process):
        """Test that cleanup handles already terminated processes gracefully."""
        # Setup mocks
        mock_listener_conn = Mock()
        mock_main_conn = Mock()
        mock_pipe.return_value = (mock_listener_conn, mock_main_conn)
        
        mock_process_instance = Mock()
        mock_process.return_value = mock_process_instance
        # Three is_alive calls: 1) before setting stop event, 2) before join, 3) after join to check terminate
        # Return True for all to force terminate to be called
        mock_process_instance.is_alive.side_effect = [True, True, True]  
        
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        mock_thread_instance.is_alive.return_value = False
        
        threadlet = Threadlet(
            process_id=self.process_id,
            device_uuid=self.device_uuid,
            torchLoom_addr=self.torchLoom_addr
        )
        
        # Start and cleanup
        threadlet.start()
        threadlet._cleanup()
        
        # Verify terminate was called when process didn't stop gracefully after join
        mock_process_instance.terminate.assert_called_once()

    def test_publish_methods_send_to_pipe(self):
        """Test that publish methods send messages to the listener via pipe."""
        threadlet = Threadlet(
            process_id=self.process_id,
            device_uuid=self.device_uuid,
            torchLoom_addr=self.torchLoom_addr
        )
        
        # Mock the internal message sender
        with patch.object(threadlet, '_send_internal_message_to_listener') as mock_sender:
            # Test publish_heartbeat
            threadlet.publish_heartbeat(status="active", metadata={"test": "data"})
            
            expected_payload = {
                "action": "publish_event",
                "event_type": "heartbeat",
                "event_data": {
                    "process_id": self.process_id,
                    "device_uuid": self.device_uuid,
                    "status": "active",
                    "metadata": {"test": "data"}
                }
            }
            mock_sender.assert_called_with(expected_payload)
            
            # Test publish_training_status
            threadlet.publish_training_status(status_data={"step": 100})
            
            expected_payload = {
                "action": "publish_event",
                "event_type": "training_status",
                "event_data": {
                    "process_id": self.process_id,
                    "device_uuid": self.device_uuid,
                    "status_data": {"step": 100}
                }
            }
            mock_sender.assert_called_with(expected_payload)

    def test_register_handler(self):
        """Test that handlers are properly registered."""
        threadlet = Threadlet(
            process_id=self.process_id,
            device_uuid=self.device_uuid,
            torchLoom_addr=self.torchLoom_addr
        )
        
        # Create a test handler
        test_handler = Mock()
        
        # Register the handler
        threadlet.register_handler('test_param', test_handler)
        
        # Verify it was registered
        assert threadlet._handler_registry.has_handler('test_param')
        retrieved_handler = threadlet._handler_registry.get_handler('test_param')
        assert retrieved_handler == test_handler

    def test_send_internal_message_to_listener(self):
        """Test that internal messages are sent to listener correctly."""
        threadlet = Threadlet(
            process_id=self.process_id,
            device_uuid=self.device_uuid,
            torchLoom_addr=self.torchLoom_addr
        )
        
        # Setup mock pipe connection
        mock_main_conn = Mock()
        mock_main_conn.closed = False  # Pipe is not closed
        threadlet._main_pipe_conn = mock_main_conn
        
        # Test sending a message
        test_payload = {"action": "test", "data": "test_data"}
        threadlet._send_internal_message_to_listener(test_payload)
        
        # Verify message was sent via pipe
        mock_main_conn.send.assert_called_once_with(test_payload)

    def test_send_internal_message_broken_pipe(self):
        """Test that broken pipe errors are handled gracefully."""
        threadlet = Threadlet(
            process_id=self.process_id,
            device_uuid=self.device_uuid,
            torchLoom_addr=self.torchLoom_addr
        )
        
        # Setup mock pipe connection that raises BrokenPipeError
        mock_main_conn = Mock()
        mock_main_conn.closed = False
        mock_main_conn.send.side_effect = BrokenPipeError("Pipe broken")
        threadlet._main_pipe_conn = mock_main_conn
        
        # This should not raise an exception
        test_payload = {"action": "test", "data": "test_data"}
        threadlet._send_internal_message_to_listener(test_payload)
