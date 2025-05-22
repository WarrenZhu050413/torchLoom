import pytest
import sys
import io
from unittest.mock import patch, MagicMock, AsyncMock
import cmd
import asyncio

# Import the CLI class (assuming it's available as shown in the monitor_cli.py file)
from torchLoom.monitor_cli import MyShell

@pytest.fixture
def cli():
    """Create a CLI instance for testing."""
    return MyShell()

class TestMonitorCLI:
    def test_cli_initialization(self, cli):
        """Test that the CLI initializes with the correct attributes."""
        assert isinstance(cli, cmd.Cmd)
        assert cli.prompt == "torchLoom> "
        assert cli.intro.startswith("Welcome to torchLoom")
    
    def test_do_exit(self, cli):
        """Test that the exit command returns True to exit the CLI."""
        assert cli.do_exit("") is True
    
    def test_do_quit(self, cli):
        """Test that the quit command returns True to exit the CLI."""
        assert cli.do_quit("") is True
    
    def test_help_commands(self, cli):
        """Test that help text is available for all commands."""
        # Get the docstring directly to verify it contains what we expect
        assert "Exit the CLI" in cli.do_exit.__doc__
        
        # For quit, we can check if its docstring exists
        assert cli.do_quit.__doc__ is not None
    
    @patch('torchLoom.monitor_cli.nats.connect')
    def test_do_register_device(self, mock_nats_connect, cli):
        """Test the register_device command."""
        # Setup mocks
        mock_nc = AsyncMock()
        mock_nats_connect.return_value = mock_nc
        
        # Override asyncio.run to execute the coroutine directly
        with patch('asyncio.run') as mock_run:
            def side_effect(coro):
                # Create a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Run the coroutine in the loop
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
            
            mock_run.side_effect = side_effect
            
            # Call the command
            cli.do_register_device("device123 replica456")
            
            # Check that publish was called with the right arguments
            mock_nc.publish.assert_called_once()
            
            # First argument to publish should be the subject
            subject = mock_nc.publish.call_args[0][0]
            assert "torchLoom" in subject
            
            # Second argument should be the serialized event
            data = mock_nc.publish.call_args[0][1]
            assert isinstance(data, bytes)
    
    @patch('torchLoom.monitor_cli.nats.connect')
    def test_do_fail_device(self, mock_nats_connect, cli):
        """Test the fail_device command."""
        # Setup mocks
        mock_nc = AsyncMock()
        mock_nats_connect.return_value = mock_nc
        
        # Override asyncio.run to execute the coroutine directly
        with patch('asyncio.run') as mock_run:
            def side_effect(coro):
                # Create a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Run the coroutine in the loop
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
            
            mock_run.side_effect = side_effect
            
            # Call the command
            cli.do_fail_device("device123")
            
            # Check that publish was called with the right arguments
            mock_nc.publish.assert_called_once()
            
            # First argument to publish should be the subject
            subject = mock_nc.publish.call_args[0][0]
            assert "torchLoom" in subject
            
            # Second argument should be the serialized event
            data = mock_nc.publish.call_args[0][1]
            assert isinstance(data, bytes)
    
    @patch('torchLoom.monitor_cli.nats.connect')
    def test_do_reset_lr(self, mock_nats_connect, cli):
        """Test the reset_lr command."""
        # Setup mocks
        mock_nc = AsyncMock()
        mock_nats_connect.return_value = mock_nc
        
        # Override asyncio.run to execute the coroutine directly
        with patch('asyncio.run') as mock_run:
            def side_effect(coro):
                # Create a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Run the coroutine in the loop
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
            
            mock_run.side_effect = side_effect
            
            # Call the command
            cli.do_reset_lr("0.001")
            
            # Check that publish was called with the right arguments
            mock_nc.publish.assert_called_once()
            
            # First argument to publish should be the subject
            subject = mock_nc.publish.call_args[0][0]
            assert "torchLoom" in subject
            
            # Second argument should be the serialized event
            data = mock_nc.publish.call_args[0][1]
            assert isinstance(data, bytes)
    
    def test_emptyline(self, cli):
        """Test that empty line does nothing."""
        assert cli.emptyline() is None 
 