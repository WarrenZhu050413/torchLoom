"""
Unit tests for the Weaver class focusing on WebSocket server integration.

This module tests that the Weaver properly initializes, configures, and uses
the WebSocket server for UI communication.
"""

import asyncio
import json
import pytest
import pytest_asyncio
import time
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock, patch, AsyncMock, call

from torchLoom.weaver.weaver import Weaver
from torchLoom.proto.torchLoom_pb2 import EventEnvelope, TrainingStatus, deviceStatus
from torchLoom.common.constants import UINetworkConstants


class TestWeaverWebSocketIntegration:
    """Test Weaver's WebSocket server integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.default_host = "127.0.0.1"
        self.default_port = 8765
        self.torchLoom_addr = "nats://localhost:4222"
        
    @pytest_asyncio.fixture
    async def mock_subscription_manager(self):
        """Create a mock subscription manager."""
        mock_manager = AsyncMock()
        mock_manager.nc = AsyncMock()
        mock_manager.js = AsyncMock()
        mock_manager.stream_manager = AsyncMock()
        return mock_manager
    
    @pytest_asyncio.fixture
    async def weaver_with_ui(self, mock_subscription_manager):
        """Create a Weaver instance with UI enabled and mocked dependencies."""
        with patch('torchLoom.weaver.weaver.SubscriptionManager') as mock_sub_mgr:
            mock_sub_mgr.return_value = mock_subscription_manager
            
            with patch('torchLoom.weaver.weaver.ThreadletCommandPublisher') as mock_cmd_pub:
                mock_cmd_pub.return_value = AsyncMock()
                
                with patch('torchLoom.weaver.weaver.WebSocketServer') as mock_ws_server:
                    mock_ws_instance = Mock()
                    mock_ws_instance.send_to_all = AsyncMock()
                    mock_ws_instance.set_ui_command_handler = Mock()
                    mock_ws_instance.set_initial_status_provider = Mock()
                    mock_ws_instance.run_server = AsyncMock()
                    mock_ws_server.return_value = mock_ws_instance
                    
                    weaver = Weaver(
                        torchLoom_addr=self.torchLoom_addr,
                        enable_ui=True,
                        ui_host=self.default_host,
                        ui_port=self.default_port
                    )
                    weaver._subscription_manager = mock_subscription_manager
                    
                    await weaver.initialize()
                    return weaver
    
    @pytest_asyncio.fixture
    async def weaver_without_ui(self, mock_subscription_manager):
        """Create a Weaver instance with UI disabled."""
        with patch('torchLoom.weaver.weaver.SubscriptionManager') as mock_sub_mgr:
            mock_sub_mgr.return_value = mock_subscription_manager
            
            with patch('torchLoom.weaver.weaver.ThreadletCommandPublisher') as mock_cmd_pub:
                mock_cmd_pub.return_value = AsyncMock()
                
                weaver = Weaver(
                    torchLoom_addr=self.torchLoom_addr,
                    enable_ui=False
                )
                weaver._subscription_manager = mock_subscription_manager
                
                await weaver.initialize()
                return weaver
    
    def test_weaver_initialization_with_ui_enabled(self):
        """Test that Weaver initializes correctly with UI enabled."""
        weaver = Weaver(
            enable_ui=True,
            ui_host=self.default_host,
            ui_port=self.default_port
        )
        
        assert weaver.enable_ui is True
        assert weaver.ui_host == self.default_host
        assert weaver.ui_port == self.default_port
        assert weaver.websocket_server is None  # Not initialized until initialize() is called
        assert weaver.ui_notification_manager is not None
    
    def test_weaver_initialization_with_ui_disabled(self):
        """Test that Weaver initializes correctly with UI disabled."""
        weaver = Weaver(enable_ui=False)
        
        assert weaver.enable_ui is False
        assert weaver.ui_host == UINetworkConstants.DEFAULT_UI_HOST
        assert weaver.ui_port == UINetworkConstants.DEFAULT_UI_PORT
        assert weaver.websocket_server is None
        assert weaver.ui_notification_manager is not None
    
    @pytest.mark.asyncio
    async def test_weaver_websocket_server_setup_during_initialization(self, mock_subscription_manager):
        """Test that WebSocket server is properly set up during weaver initialization."""
        with patch('torchLoom.weaver.weaver.SubscriptionManager') as mock_sub_mgr:
            mock_sub_mgr.return_value = mock_subscription_manager
            
            with patch('torchLoom.weaver.weaver.ThreadletCommandPublisher') as mock_cmd_pub:
                mock_cmd_pub.return_value = AsyncMock()
                
                with patch('torchLoom.weaver.weaver.WebSocketServer') as mock_ws_server:
                    mock_ws_instance = Mock()
                    mock_ws_instance.send_to_all = AsyncMock()
                    mock_ws_instance.set_ui_command_handler = Mock()
                    mock_ws_instance.set_initial_status_provider = Mock()
                    mock_ws_server.return_value = mock_ws_instance
                    
                    weaver = Weaver(enable_ui=True, ui_host="localhost", ui_port=9999)
                    weaver._subscription_manager = mock_subscription_manager
                    
                    await weaver.initialize()
                    
                    # Verify WebSocket server was created with correct parameters
                    mock_ws_server.assert_called_once_with(host="localhost", port=9999)
                    
                    # Verify websocket server callbacks were set
                    mock_ws_instance.set_ui_command_handler.assert_called_once()
                    mock_ws_instance.set_initial_status_provider.assert_called_once()
                    
                    # Verify UI notification manager was connected to websocket
                    assert weaver.ui_notification_manager._websocket_send_func is not None
                    
                    # Verify status tracker is connected to UI manager
                    assert weaver.status_tracker._ui_notification_callback is not None
    
    @pytest.mark.asyncio
    async def test_weaver_no_websocket_setup_when_ui_disabled(self, mock_subscription_manager):
        """Test that WebSocket server is not set up when UI is disabled."""
        with patch('torchLoom.weaver.weaver.SubscriptionManager') as mock_sub_mgr:
            mock_sub_mgr.return_value = mock_subscription_manager
            
            with patch('torchLoom.weaver.weaver.ThreadletCommandPublisher') as mock_cmd_pub:
                mock_cmd_pub.return_value = AsyncMock()
                
                with patch('torchLoom.weaver.weaver.WebSocketServer') as mock_ws_server:
                    weaver = Weaver(enable_ui=False)
                    weaver._subscription_manager = mock_subscription_manager
                    
                    await weaver.initialize()
                    
                    # Verify WebSocket server was NOT created
                    mock_ws_server.assert_not_called()
                    assert weaver.websocket_server is None
    
    @pytest.mark.asyncio
    async def test_ui_websocket_command_handling(self, weaver_with_ui):
        """Test that weaver properly handles UI commands received via WebSocket."""
        # Mock the handler registry to track handler calls
        mock_handler = AsyncMock()
        weaver_with_ui._handler_registry.register_handler("ui_command", mock_handler)
        
        # Simulate receiving a UI command via WebSocket
        websocket_data = {
            "type": "ui_command",
            "data": {
                "command_type": "pause_training",
                "process_id": "test-process-123",
                "params": {
                    "immediate": "true"
                }
            }
        }
        
        # Call the websocket command handler
        await weaver_with_ui._handle_ui_websocket_command(websocket_data)
        
        # Verify the handler was called with correct protobuf envelope
        mock_handler.assert_called_once()
        call_args = mock_handler.call_args[1]
        
        assert "env" in call_args
        env = call_args["env"]
        assert env.WhichOneof("body") == "ui_command"
        assert env.ui_command.command_type == "pause_training"
        assert env.ui_command.process_id == "test-process-123"
        assert env.ui_command.params["immediate"] == "true"
    
    @pytest.mark.asyncio
    async def test_unknown_websocket_command_handling(self, weaver_with_ui):
        """Test that weaver handles unknown WebSocket command types gracefully."""
        # Mock the handler registry to verify no handler is called
        mock_handler = AsyncMock()
        weaver_with_ui._handler_registry.register_handler("ui_command", mock_handler)
        
        # Simulate receiving an unknown command via WebSocket
        websocket_data = {
            "type": "unknown_command",
            "data": {"some": "data"}
        }
        
        # Call the websocket command handler - should not raise exception
        await weaver_with_ui._handle_ui_websocket_command(websocket_data)
        
        # Verify no handler was called for unknown command
        mock_handler.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_websocket_command_handler_error_handling(self, weaver_with_ui):
        """Test that weaver handles errors in websocket command processing gracefully."""
        # Mock the handler to raise an exception
        mock_handler = AsyncMock(side_effect=Exception("Handler error"))
        weaver_with_ui._handler_registry.register_handler("ui_command", mock_handler)
        
        websocket_data = {
            "type": "ui_command",
            "data": {
                "command_type": "test_command",
                "process_id": "test-process",
                "params": {}
            }
        }
        
        # Should not raise exception despite handler error
        await weaver_with_ui._handle_ui_websocket_command(websocket_data)
        
        # Verify handler was called
        mock_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_ui_server_with_websocket_enabled(self, weaver_with_ui):
        """Test that start_ui_server properly starts the WebSocket server."""
        # Mock the websocket server run_server method
        weaver_with_ui.websocket_server.run_server = AsyncMock()
        
        # Mock the UI notification manager start_broadcaster_task
        weaver_with_ui.ui_notification_manager.start_broadcaster_task = Mock()
        
        # Start the UI server
        await weaver_with_ui.start_ui_server()
        
        # Verify broadcaster task was started
        weaver_with_ui.ui_notification_manager.start_broadcaster_task.assert_called_once()
        
        # Verify websocket server was started
        weaver_with_ui.websocket_server.run_server.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_ui_server_without_websocket(self, weaver_without_ui):
        """Test that start_ui_server handles case where WebSocket server is not initialized."""
        # Should not raise exception when websocket_server is None
        await weaver_without_ui.start_ui_server()
        
        # Verify no errors occurred
        assert weaver_without_ui.websocket_server is None
    
    @pytest.mark.asyncio
    async def test_status_change_notification_triggers_websocket_broadcast(self, weaver_with_ui):
        """Test that status changes trigger WebSocket broadcasts to UI clients."""
        # Mock the websocket send function to track calls
        mock_send_func = AsyncMock()
        weaver_with_ui.websocket_server.send_to_all = mock_send_func
        weaver_with_ui.ui_notification_manager.set_websocket_send_func(mock_send_func)
        
        # Trigger a status change
        weaver_with_ui.status_tracker.update_training_progress(
            process_id="test-process",
            current_step=100,
            epoch=5
        )
        
        # Give time for async broadcast
        await asyncio.sleep(0.1)
        
        # Verify websocket broadcast was triggered
        mock_send_func.assert_called()
        
        # Verify the data format
        call_args = mock_send_func.call_args[0][0]
        assert call_args["type"] == "status_update"
        assert "data" in call_args
        assert "training_status" in call_args["data"]
        assert "devices" in call_args["data"]
        assert "timestamp" in call_args["data"]
    
    @pytest.mark.asyncio
    async def test_device_status_update_triggers_websocket_broadcast(self, weaver_with_ui):
        """Test that device status updates trigger WebSocket broadcasts."""
        mock_send_func = AsyncMock()
        weaver_with_ui.websocket_server.send_to_all = mock_send_func
        weaver_with_ui.ui_notification_manager.set_websocket_send_func(mock_send_func)
        
        # Create a device status and update it
        device_status = deviceStatus()
        device_status.device_uuid = "test-device-123"
        device_status.process_id = "test-process-456"
        device_status.utilization = 85.5
        device_status.temperature = 72.0
        
        weaver_with_ui.status_tracker.update_device_status_from_proto(device_status)
        
        # Give time for async broadcast
        await asyncio.sleep(0.1)
        
        # Verify websocket broadcast was triggered
        mock_send_func.assert_called()
        
        # Verify device data is included
        call_args = mock_send_func.call_args[0][0]
        assert call_args["type"] == "status_update"
        device_data = call_args["data"]["devices"]
        assert len(device_data) > 0
        assert device_data[0]["device_uuid"] == "test-device-123"
        assert device_data[0]["utilization"] == 85.5
    
    @pytest.mark.asyncio
    async def test_websocket_server_callbacks_are_set_correctly(self, weaver_with_ui):
        """Test that WebSocket server callbacks are properly configured."""
        # Verify the UI command handler callback is the weaver's method
        weaver_with_ui.websocket_server.set_ui_command_handler.assert_called_once_with(
            weaver_with_ui._handle_ui_websocket_command
        )
        
        # Verify the initial status provider callback is set
        weaver_with_ui.websocket_server.set_initial_status_provider.assert_called_once()
        
        # Get the actual callback that was set
        call_args = weaver_with_ui.websocket_server.set_initial_status_provider.call_args[0]
        initial_status_callback = call_args[0]
        
        # Verify the callback is the UI notification manager's method
        assert initial_status_callback == weaver_with_ui.ui_notification_manager.get_status_data_for_initial_connection
    
    @pytest.mark.asyncio
    async def test_weaver_stop_cleans_up_ui_components(self, weaver_with_ui):
        """Test that stopping the weaver properly cleans up UI components."""
        # Mock the UI notification manager stop method
        weaver_with_ui.ui_notification_manager.stop_broadcaster = AsyncMock()
        
        # Stop the weaver
        await weaver_with_ui.stop()
        
        # Verify UI notification manager was stopped
        weaver_with_ui.ui_notification_manager.stop_broadcaster.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_websocket_broadcast_formatting(self, weaver_with_ui):
        """Test that WebSocket broadcasts have correct data formatting."""
        mock_send_func = AsyncMock()
        weaver_with_ui.ui_notification_manager.set_websocket_send_func(mock_send_func)
        
        # Add some test data
        weaver_with_ui.status_tracker.update_training_progress(
            process_id="proc-1",
            current_step=50,
            epoch=2,
            metrics={"loss": "0.25", "accuracy": "0.95"}
        )
        
        device_status = deviceStatus()
        device_status.device_uuid = "device-1"
        device_status.process_id = "proc-1"
        device_status.utilization = 75.0
        device_status.memory_used = 8192
        device_status.memory_total = 16384
        
        weaver_with_ui.status_tracker.update_device_status_from_proto(device_status)
        
        # Give time for async broadcast
        await asyncio.sleep(0.1)
        
        # Verify the broadcast data structure
        mock_send_func.assert_called()
        broadcast_data = mock_send_func.call_args[0][0]
        
        assert broadcast_data["type"] == "status_update"
        data = broadcast_data["data"]
        
        # Check training status formatting
        assert len(data["training_status"]) == 1
        training = data["training_status"][0]
        assert training["process_id"] == "proc-1"
        assert training["current_step"] == 50
        assert training["epoch"] == 2
        assert training["metrics"]["loss"] == "0.25"
        assert training["metrics"]["accuracy"] == "0.95"
        
        # Check device status formatting
        assert len(data["devices"]) == 1
        device = data["devices"][0]
        assert device["device_uuid"] == "device-1"
        assert device["process_id"] == "proc-1"
        assert device["utilization"] == 75.0
        assert device["memory_used"] == 8192
        assert device["memory_total"] == 16384
        
        # Check timestamp
        assert isinstance(data["timestamp"], int)
        assert data["timestamp"] <= int(time.time())
    
    @pytest.mark.asyncio
    async def test_multiple_websocket_commands_processed_correctly(self, weaver_with_ui):
        """Test that multiple WebSocket commands are processed correctly in sequence."""
        processed_commands = []
        
        async def mock_handler(env, status_tracker, heartbeat_tracker, weaver_publish_command_func):
            processed_commands.append({
                "command_type": env.ui_command.command_type,
                "process_id": env.ui_command.process_id,
                "params": dict(env.ui_command.params)
            })
        
        weaver_with_ui._handler_registry.register_handler("ui_command", mock_handler)
        
        # Send multiple commands
        commands = [
            {
                "type": "ui_command",
                "data": {
                    "command_type": "pause_training",
                    "process_id": "proc-1",
                    "params": {"immediate": "true"}
                }
            },
            {
                "type": "ui_command", 
                "data": {
                    "command_type": "resume_training",
                    "process_id": "proc-2",
                    "params": {"delay": "5"}
                }
            },
            {
                "type": "ui_command",
                "data": {
                    "command_type": "update_config",
                    "process_id": "proc-3",
                    "params": {"learning_rate": "0.001", "batch_size": "64"}
                }
            }
        ]
        
        # Process each command
        for cmd in commands:
            await weaver_with_ui._handle_ui_websocket_command(cmd)
        
        # Verify all commands were processed
        assert len(processed_commands) == 3
        
        assert processed_commands[0]["command_type"] == "pause_training"
        assert processed_commands[0]["process_id"] == "proc-1"
        assert processed_commands[0]["params"]["immediate"] == "true"
        
        assert processed_commands[1]["command_type"] == "resume_training"
        assert processed_commands[1]["process_id"] == "proc-2"
        assert processed_commands[1]["params"]["delay"] == "5"
        
        assert processed_commands[2]["command_type"] == "update_config"
        assert processed_commands[2]["process_id"] == "proc-3"
        assert processed_commands[2]["params"]["learning_rate"] == "0.001"
        assert processed_commands[2]["params"]["batch_size"] == "64" 