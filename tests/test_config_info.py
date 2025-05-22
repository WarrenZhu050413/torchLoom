import asyncio
import unittest
import time
from unittest.mock import patch, MagicMock, AsyncMock

import nats
from nats.js.client import JetStreamContext

from torchLoom.weaver import Weaver
from torchLoom.torchLoom_pb2 import EventEnvelope, ChangeConfigEvent
from torchLoom.constants import torchLoomConstants

class TestConfigInfo(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Mock NATS and JetStream
        self.mock_nc = MagicMock()
        self.mock_js = AsyncMock()
        
        # Create a weaver instance with mocked dependencies
        self.weaver = Weaver()
        self.weaver._nc = self.mock_nc
        self.weaver._js = self.mock_js
        self.mock_nc.jetstream.return_value = self.mock_js
        
    async def test_config_info_event_handling(self):
        # Create a config_info event
        env = EventEnvelope()
        env.config_info.config_params["learning_rate"] = "0.001"
        env.config_info.config_params["batch_size"] = "64"
        
        # Call the message handler
        await self.weaver.message_handler_config_info(env)
        
        # Verify that publish was called for learning rate
        self.mock_js.publish.assert_any_call(
            "torchLoom.training.reset_lr", 
            "0.001".encode("utf-8")
        )
        
        # Verify that publish was called for the entire config
        self.mock_js.publish.assert_any_call(
            torchLoomConstants.subjects.CONFIG_INFO, 
            env.SerializeToString()
        )
        
    async def test_config_info_without_learning_rate(self):
        # Create a config_info event without learning_rate
        env = EventEnvelope()
        env.config_info.config_params["batch_size"] = "64"
        env.config_info.config_params["num_workers"] = "4"
        
        # Call the message handler
        await self.weaver.message_handler_config_info(env)
        
        # Verify that publish was called only for the config subject
        self.mock_js.publish.assert_called_once_with(
            torchLoomConstants.subjects.CONFIG_INFO, 
            env.SerializeToString()
        )

if __name__ == '__main__':
    unittest.main() 