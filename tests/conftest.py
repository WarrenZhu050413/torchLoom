import pytest
import pytest_asyncio
import asyncio
import os
import sys
from unittest.mock import MagicMock, AsyncMock, patch
import nats
from nats.aio.client import Client
from nats.js.client import JetStreamContext

# Add the parent directory to sys.path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchLoom.config import Config

# Override log file paths for testing to avoid writing to real log files
Config.MANAGER_torchLoom_LOG_FILE = "/tmp/test_torchLoom.log"
Config.MANAGER_RUNTIME_LOG_FILE = "/tmp/test_manager.log"
Config.torchLoom_CONSTANTS_LOG_FILE = "/tmp/test_torchLoom_constants.log"
Config.torchLoom_UTILS_LOG_FILE = "/tmp/test_torchLoom_utils.log"
Config.torchLoom_CONTROLLER_LOG_FILE = "/tmp/test_torchLoom_weaver.log"
Config.torchLoom_MONITOR_CLI_LOG_FILE = "/tmp/test_torchLoom_monitor_cli.log"

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture
def mock_nats_client():
    """Create a mock NATS client for testing."""
    mock_client = AsyncMock(spec=Client)
    mock_client.is_closed = False
    mock_client.jetstream.return_value = AsyncMock(spec=JetStreamContext)
    return mock_client

@pytest.fixture
def mock_jetstream():
    """Create a mock JetStream for testing."""
    mock_js = AsyncMock(spec=JetStreamContext)
    return mock_js

@pytest.fixture
def mock_nats_msg():
    """Create a mock NATS message for testing."""
    mock_msg = AsyncMock()
    mock_msg.subject = "test.subject"
    mock_msg.data = b""  # Will be overridden in tests that need specific data
    return mock_msg

@pytest.fixture
def monkeypatch_nats_connect(monkeypatch, mock_nats_client):
    """Patch nats.connect to return a mock client."""
    async def mock_connect(*args, **kwargs):
        return mock_nats_client
    
    monkeypatch.setattr(nats, "connect", mock_connect)
    return mock_nats_client

@pytest.fixture
def weaver(event_loop):
    """Create a Weaver instance for testing with the event loop properly set."""
    from torchLoom.weaver import Weaver
    
    # Create a patch for the Weaver.__init__ method to avoid getting an event loop
    with patch('asyncio.get_event_loop', return_value=event_loop):
        wvr = Weaver()
    
    return wvr

@pytest_asyncio.fixture
async def initialized_weaver(weaver, mock_nats_client):
    """Create an initialized Weaver instance for testing."""
    with patch("nats.connect", return_value=mock_nats_client):
        await weaver.initialize()
    weaver._nc = mock_nats_client
    weaver._js = mock_nats_client.jetstream()
    return weaver 