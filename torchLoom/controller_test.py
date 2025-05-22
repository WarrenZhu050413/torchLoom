import asyncio
import pytest
from torchLoom.controller import Weaver
from torchLoom.torchLoom_pb2 import EventEnvelope

@pytest.mark.asyncio
async def test_register_device_mapping():
    weaver = Weaver()
    env = EventEnvelope()
    env.register_device.device_uuid = "gpu0"
    env.register_device.replica_id = "replica0"
    await weaver.message_handler_register_device(env)
    assert weaver.get_replicas_for_device("gpu0") == {"replica0"}
    assert weaver.get_devices_for_replica("replica0") == {"gpu0"}

@pytest.mark.asyncio
async def test_maybe_log_and_raise_exception():
    weaver = Weaver()
    with pytest.raises(Exception):
        weaver.maybe_log_and_raise_exception("nc")
    with pytest.raises(Exception):
        weaver.maybe_log_and_raise_exception("js")
