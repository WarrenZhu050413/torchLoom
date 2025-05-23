import asyncio
import pytest
from torchLoom.utils import cancel_subscriptions


class DummySub:
    def __init__(self):
        self.unsub_called = False

    async def unsubscribe(self):
        self.unsub_called = True


@pytest.mark.asyncio
async def test_cancel_subscriptions():
    async def dummy_task():
        await asyncio.sleep(0)

    sub = DummySub()
    task = asyncio.create_task(dummy_task())
    subscriptions = {"a": (sub, task)}
    await cancel_subscriptions(subscriptions)
    assert task.cancelled()
    assert sub.unsub_called
