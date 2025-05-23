import asyncio
import multiprocessing as mp

import nats
from nats.js.api import StreamConfig

from torchLoom.constants import torchLoomConstants
from torchLoom.torchLoom_pb2 import EventEnvelope


async def _listen_for_config(queue: mp.Queue, addr: str) -> None:
    nc = await nats.connect(addr)
    js = nc.jetstream()
    # ensure stream exists for config updates
    await js.add_stream(StreamConfig(name="WEAVELET_STREAM", subjects=[torchLoomConstants.subjects.CONFIG_INFO]))
    sub = await js.pull_subscribe(
        torchLoomConstants.subjects.CONFIG_INFO,
        durable="weavelet",
        stream="WEAVELET_STREAM",
    )
    while True:
        try:
            msgs = await sub.fetch(1, timeout=1)
        except Exception:
            continue
        for msg in msgs:
            env = EventEnvelope()
            env.ParseFromString(msg.data)
            if env.HasField("config_info"):
                params = env.config_info.config_params
                if "optimizer_type" in params:
                    queue.put(params["optimizer_type"])
            await msg.ack()


def weavelet_process(queue: mp.Queue, addr: str = torchLoomConstants.DEFAULT_ADDR) -> None:
    asyncio.run(_listen_for_config(queue, addr))

