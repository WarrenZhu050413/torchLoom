from torchLoom.constants import torchLoomConstants


def test_default_addr():
    assert torchLoomConstants.DEFAULT_ADDR.startswith("nats://")


def test_weaver_stream_name():
    assert torchLoomConstants.weaver_stream.STREAM == "CONTROLLER-STREAM"
