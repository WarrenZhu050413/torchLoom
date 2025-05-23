from torchLoom.config import Config


def test_default_addr():
    assert Config.DEFAULT_ADDR == "nats://0.0.0.0:4222"


def test_log_files_exist():
    assert Config.MANAGER_torchLoom_LOG_FILE.endswith(".log")
    assert Config.torchLoom_CONTROLLER_LOG_FILE.endswith(".log")
