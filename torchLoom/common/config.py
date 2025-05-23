class Config:
    DEFAULT_ADDR: str = "nats://localhost:4222"
    MANAGER_torchLoom_LOG_FILE: str = "./torchLoom/log/torchLoom.log"
    MANAGER_RUNTIME_LOG_FILE: str = "./torchLoom/log/manager.log"
    torchLoom_CONSTANTS_LOG_FILE: str = "./torchLoom/log/torchLoom_constants.log"
    torchLoom_UTILS_LOG_FILE: str = "./torchLoom/log/torchLoom_utils.log"
    torchLoom_CONTROLLER_LOG_FILE: str = "./torchLoom/log/torchLoom_weaver.log"
    torchLoom_MONITOR_CLI_LOG_FILE: str = "./torchLoom/log/torchLoom_monitor_cli.log"
    FORMAT_LOG: bool = False
    NC_TIMEOUT: float = 1
    EXCEPTION_RETRY_TIME: float = 1
    CONNECTION_RETRY_TIME: float = 1