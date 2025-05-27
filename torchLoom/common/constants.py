from dataclasses import dataclass

from torchLoom.log.logger import setup_logger

class LoggerConstants:
    MANAGER_torchLoom_LOG_FILE: str = "./torchLoom/log/torchLoom.log"
    MANAGER_RUNTIME_LOG_FILE: str = "./torchLoom/log/manager.log"
    torchLoom_CONSTANTS_LOG_FILE: str = "./torchLoom/log/torchLoom_constants.log"
    torchLoom_UTILS_LOG_FILE: str = "./torchLoom/log/torchLoom_utils.log"
    torchLoom_CONTROLLER_LOG_FILE: str = "./torchLoom/log/torchLoom_weaver.log"
    torchLoom_MONITOR_CLI_LOG_FILE: str = "./torchLoom/log/torchLoom_monitor_cli.log"
    FORMAT_LOG: bool = False
    
logger = setup_logger(
    name="torchLoom_constants", log_file=LoggerConstants.torchLoom_CONSTANTS_LOG_FILE
)

# ===========================================
# HANDLER CONSTANTS AND CONFIGURATIONS
# ===========================================

class HandlerConstants:
    """Constants related to handler configurations."""

    # Threadlet handler event types
    THREADLET_EVENTS = [
        "register_device",
        "heartbeat",
        "training_status",
        "device_status",
    ]

    # External handler event types
    EXTERNAL_EVENTS = ["monitored_fail"]

    # UI handler event types (merged into single handler)
    UI_EVENTS = ["ui_command"]

# ===========================================
# TIME AND NETWORK CONSTANTS
# ===========================================

class TimeConstants:
    """All timing-related constants for torchLoom."""

    # Broadcast and monitoring intervals
    STATUS_BROADCAST_IN: float = 1.0
    HEARTBEAT_MONITOR_INTERVAL: float = 30.0
    HEARTBEAT_SEND_INTERVAL: float = 3.0
    HEARTBEAT_TIMEOUT: float = 10.0

    # Sleep intervals for various operations
    PIPE_POLL_INTERVAL: float = 0.1
    ERROR_RETRY_SLEEP: float = 5.0
    CLEANUP_SLEEP: float = 1.0

    # Process management timeouts
    PIPE_LISTENER_TIMEOUT: float = 1.0
    PIPE_LISTENER_STOP_TIMEOUT: float = 2.0
    THREADLET_PROCESS_TIMEOUT: float = 5.0
    THREADLET_PROCESS_TERMINATE_TIMEOUT: float = 2.0
    ASYNC_PIPE_POLL_INTERVAL = 0.1

    # Async operation timeouts
    MONITOR_STOP_EVENT_SLEEP: float = 0.1
    BRIEF_PAUSE: float = 0.1
   
    # NATS connection constants 
    NC_TIMEOUT: float = 1

class UINetworkConstants:
    """Network-related constants for torchLoom."""

    DEFAULT_UI_HOST: str = "0.0.0.0"
    DEFAULT_UI_PORT: int = 8080

    # CORS origins for WebSocket connections
    CORS_ORIGINS = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]

# ===========================================
# NATS SUBJECTS AND STREAMS
# ===========================================

class torchLoomSubjects:
    WEAVER_COMMANDS: str = "torchLoom.weaver.commands" # Weaver -> Training Process subjects
    THREADLET_EVENTS: str = "torchLoom.threadlet.events" # All events from threadlets
    EXTERNAL_EVENTS: str = "torchLoom.external.events" # All events from external systems

class StreamSpec:
    STREAM = None
    CONSUMER = None
    subjects = None

# (Weaver -> Threadlets)
class WeaverOutgressStream(StreamSpec):
    STREAM: str = "WEAVER_COMMANDS_STREAM"
    CONSUMER: str = "weaver-commands-consumer"
    subjects = [torchLoomSubjects.WEAVER_COMMANDS]

# (Threadlets/External -> Weaver)
class WeaverIngressStream(StreamSpec):
    STREAM: str = "WEAVER_INGRESS_STREAM"
    CONSUMER: str = "weaver-ingress-consumer"
    subjects = [
        torchLoomSubjects.THREADLET_EVENTS,
        torchLoomSubjects.EXTERNAL_EVENTS,
    ]


@dataclass
class NatsConstantsClass:
    """Constants for torchLoom communication channels and configuration."""
    subjects: torchLoomSubjects = torchLoomSubjects()
    weaver_stream: WeaverOutgressStream = WeaverOutgressStream()
    weaver_ingress_stream: WeaverIngressStream = WeaverIngressStream()
    DEFAULT_ADDR: str = "nats://localhost:4222"

    def __post_init__(self):
        logger.debug(
            f"NatsConstants initialized with DEFAULT_ADDR: {self.DEFAULT_ADDR}"
        )
        logger.debug(
            f"Weaver stream: {self.weaver_stream.STREAM}, Consumer: {self.weaver_stream.CONSUMER}"
        )
        logger.debug(
            f"UI stream: {self.ui_stream.STREAM}, Consumer: {self.ui_stream.CONSUMER}"
        )
        logger.debug(
            f"Weaver ingress stream: {self.weaver_ingress_stream.STREAM}, Consumer: {self.weaver_ingress_stream.CONSUMER}, Subjects: {self.weaver_ingress_stream.subjects}"
        )


# Create a global instance for easy access
NatsConstants = NatsConstantsClass()

# Log important constants on module import
logger.info("torchLoom constants module loaded")
