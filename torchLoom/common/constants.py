from dataclasses import dataclass

from torchLoom.log_utils.logger import setup_logger

class LoggerConstants:
    MANAGER_torchLoom_LOG_FILE: str = "./torchLoom/log_utils/torchLoom.log_utils"
    MANAGER_RUNTIME_LOG_FILE: str = "./torchLoom/log_utils/manager.log"
    torchLoom_CONSTANTS_LOG_FILE: str = "./torchLoom/log_utils/torchLoom_constants.log"
    torchLoom_UTILS_LOG_FILE: str = "./torchLoom/log_utils/torchLoom_utils.log"
    torchLoom_CONTROLLER_LOG_FILE: str = "./torchLoom/log_utils/torchLoom_weaver.log"
    torchLoom_MONITOR_CLI_LOG_FILE: str = "./torchLoom/log_utils/torchLoom_monitor_cli.log"
    FORMAT_LOG: bool = False
    
logger = setup_logger(
    name="torchLoom_constants", log_file=LoggerConstants.torchLoom_CONSTANTS_LOG_FILE
)

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

class TimeConstants:
    """All timing-related constants for torchLoom."""

    # Broadcast and monitoring intervals
    STATUS_BROADCAST_IN: float = 1.0
    HEARTBEAT_MONITOR_INTERVAL: float = 5.0
    HEARTBEAT_SEND_INTERVAL: float = 3.0
    HEARTBEAT_TIMEOUT: float = 8.0

    # Sleep intervals for various operations
    PIPE_POLL_INTERVAL: float = 0.1
    EXCEPTION_SLEEP: float = 5.0
    CLEANUP_SLEEP: float = 0.1

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
    
    # UI constants
    DEVICE_UNKNOWN_TIMEOUT: float = 10.0
    DEVICE_LEAVE_TIMEOUT: float = 60.0

class UINetworkConstants:
    """Network-related constants for torchLoom."""

    DEFAULT_UI_HOST: str = "0.0.0.0"
    DEFAULT_UI_PORT: int = 8079

    # CORS origins for WebSocket connections
    CORS_ORIGINS = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]


# NATS related constants
class torchLoomSubjects:
    """Subjects for torchLoom communication channels and configuration."""
    WEAVER_COMMANDS: str = "torchLoom.weaver.commands" # Weaver -> Training Process subjects
    THREADLET_EVENTS: str = "torchLoom.threadlet.events" # All events from threadlets
    EXTERNAL_EVENTS: str = "torchLoom.external.events" # All events from external systems

class StreamSpec:
    STREAM = None
    CONSUMER = None
    subjects = None

class WeaverOutgressStream(StreamSpec):
    """Weaver -> Training Process subjects"""
    STREAM: str = "WEAVER_COMMANDS_STREAM"
    CONSUMER: str = "weaver-commands-consumer"
    subjects = [torchLoomSubjects.WEAVER_COMMANDS]

class WeaverIngressStream(StreamSpec):
    """Threadlets/External -> Weaver"""
    STREAM: str = "WEAVER_INGRESS_STREAM"
    CONSUMER: str = "weaver-ingress-consumer"
    subjects = [
        torchLoomSubjects.THREADLET_EVENTS,
        torchLoomSubjects.EXTERNAL_EVENTS,
    ]

class NatsConstants:
    """Constants for torchLoom communication channels and configuration."""
    subjects: torchLoomSubjects = torchLoomSubjects()
    weaver_outgress_stream: WeaverOutgressStream = WeaverOutgressStream()
    weaver_ingress_stream: WeaverIngressStream = WeaverIngressStream()
    DEFAULT_ADDR: str = "nats://localhost:4222"

logger.info("torchLoom constants module loaded")
