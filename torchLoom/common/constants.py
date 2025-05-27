from dataclasses import dataclass

from torchLoom.common.config import Config
from torchLoom.log.logger import setup_logger

logger = setup_logger(
    name="torchLoom_constants", log_file=Config.torchLoom_CONSTANTS_LOG_FILE
)

NC = "nc"
JS = "js"
NATS_SERVER_PATH = "./nats/nats-server"


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

    # Common configuration parameters for threadlet handlers
    COMMON_CONFIG_PARAMS = [
        "learning_rate",
        "lr",
        "batch_size",
        "momentum",
        "weight_decay",
        "optimizer_type",
        "optimizer",
        "training_enabled",
        "pause_training",
        "resume_training",
        "dropout_rate",
        "dropout",
        "log_level",
        "logging_interval",
        "verbose",
        "gradient_clip_val",
        "accumulate_grad_batches",
    ]

    # UI command types
    UI_COMMAND_TYPES = [
        "deactivate_device",
        "reactivate_group",
        "update_config",
        "global_config",
        "pause_training",
        "resume_training",
        "drain",
    ]


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
    ASYNC_TASK_SLEEP: float = 0.01
    ERROR_RETRY_SLEEP: float = 5.0
    UI_UPDATE_INTERVAL: float = 2.0
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


class NetworkConstants:
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
    # UI <-> Weaver subjects
    UI_COMMANDS: str = "torchLoom.ui.commands"

    # Weaver -> Training Process subjects
    WEAVER_COMMANDS: str = "torchLoom.weaver.commands"

    # New consolidated subjects for Weaver ingress
    THREADLET_EVENTS: str = "torchLoom.threadlet.events"  # All events from threadlets
    EXTERNAL_EVENTS: str = (
        "torchLoom.external.events"  # All events from external systems
    )


class StreamSpec:
    STREAM = None
    CONSUMER = None
    subjects = None


# UI-specific stream for WebSocket updates
class UISubjects:
    UI_COMMANDS: str = torchLoomSubjects.UI_COMMANDS


class UIStream(StreamSpec):
    STREAM: str = "UI_STREAM"
    CONSUMER: str = "ui-consumer"
    subjects = [torchLoomSubjects.UI_COMMANDS]


# Weaver commands stream (Weaver -> Threadlets)
class WeaverStream(StreamSpec):
    STREAM: str = "WEAVER_COMMANDS_STREAM"
    CONSUMER: str = "weaver-commands-consumer"
    subjects = [torchLoomSubjects.WEAVER_COMMANDS]


# New stream for consolidated Weaver ingress
class WeaverIngressSubjects:
    THREADLET_EVENTS: str = torchLoomSubjects.THREADLET_EVENTS
    EXTERNAL_EVENTS: str = torchLoomSubjects.EXTERNAL_EVENTS


class WeaverIngressStream(StreamSpec):
    STREAM: str = "WEAVER_INGRESS_STREAM"
    CONSUMER: str = "weaver-ingress-consumer"
    subjects = [
        torchLoomSubjects.THREADLET_EVENTS,
        torchLoomSubjects.EXTERNAL_EVENTS,
    ]


@dataclass
class torchLoomConstantsClass:
    """Constants for torchLoom communication channels and configuration."""

    subjects: torchLoomSubjects = torchLoomSubjects()
    ui_stream: UIStream = UIStream()
    weaver_stream: WeaverStream = WeaverStream()
    weaver_ingress_stream: WeaverIngressStream = WeaverIngressStream()
    DEFAULT_ADDR: str = Config.DEFAULT_ADDR

    def __post_init__(self):
        logger.debug(
            f"torchLoomConstants initialized with DEFAULT_ADDR: {self.DEFAULT_ADDR}"
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
torchLoomConstants = torchLoomConstantsClass()

# Log important constants on module import
logger.info("torchLoom constants module loaded")
