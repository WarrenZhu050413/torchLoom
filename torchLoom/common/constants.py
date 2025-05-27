from dataclasses import dataclass

from torchLoom.common.config import Config
from torchLoom.log.logger import setup_logger

logger = setup_logger(
    name="torchLoom_constants", log_file=Config.torchLoom_CONSTANTS_LOG_FILE
)

NC = "nc"
JS = "js"
NATS_SERVER_PATH = "./nats/nats-server"


class TimeConstants:
    STATUS_BROADCAST_IN: float = 1.0
    HEARTBEAT_MONITOR_INTERVAL: float = 30.0


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
