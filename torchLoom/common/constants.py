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
    MONITOR: str = "torchLoom.monitored.failure"
    CONTROLLER_EVENTS: str = "torchLoom.weaver.events"
    DR_SUBJECT: str = "torchLoom.DRentry"
    REPLICA_FAIL: str = "torchLoom.replica.fail"
    CONFIG_INFO: str = "torchLoom.config.info"

    # Heartbeat subject
    HEARTBEAT: str = "torchLoom.heartbeat"

    # Training Process -> Weaver subjects
    TRAINING_STATUS: str = "torchLoom.training.status"
    device_STATUS: str = "torchLoom.device.status"
    
    # UI <-> Weaver subjects
    UI_COMMANDS: str = "torchLoom.ui.commands"
    UI_UPDATE: str = "torchLoom.ui.update"
    
    # Weaver -> Training Process subjects
    WEAVER_COMMANDS: str = "torchLoom.weaver.commands"

    WEAVELET_STATUS: str = "torchLoom.weavelet.status"


class WeaverSubjects:
    DR_SUBJECT: str = torchLoomSubjects.DR_SUBJECT
    UI_COMMANDS: str = torchLoomSubjects.UI_COMMANDS
    CONFIG_INFO: str = torchLoomSubjects.CONFIG_INFO
    WEAVER_COMMANDS: str = torchLoomSubjects.WEAVER_COMMANDS


class StreamSpec:
    STREAM = None
    CONSUMER = None
    subjects = None


class WeaverStream(StreamSpec):
    STREAM: str = "WEAVELET_STREAM"
    CONSUMER: str = "weaver-consumer"
    subjects = WeaverSubjects()


class MonitorSubjects:
    MONITOR: str = torchLoomSubjects.MONITOR
    CONTROLLER_EVENTS: str = torchLoomSubjects.CONTROLLER_EVENTS


class MonitorStream(StreamSpec):
    STREAM: str = "CONTROLLER_STREAM"
    CONSUMER: str = "monitor-consumer"
    subjects = MonitorSubjects()


# UI-specific stream for WebSocket updates
class UISubjects:
    UI_COMMANDS: str = torchLoomSubjects.UI_COMMANDS
    UI_UPDATE: str = torchLoomSubjects.UI_UPDATE


class UIStream(StreamSpec):
    STREAM: str = "UI_STREAM"
    CONSUMER: str = "ui-consumer"
    subjects = UISubjects()


@dataclass
class torchLoomConstantsClass:
    """Constants for torchLoom communication channels and configuration."""

    subjects: torchLoomSubjects = torchLoomSubjects()
    weaver_stream: WeaverStream = WeaverStream()
    monitor_stream: MonitorStream = MonitorStream()
    ui_stream: UIStream = UIStream()
    DEFAULT_ADDR: str = Config.DEFAULT_ADDR

    def __post_init__(self):
        logger.debug(
            f"torchLoomConstants initialized with DEFAULT_ADDR: {self.DEFAULT_ADDR}"
        )
        logger.debug(
            f"Weaver stream: {self.weaver_stream.STREAM}, Consumer: {self.weaver_stream.CONSUMER}"
        )
        logger.debug(
            f"Monitor stream: {self.monitor_stream.STREAM}, Consumer: {self.monitor_stream.CONSUMER}"
        )
        logger.debug(
            f"UI stream: {self.ui_stream.STREAM}, Consumer: {self.ui_stream.CONSUMER}"
        )
        logger.debug(
            f"Subject paths: DR_SUBJECT={self.subjects.DR_SUBJECT}, MONITOR={self.subjects.MONITOR}"
        )
        logger.debug(
            f"UI subjects: UI_COMMANDS={self.subjects.UI_COMMANDS}, UI_UPDATE={self.subjects.UI_UPDATE}"
        )


# Create a global instance for easy access
torchLoomConstants = torchLoomConstantsClass()

# Log important constants on module import
logger.info("torchLoom constants module loaded")
