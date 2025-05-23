from dataclasses import dataclass

from torchLoom.config import Config
from torchLoom.log.logger import setup_logger

logger = setup_logger(
    name="torchLoom_constants", log_file=Config.torchLoom_CONSTANTS_LOG_FILE
)

NC = "nc"
JS = "js"


class torchLoomSubjects:
    MONITOR: str = "torchLoom.monitored.failure"
    CONTROLLER_EVENTS: str = "torchLoom.weaver.events"
    DR_SUBJECT: str = "torchLoom.DRentry"
    REPLICA_FAIL: str = "torchLoom.replica.fail"
    CONFIG_INFO: str = "torchLoom.config.info"
    # UI-related subjects
    GPU_STATUS: str = "torchLoom.gpu.status"
    TRAINING_PROGRESS: str = "torchLoom.training.progress"
    SYSTEM_TOPOLOGY: str = "torchLoom.system.topology"
    UI_COMMAND: str = "torchLoom.ui.command"
    UI_STATUS_UPDATE: str = "torchLoom.ui.status.update"


class WeaverSubjects:
    DR_SUBJECT: str = torchLoomSubjects.DR_SUBJECT


class StreamSpec:
    STREAM = None
    CONSUMER = None
    subjects = None


class WeaverStream(StreamSpec):
    STREAM: str = "CONTROLLER-STREAM"
    CONSUMER: str = "weaver-consumer"
    subjects = WeaverSubjects()


class MonitorSubjects:
    MONITOR: str = torchLoomSubjects.MONITOR
    CONTROLLER_EVENTS: str = torchLoomSubjects.CONTROLLER_EVENTS


class MonitorStream(StreamSpec):
    STREAM: str = "MONITOR-STREAM"
    CONSUMER: str = "monitor-consumer"
    subjects = MonitorSubjects()


# UI-specific stream for WebSocket updates
class UISubjects:
    GPU_STATUS: str = torchLoomSubjects.GPU_STATUS
    TRAINING_PROGRESS: str = torchLoomSubjects.TRAINING_PROGRESS
    SYSTEM_TOPOLOGY: str = torchLoomSubjects.SYSTEM_TOPOLOGY
    UI_COMMAND: str = torchLoomSubjects.UI_COMMAND
    UI_STATUS_UPDATE: str = torchLoomSubjects.UI_STATUS_UPDATE


class UIStream(StreamSpec):
    STREAM: str = "UI-STREAM"
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


# Create a global instance for easy access
torchLoomConstants = torchLoomConstantsClass()

# Log important constants on module import
logger.info("torchLoom constants module loaded")
