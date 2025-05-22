from dataclasses import dataclass
from torchLoom.log.logger import setup_logger
from torchLoom.config import Config

logger = setup_logger(name="torchLoom_constants", log_file=Config.torchLoom_CONSTANTS_LOG_FILE)

NC = "nc"
JS = "js"

class torchLoomSubjects:
    EXTERNAL: str = "torchLoom.monitored.failure"
    CONTROLLER_EVENTS: str = "torchLoom.weaver.events"
    DR_SUBJECT: str = "torchLoom.DRentry"
    REPLICA_FAIL: str = "torchLoom.replica.fail"

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
    EXTERNAL: str = torchLoomSubjects.EXTERNAL
    CONTROLLER_EVENTS: str = torchLoomSubjects.CONTROLLER_EVENTS

class MonitorStream(StreamSpec):
    STREAM: str = "MONITOR-STREAM"
    CONSUMER: str = "monitor-consumer"
    subjects = MonitorSubjects()

@dataclass
class torchLoomConstants:
    """Constants for torchLoom communication channels and configuration."""
    subjects: torchLoomSubjects = torchLoomSubjects()
    weaver_stream: WeaverStream = WeaverStream()
    monitor_stream: MonitorStream = MonitorStream()
    DEFAULT_ADDR: str = Config.DEFAULT_ADDR
    
    def __post_init__(self):
        logger.debug(f"torchLoomConstants initialized with DEFAULT_ADDR: {self.DEFAULT_ADDR}")
        logger.debug(f"Weaver stream: {self.weaver_stream.STREAM}, Consumer: {self.weaver_stream.CONSUMER}")
        logger.debug(f"Monitor stream: {self.monitor_stream.STREAM}, Consumer: {self.monitor_stream.CONSUMER}")
        logger.debug(f"Subject paths: DR_SUBJECT={self.subjects.DR_SUBJECT}, EXTERNAL={self.subjects.EXTERNAL}")

# Log important constants on module import
logger.info("torchLoom constants module loaded")