"""
Common status type definitions for torchLoom.

These dataclasses provide a clean, reusable way to structure status information
that gets published from training processes to the weaver and UI.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time


@dataclass
class TrainingStatus:
    """Training status information matching the protobuf TrainingStatus message."""
    
    replica_id: str
    status_type: str  # "training_start", "epoch_start", "batch_update", "epoch_complete", "training_complete"
    current_step: int = 0
    epoch: int = 0
    step_progress: float = 0.0  # 0-100% within current step
    epoch_progress: float = 0.0  # 0-100% within current epoch
    status: str = "training"  # "starting", "training", "completed", "paused"
    metrics: Dict[str, str] = field(default_factory=dict)  # loss, accuracy, learning_rate, etc.
    training_time: float = 0.0  # Total training time in seconds
    batch_idx: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "replica_id": self.replica_id,
            "status_type": self.status_type,
            "current_step": self.current_step,
            "epoch": self.epoch,
            "step_progress": self.step_progress,
            "epoch_progress": self.epoch_progress,
            "status": self.status,
            "metrics": self.metrics,
            "training_time": self.training_time,
            "batch_idx": self.batch_idx,
            "timestamp": self.timestamp,
            "type": "training_status"  # For backward compatibility
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingStatus":
        """Create from dictionary."""
        return cls(
            replica_id=data.get("replica_id", ""),
            status_type=data.get("status_type", "batch_update"),
            current_step=data.get("current_step", data.get("step", 0)),
            epoch=data.get("epoch", 0),
            step_progress=data.get("step_progress", 0.0),
            epoch_progress=data.get("epoch_progress", 0.0),
            status=data.get("status", "training"),
            metrics=data.get("metrics", {}),
            training_time=data.get("training_time", 0.0),
            batch_idx=data.get("batch_idx", 0),
            timestamp=data.get("timestamp", time.time())
        )


@dataclass
class deviceStatus:
    """device status information matching the protobuf deviceStatus message."""
    
    device_id: str
    replica_id: str
    server_id: str
    status: str = "active"  # "active", "offline", "failed"
    utilization: float = 0.0  # 0-100%
    temperature: float = 40.0  # Celsius
    memory_used: float = 0.0  # GB
    memory_total: float = 8.0  # GB
    config: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "type": "device_status",
            "device_id": self.device_id,
            "replica_id": self.replica_id,
            "server_id": self.server_id,
            "status": self.status,
            "utilization": self.utilization,
            "temperature": self.temperature,
            "memory_used": self.memory_used,
            "memory_total": self.memory_total,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "deviceStatus":
        """Create deviceStatus from dictionary."""
        return cls(
            device_id=data.get("device_id", ""),
            replica_id=data.get("replica_id", ""),
            server_id=data.get("server_id", ""),
            status=data.get("status", "active"),
            utilization=data.get("utilization", 0.0),
            temperature=data.get("temperature", 40.0),
            memory_used=data.get("memory_used", 0.0),
            memory_total=data.get("memory_total", 8.0),
            config=data.get("config", {}),
        )

@dataclass
class ModelUpdateStatus:
    """Lightweight status for model updates intended for federated aggregation."""
    
    replica_id: str
    epoch: int
    step: int
    update_type: str = "full"  # or "gradient", "delta", etc.
    timestamp: float = field(default_factory=time.time)
    model_path: str = ""  # points to saved .pt file on shared FS or object store
    meta: Dict[str, Any] = field(default_factory=dict)  # optional: loss, sample count, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replica_id": self.replica_id,
            "epoch": self.epoch,
            "step": self.step,
            "update_type": self.update_type,
            "timestamp": self.timestamp,
            "model_path": self.model_path,
            "meta": self.meta,
            "type": "model_update"
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelUpdateStatus":
        return cls(
            replica_id=data["replica_id"],
            epoch=data.get("epoch", 0),
            step=data.get("step", 0),
            update_type=data.get("update_type", "full"),
            timestamp=data.get("timestamp", time.time()),
            model_path=data.get("model_path", ""),
            meta=data.get("meta", {}),
        )



# Union type for status updates
StatusUpdate = TrainingStatus | deviceStatus


def create_training_start_status(replica_id: str, epochs: int = 1) -> TrainingStatus:
    """Helper to create a training start status."""
    return TrainingStatus(
        replica_id=replica_id,
        status_type="training_start",
        status="starting",
        metrics={"total_epochs": str(epochs)}
    )


def create_epoch_start_status(replica_id: str, epoch: int, total_batches: int = 0) -> TrainingStatus:
    """Helper to create an epoch start status."""
    return TrainingStatus(
        replica_id=replica_id,
        status_type="epoch_start",
        epoch=epoch,
        status="training",
        metrics={"total_batches": str(total_batches)}
    )


def create_batch_update_status(
    replica_id: str, 
    epoch: int, 
    batch_idx: int, 
    step: int, 
    loss: float, 
    learning_rate: float,
    step_progress: float = 0.0
) -> TrainingStatus:
    """Helper to create a batch update status."""
    return TrainingStatus(
        replica_id=replica_id,
        status_type="batch_update",
        epoch=epoch,
        batch_idx=batch_idx,
        current_step=step,
        step_progress=step_progress,
        status="training",
        metrics={
            "loss": str(loss),
            "learning_rate": str(learning_rate)
        }
    )


def create_epoch_complete_status(replica_id: str, epoch: int, accuracy: Optional[float] = None) -> TrainingStatus:
    """Helper to create an epoch complete status."""
    metrics = {}
    if accuracy is not None:
        metrics["accuracy"] = str(accuracy)
    
    return TrainingStatus(
        replica_id=replica_id,
        status_type="epoch_complete",
        epoch=epoch,
        status="training",
        metrics=metrics
    )


def create_training_complete_status(replica_id: str, final_metrics: Optional[Dict[str, float]] = None) -> TrainingStatus:
    """Helper to create a training complete status."""
    metrics = {}
    if final_metrics:
        metrics.update({k: str(v) for k, v in final_metrics.items()})
    
    return TrainingStatus(
        replica_id=replica_id,
        status_type="training_complete",
        status="completed",
        metrics=metrics
    )