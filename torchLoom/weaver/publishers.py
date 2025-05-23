"""
Publishers for the torchLoom Weaver.

This module contains publishers for sending messages FROM the weaver to other components:
- UI publishers: Publish updates and responses to the UI
- Weavelet publishers: Publish commands and notifications to weavelets
- Demo utilities: Simulate data for demonstration purposes
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Set

from torchLoom.common.constants import torchLoomConstants
from torchLoom.log.logger import setup_logger
from torchLoom.proto.torchLoom_pb2 import EventEnvelope

logger = setup_logger(name="publishers")


class Publisher(ABC):
    """Abstract base class for message publishers."""

    @abstractmethod
    async def publish(self, *args, **kwargs) -> None:
        """Publish a message or update."""
        pass


# ===========================================
# UI PUBLISHERS (Weaver -> UI)
# ===========================================


class UIUpdatePublisher(Publisher):
    """Publisher for sending consolidated UI updates FROM the weaver TO the UI."""

    def __init__(self, status_tracker, nats_client=None):
        self.status_tracker = status_tracker
        self.nats_client = nats_client

    async def publish_ui_update(self) -> None:
        """Publish consolidated UIStatusUpdate to the UI."""
        try:
            if not self.nats_client:
                logger.warning("[WEAVER->UI] Cannot publish UI update - no NATS client")
                return

            # Create consolidated UIStatusUpdate
            envelope = EventEnvelope()
            ui_update = envelope.ui_status_update
            ui_update.communication_status = self.status_tracker.communication_status
            ui_update.timestamp = int(time.time())

            # Add all GPU statuses
            for gpu_info in self.status_tracker.gpus.values():
                gpu_status = ui_update.gpus.add()
                gpu_status.gpu_id = gpu_info.gpu_id
                gpu_status.replica_id = gpu_info.replica_id
                gpu_status.server_id = gpu_info.server_id
                gpu_status.status = gpu_info.status
                gpu_status.utilization = gpu_info.utilization
                gpu_status.temperature = gpu_info.temperature
                gpu_status.memory_used = gpu_info.memory_used
                gpu_status.memory_total = gpu_info.memory_total

                # Add config
                for key, value in gpu_info.config.items():
                    gpu_status.config[key] = str(value)

            # Add all training statuses
            for replica_info in self.status_tracker.replicas.values():
                training_status = ui_update.training_status.add()
                training_status.replica_id = replica_info.replica_id
                training_status.status_type = "training_update"
                training_status.current_step = replica_info.current_step
                training_status.step_progress = replica_info.step_progress
                training_status.status = replica_info.status
                training_status.batch_idx = replica_info.last_active_step

            # Add topology information
            for server_info in self.status_tracker.servers.values():
                topology = ui_update.topology.add()
                topology.server_id = server_info.server_id
                topology.replica_group_id = server_info.replica_group_id
                for gpu_id in server_info.gpu_ids:
                    topology.gpu_ids.append(gpu_id)

            # Publish to UI
            await self.nats_client.publish(
                torchLoomConstants.subjects.UI_UPDATE, envelope.SerializeToString()
            )

            logger.debug(
                "[WEAVER->UI] Published UI update to clients"
            )

        except Exception as e:
            logger.exception(f"[WEAVER->UI] Failed to publish UI update: {e}")

    async def publish(self) -> None:
        """Implement the abstract publish method."""
        await self.publish_ui_update()


# ===========================================
# WEAVELET PUBLISHERS (Weaver -> Training Processes)
# ===========================================


class WeaveletCommandPublisher(Publisher):
    """Publisher for sending commands FROM the weaver TO weavelets/training processes."""

    def __init__(self, nats_client=None):
        self.nats_client = nats_client

    async def publish_replica_fail_event(self, replica_id: str) -> None:
        """Publish a replica failure event to training processes."""
        try:
            if not self.nats_client:
                logger.warning(
                    "[WEAVER->WEAVELET] Cannot publish replica fail event - no NATS client"
                )
                return

            envelope = EventEnvelope()
            envelope.replica_fail.replica_id = replica_id

            await self.nats_client.publish(
                torchLoomConstants.subjects.REPLICA_FAIL, envelope.SerializeToString()
            )

            logger.info(
                f"[WEAVER->WEAVELET] Published replica fail event for {replica_id}"
            )

        except Exception as e:
            logger.exception(
                f"[WEAVER->WEAVELET] Failed to publish replica fail event: {e}"
            )

    async def publish_weaver_command(
        self,
        command_type: str,
        target_replica_id: str,
        params: Optional[Dict[str, str]] = None,
    ) -> None:
        """Publish a weaver command to training processes."""
        try:
            if not self.nats_client:
                logger.warning(
                    "[WEAVER->WEAVELET] Cannot publish weaver command - no NATS client"
                )
                return

            envelope = EventEnvelope()
            weaver_command = envelope.weaver_command
            weaver_command.command_type = command_type
            weaver_command.target_replica_id = target_replica_id

            if params:
                for key, value in params.items():
                    weaver_command.params[key] = str(value)

            js = self.nats_client.jetstream()
            await js.publish(
                torchLoomConstants.subjects.WEAVER_COMMANDS,
                envelope.SerializeToString(),
            )

            logger.info(
                f"[WEAVER->WEAVELET] Published command: {command_type} to {target_replica_id}"
            )

        except Exception as e:
            logger.exception(
                f"[WEAVER->WEAVELET] Failed to publish weaver command: {e}"
            )

    async def publish_config_update(self, config_params: Dict[str, str]) -> None:
        """Publish configuration updates to all training processes."""
        try:
            if not self.nats_client:
                logger.warning(
                    "[WEAVER->WEAVELET] Cannot publish config update - no NATS client"
                )
                return

            js = self.nats_client.jetstream()

            envelope = EventEnvelope()
            config_info = envelope.config_info
            for key, value in config_params.items():
                config_info.config_params[key] = str(value)

            await js.publish(
                torchLoomConstants.subjects.CONFIG_INFO, envelope.SerializeToString()
            )

            logger.info(f"[WEAVER->WEAVELET] Published config update: {config_params}")

        except Exception as e:
            logger.exception(f"[WEAVER->WEAVELET] Failed to publish config update: {e}")

    async def publish(self, message_type: str, **kwargs) -> None:
        """Generic publish method for different message types."""
        if message_type == "replica_fail":
            replica_id = kwargs.get("replica_id")
            if replica_id is not None:
                await self.publish_replica_fail_event(replica_id)
            else:
                logger.warning(
                    "[WEAVER->WEAVELET] Missing replica_id for replica_fail message"
                )
        elif message_type == "weaver_command":
            command_type = kwargs.get("command_type")
            target_replica_id = kwargs.get("target_replica_id")
            params = kwargs.get("params")
            if command_type is not None and target_replica_id is not None:
                await self.publish_weaver_command(
                    command_type, target_replica_id, params
                )
            else:
                logger.warning(
                    "[WEAVER->WEAVELET] Missing required parameters for weaver_command message"
                )
        elif message_type == "config_update":
            config_params = kwargs.get("config_params")
            if config_params is not None:
                await self.publish_config_update(config_params)
            else:
                logger.warning(
                    "[WEAVER->WEAVELET] Missing config_params for config_update message"
                )
        else:
            logger.warning(f"[WEAVER->WEAVELET] Unknown message type: {message_type}")


# ===========================================
# HEARTBEAT MONITORING (Special Publisher)
# ===========================================


class HeartbeatMonitor(Publisher):
    """Monitor for dead replicas based on heartbeat timeouts and publish failure events."""

    def __init__(self, heartbeat_handler, weavelet_command_publisher):
        self.heartbeat_handler = (
            heartbeat_handler  # Reference to the inbound heartbeat handler
        )
        self.weavelet_command_publisher = weavelet_command_publisher

    async def check_and_publish_dead_replicas(self) -> Set[str]:
        """Check for dead replicas and publish failure events for them."""
        try:
            # Get newly dead replicas from the heartbeat handler
            newly_dead_replicas = self.heartbeat_handler.check_dead_replicas()

            # Publish replica fail events for newly dead replicas
            for replica_id in newly_dead_replicas:
                await self.weavelet_command_publisher.publish_replica_fail_event(
                    replica_id
                )

            return newly_dead_replicas

        except Exception as e:
            logger.exception(f"[WEAVER] Error checking dead replicas: {e}")
            return set()

    async def publish(self) -> None:
        """Implement the abstract publish method."""
        await self.check_and_publish_dead_replicas()


# ===========================================
# DEMO UTILITIES
# ===========================================


class DemoDataSimulator:
    """Simulates training data for demo purposes."""

    def __init__(self, status_tracker):
        self.status_tracker = status_tracker
        self.demo_replicas = ["demo_replica_1", "demo_replica_2", "demo_replica_3"]
        self.demo_servers = ["server-1-0", "server-1-1", "server-2-0", "server-3-0"]

        logger.info("[DEMO] Demo data simulator initialized")

    def initialize_demo_data(self):
        """Initialize demo GPUs and replicas."""
        gpu_counter = 0

        for i, replica_id in enumerate(self.demo_replicas):
            # Create 2-4 GPUs per replica
            gpu_count = 2 + (i % 3)  # 2, 3, or 4 GPUs
            server_id = self.demo_servers[i % len(self.demo_servers)]

            for j in range(gpu_count):
                gpu_id = f"gpu-{gpu_counter}"
                gpu_counter += 1

                # Initialize GPU with demo data
                self.status_tracker.update_gpu_status(
                    gpu_id=gpu_id,
                    replica_id=replica_id,
                    server_id=server_id,
                    status="active",
                    utilization=60.0 + (gpu_counter % 30),  # 60-90%
                    temperature=50.0 + (gpu_counter % 25),  # 50-75°C
                    memory_used=2.0 + (gpu_counter % 6),  # 2-8GB
                    memory_total=8.0,
                    config={
                        "batch_size": "32",
                        "learning_rate": "0.001",
                        "optimizer_type": "Adam",
                    },
                )

            # Initialize replica progress
            self.status_tracker.update_training_progress(
                replica_id=replica_id,
                current_step=0,
                step_progress=0.0,
                status="training",
            )

        logger.info(
            f"[DEMO] Initialized demo data: {gpu_counter} GPUs across {len(self.demo_replicas)} replicas"
        )

    def simulate_training_step(self):
        """Simulate one training step across all demo components."""
        self.status_tracker.simulate_training_progress()
        logger.debug(
            "[DEMO] Simulated training step"
        )
