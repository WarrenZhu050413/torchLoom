#!/usr/bin/env python3
"""
Test script for heartbeat functionality.
Demonstrates how the weaver detects dead processes based on missing heartbeats.
"""

import asyncio
import os
import signal
import sys
import time
from typing import Set
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightning as L
import torch
import torch.nn as nn

from torchLoom.lightning_wrapper import (
    ThreadletWrapper,
    make_threadlet,
    threadlet_handler,
)
from torchLoom.proto.torchLoom_pb2 import EventEnvelope, Heartbeat
from torchLoom.weaver.handlers import ThreadletHandler
from torchLoom.weaver.status_tracker import StatusTracker

# Mock NATS connection
with patch("nats.connect") as mock_connect:
    mock_nc = MagicMock()
    mock_js = MagicMock()
    mock_nc.jetstream.return_value = mock_js
    mock_connect.return_value = mock_nc


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class HeartbeatTestModule(L.LightningModule):
    """Lightning module for testing heartbeat functionality."""

    def __init__(self):
        super().__init__()
        self.learning_rate = 0.001
        self.model = TestModel()

    @threadlet_handler("learning_rate", float)
    def update_learning_rate(self, new_lr: float):
        """Handler for dynamic learning rate updates."""
        print(f"📈 Learning rate updated to: {new_lr}")
        self.learning_rate = new_lr

    def training_step(self, batch, batch_idx):
        # Simulate training step
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


def simulate_training_replica(replica_id: str, duration: int = 60):
    """Simulate a training replica that will be alive for a certain duration."""
    print(f"🚀 Starting training replica: {replica_id}")

    # Create Lightning module and wrap with threadlet
    trainer_module = HeartbeatTestModule()
    threadlet_trainer = make_threadlet(trainer_module, replica_id=replica_id)

    print(f"✅ Threadlet started for replica: {replica_id}")
    print(f"   PID: {os.getpid()}")
    print(f"   Will run for {duration} seconds...")

    try:
        # Simulate being alive for the specified duration
        time.sleep(duration)
        print(f"⏰ Replica {replica_id} completed its run duration")

    except KeyboardInterrupt:
        print(f"🛑 Replica {replica_id} interrupted by user")

    finally:
        print(f"🧹 Cleaning up replica: {replica_id}")
        threadlet_trainer.cleanup()
        print(f"✅ Replica {replica_id} cleaned up successfully")


async def simulate_weaver_heartbeat_monitor():
    """Simulate a weaver monitoring heartbeats."""
    print("🎯 Starting weaver heartbeat monitor...")

    # Create status tracker and threadlet handler
    status_tracker = StatusTracker()
    device_mapper = None  # Mock device mapper for testing
    threadlet_handler = ThreadletHandler(
        device_mapper=device_mapper,
        status_tracker=status_tracker,
        heartbeat_timeout=45.0,  # 45 second timeout for testing
    )

    print("📡 Heartbeat monitor active (45-second timeout)")
    print("   Will check for dead replicas every 10 seconds...")

    try:
        while True:
            await asyncio.sleep(10)  # Check every 10 seconds

            # Check for dead replicas
            dead_replicas = threadlet_handler.check_dead_replicas()

            if dead_replicas:
                print(f"💀 DEAD REPLICAS DETECTED: {list(dead_replicas)}")
                # In a real weaver, this would trigger recovery actions

            # ThreadletHandler doesn't have get_live_replicas method,
            # so we'll track live replicas differently
            all_replicas = set(threadlet_handler._last_heartbeats.keys())
            live_replicas = all_replicas - threadlet_handler._dead_replicas
            total_dead = len(threadlet_handler._dead_replicas)

            if live_replicas or total_dead > 0:
                print(f"📊 Status - Live: {len(live_replicas)}, Dead: {total_dead}")
                if live_replicas:
                    print(f"   Live replicas: {list(live_replicas)}")

    except KeyboardInterrupt:
        print("🛑 Heartbeat monitor stopped by user")


def test_scenario_1_normal_operation():
    """Test scenario 1: Normal operation with heartbeats."""
    print("\n" + "=" * 80)
    print("🧪 TEST SCENARIO 1: Normal Operation")
    print("   Replica will run for 30 seconds with regular heartbeats")
    print("=" * 80)

    simulate_training_replica("test_replica_1", duration=30)
    print("✅ Scenario 1 completed successfully")


def test_scenario_2_process_death():
    """Test scenario 2: Process dies (simulated by ending early)."""
    print("\n" + "=" * 80)
    print("🧪 TEST SCENARIO 2: Process Death Simulation")
    print("   Replica will run for 20 seconds, then die")
    print("   Weaver should detect death after ~45 seconds of no heartbeats")
    print("=" * 80)

    simulate_training_replica("test_replica_2", duration=20)
    print("💀 Replica 2 has 'died' (stopped sending heartbeats)")
    print("   Monitor should detect this as dead in ~25 more seconds...")


async def test_scenario_3_concurrent_replicas():
    """Test scenario 3: Multiple replicas with different lifespans."""
    print("\n" + "=" * 80)
    print("🧪 TEST SCENARIO 3: Multiple Replicas")
    print("   Testing concurrent replicas with different lifespans")
    print("=" * 80)

    import multiprocessing

    # Start multiple replica processes
    processes = []

    for i in range(3):
        replica_id = f"concurrent_replica_{i+1}"
        duration = 20 + (i * 15)  # 20, 35, 50 seconds

        process = multiprocessing.Process(
            target=simulate_training_replica,
            args=(replica_id, duration),
            name=f"replica-{replica_id}",
        )
        process.start()
        processes.append(process)
        print(f"🚀 Started {replica_id} (will run for {duration}s)")
        await asyncio.sleep(2)  # Stagger starts

    # Wait for all processes to complete
    for process in processes:
        process.join()

    print("✅ All concurrent replicas completed")


async def main():
    """Main test function."""
    print("🧪 HEARTBEAT FUNCTIONALITY TEST")
    print("=" * 80)
    print("This test demonstrates:")
    print("  1. Threadlets sending periodic heartbeats to weaver")
    print("  2. Weaver detecting dead processes when heartbeats stop")
    print("  3. Process lifecycle management")
    print("=" * 80)

    choice = input(
        "\nSelect test scenario:\n"
        "1. Normal operation (30 seconds)\n"
        "2. Process death simulation\n"
        "3. Multiple concurrent replicas\n"
        "4. Start heartbeat monitor only\n"
        "Choice (1-4): "
    ).strip()

    if choice == "1":
        test_scenario_1_normal_operation()
    elif choice == "2":
        test_scenario_2_process_death()
    elif choice == "3":
        await test_scenario_3_concurrent_replicas()
    elif choice == "4":
        await simulate_weaver_heartbeat_monitor()
    else:
        print("❌ Invalid choice. Exiting.")
        return

    print("\n🎉 Heartbeat test completed!")
    print("In a real deployment:")
    print("  • Weaver would automatically restart dead replicas")
    print("  • Failed processes would be replaced with new ones")
    print("  • Training would continue seamlessly")


def run_interactive_heartbeat_test():
    """Interactive test that lets you control the scenario."""
    print("\n" + "=" * 80)
    print("🎮 INTERACTIVE HEARTBEAT TEST")
    print("=" * 80)

    replica_id = input(
        "Enter replica ID (or press Enter for 'interactive_test'): "
    ).strip()
    if not replica_id:
        replica_id = "interactive_test"

    duration = input("Enter duration in seconds (or press Enter for 60): ").strip()
    if not duration:
        duration = 60
    else:
        try:
            duration = int(duration)
        except ValueError:
            print("❌ Invalid duration, using 60 seconds")
            duration = 60

    print(f"\n🚀 Starting interactive replica: {replica_id}")
    print(f"💡 The replica will send heartbeats every 30 seconds")
    print(f"⏰ It will run for {duration} seconds")
    print(f"🛑 Press Ctrl+C to kill the process early (simulate death)")
    print("-" * 80)

    simulate_training_replica(replica_id, duration)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive_heartbeat_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        asyncio.run(simulate_weaver_heartbeat_monitor())
    else:
        asyncio.run(main())
