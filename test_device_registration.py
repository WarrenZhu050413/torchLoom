#!/usr/bin/env python3
"""Test script to register a device with the Weaver and verify it appears in the UI."""

import asyncio
import time
from torchLoom.cli import TorchLoomClient

async def test_device_registration():
    """Test device registration and verify it shows up in the system."""
    print("🧪 Testing device registration...")
    
    try:
        # Create a client and register a device
        async with TorchLoomClient() as client:
            device_id = f"test_device_{int(time.time())}"
            replica_id = f"test_replica_{int(time.time())}"
            
            print(f"📱 Registering device: {device_id} -> {replica_id}")
            await client.register_device(device_id, replica_id)
            
            print("✅ Device registration sent!")
            print(f"   Device: {device_id}")
            print(f"   Replica: {replica_id}")
            print("   Check the UI at http://localhost:5173 to see if it appears!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_device_registration()) 