#!/usr/bin/env python3
"""
Test script to verify UI integration with status_tracker.py
"""

import asyncio
import json
import websockets
import requests
import time
from torchLoom.weaver.status_tracker import StatusTracker
from torchLoom.weaver.websocket_server import WebSocketServer


async def test_websocket_connection():
    """Test WebSocket connection to the UI backend."""
    print("Testing WebSocket connection...")
    
    try:
        uri = "ws://localhost:8080/ws"
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connection established")
            
            # Wait for initial status message
            message = await websocket.recv()
            data = json.loads(message)
            
            print(f"✅ Received initial status: {data['type']}")
            if data['type'] == 'status_update':
                status = data['data']
                print(f"   Global step: {status.get('step', 'N/A')}")
                print(f"   Replica groups: {len(status.get('replicaGroups', {}))}")
                print(f"   Communication status: {status.get('communicationStatus', 'N/A')}")
            
            # Send a ping message
            await websocket.send(json.dumps({"type": "ping"}))
            
            # Wait for pong response
            response = await websocket.recv()
            pong_data = json.loads(response)
            
            if pong_data.get('type') == 'pong':
                print("✅ Ping/pong test successful")
            else:
                print(f"❌ Expected pong, got: {pong_data}")
                
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False
    
    return True


def test_rest_api():
    """Test REST API endpoints."""
    print("\nTesting REST API endpoints...")
    
    base_url = "http://localhost:8080/api"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health endpoint working")
            print(f"   Status: {health_data.get('status')}")
            print(f"   GPUs: {health_data.get('gpus')}")
            print(f"   Replicas: {health_data.get('replicas')}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
        
        # Test status endpoint
        response = requests.get(f"{base_url}/status")
        if response.status_code == 200:
            status_data = response.json()
            print("✅ Status endpoint working")
            print(f"   Global step: {status_data.get('step')}")
            print(f"   Replica groups: {len(status_data.get('replicaGroups', {}))}")
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ REST API test failed: {e}")
        return False
    
    return True


def test_status_tracker_data():
    """Test that StatusTracker produces expected data format."""
    print("\nTesting StatusTracker data format...")
    
    # Create a test status tracker
    tracker = StatusTracker()
    
    # Add some test data
    tracker.update_gpu_status(
        gpu_id="1-0-0",
        replica_id="demo_replica_1", 
        server_id="server-1-0",
        status="active",
        utilization=75.5,
        temperature=65.2,
        memory_used=4.2,
        memory_total=8.0,
        config={"batch_size": "32", "learning_rate": "0.001", "optimizer_type": "Adam"}
    )
    
    tracker.update_training_progress(
        replica_id="demo_replica_1",
        current_step=100,
        step_progress=45.5,
        status="training"
    )
    
    # Test that the data structure is correct
    if "1-0-0" in tracker.gpus:
        gpu = tracker.gpus["1-0-0"]
        print("✅ GPU data structure correct")
        print(f"   GPU ID: {gpu.gpu_id}")
        print(f"   Status: {gpu.status}")
        print(f"   Utilization: {gpu.utilization}%")
        print(f"   Temperature: {gpu.temperature}°C")
        print(f"   Config: {gpu.config}")
    else:
        print("❌ GPU data not found")
        return False
    
    if "demo_replica_1" in tracker.replicas:
        replica = tracker.replicas["demo_replica_1"]
        print("✅ Replica data structure correct")
        print(f"   Replica ID: {replica.replica_id}")
        print(f"   Status: {replica.status}")
        print(f"   Current step: {replica.current_step}")
        print(f"   Step progress: {replica.step_progress}%")
    else:
        print("❌ Replica data not found")
        return False
    
    return True


def test_ui_data_format():
    """Test that WebSocketServer formats data correctly for UI."""
    print("\nTesting UI data format...")
    
    # Create test components
    tracker = StatusTracker()
    server = WebSocketServer(tracker)
    
    # Add test data
    tracker.update_gpu_status(
        gpu_id="1-0-0",
        replica_id="demo_replica_1", 
        server_id="server-1-0",
        status="active",
        utilization=85.3,
        temperature=68.7,
        memory_used=5.2,
        memory_total=8.0,
        config={"batch_size": "64", "learning_rate": "0.002", "optimizer_type": "SGD"}
    )
    
    tracker.update_training_progress(
        replica_id="demo_replica_1",
        current_step=150,
        step_progress=75.0,
        status="training"
    )
    
    tracker.set_global_step(150)
    tracker.set_communication_status("stable")
    
    # Get UI-formatted data
    ui_data = server.get_ui_status_dict()
    
    # Verify structure
    expected_keys = ["step", "replicaGroups", "communicationStatus", "systemSummary", "timestamp"]
    for key in expected_keys:
        if key not in ui_data:
            print(f"❌ Missing key in UI data: {key}")
            return False
    
    print("✅ UI data format correct")
    print(f"   Global step: {ui_data['step']}")
    print(f"   Communication status: {ui_data['communicationStatus']}")
    
    # Check replica groups structure
    replica_groups = ui_data['replicaGroups']
    if "demo" in replica_groups:
        group = replica_groups["demo"]
        print(f"   Replica group 'demo' found with {len(group['gpus'])} GPUs")
        
        if "1-0-0" in group['gpus']:
            gpu = group['gpus']["1-0-0"]
            expected_gpu_keys = ["id", "server", "status", "utilization", "temperature", "batch", "lr", "opt"]
            for key in expected_gpu_keys:
                if key not in gpu:
                    print(f"❌ Missing key in GPU data: {key}")
                    return False
            print("✅ GPU data in UI format is complete")
        else:
            print("❌ GPU not found in UI data")
            return False
    else:
        print("❌ Replica group not found in UI data")
        return False
    
    return True


async def main():
    """Run all tests."""
    print("🧪 Starting torchLoom UI Integration Tests\n")
    
    # Test 1: Status tracker data format
    success1 = test_status_tracker_data()
    
    # Test 2: UI data format
    success2 = test_ui_data_format()
    
    print("\n" + "="*50)
    print("🚀 To run the live server tests:")
    print("1. Start the Weaver server: python -m torchLoom.weaver.core")
    print("2. Start the UI: cd torchLoom-ui && npm run dev")
    print("3. Run live tests: python test_ui_integration.py --live")
    print("="*50)
    
    if success1 and success2:
        print("\n✅ All static tests passed!")
        return True
    else:
        print("\n❌ Some tests failed")
        return False


if __name__ == "__main__":
    import sys
    
    if "--live" in sys.argv:
        # Run live tests that require running servers
        async def run_live_tests():
            print("🔴 Running live integration tests...")
            print("Make sure the Weaver server is running on localhost:8080")
            
            # Wait a moment for servers to be ready
            await asyncio.sleep(1)
            
            success1 = await test_websocket_connection()
            success2 = test_rest_api()
            
            if success1 and success2:
                print("\n✅ All live tests passed!")
            else:
                print("\n❌ Some live tests failed")
        
        asyncio.run(run_live_tests())
    else:
        # Run static tests
        asyncio.run(main()) 