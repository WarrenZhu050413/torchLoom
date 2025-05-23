#!/usr/bin/env python3
"""
TorchLoom System Launcher

This script helps you start the torchLoom system components in the correct order.
"""

import asyncio
import subprocess
import sys
import time
import os
import signal
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    try:
        import nats
        print("✅ NATS client available")
    except ImportError:
        print("❌ NATS client not found. Install with: pip install nats-py")
        return False
    
    try:
        import fastapi
        print("✅ FastAPI available")
    except ImportError:
        print("❌ FastAPI not found. Install with: pip install fastapi uvicorn")
        return False
    
    try:
        import websockets
        print("✅ WebSockets available")
    except ImportError:
        print("❌ WebSockets not found. Install with: pip install websockets")
        return False
    
    return True


def start_nats_server():
    """Start NATS server if not already running."""
    print("🚀 Starting NATS server...")
    
    try:
        # Check if NATS is already running
        result = subprocess.run(
            ["nats", "server", "check"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ NATS server already running")
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    try:
        # Start NATS server
        nats_process = subprocess.Popen(
            ["nats-server", "--jetstream"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Check if it's running
        if nats_process.poll() is None:
            print("✅ NATS server started successfully")
            return nats_process
        else:
            print("❌ Failed to start NATS server")
            return None
            
    except FileNotFoundError:
        print("❌ nats-server not found. Please install NATS server")
        print("   Download from: https://github.com/nats-io/nats-server/releases")
        return None


async def start_weaver():
    """Start the torchLoom Weaver."""
    print("🧵 Starting torchLoom Weaver...")
    
    try:
        # Import and run the weaver
        from torchLoom.weaver.core import main as weaver_main
        await weaver_main()
    except KeyboardInterrupt:
        print("🛑 Weaver stopped by user")
    except Exception as e:
        print(f"❌ Error starting Weaver: {e}")
        raise


def start_ui():
    """Start the Vue.js UI."""
    print("🎨 Starting torchLoom UI...")
    
    ui_dir = Path("torchLoom-ui")
    if not ui_dir.exists():
        print("❌ UI directory not found")
        return None
    
    # Check if node_modules exists
    if not (ui_dir / "node_modules").exists():
        print("📦 Installing UI dependencies...")
        try:
            subprocess.run(
                ["npm", "install"],
                cwd=ui_dir,
                check=True
            )
        except subprocess.CalledProcessError:
            print("❌ Failed to install UI dependencies")
            return None
    
    try:
        # Start the UI development server
        ui_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=ui_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("✅ UI server starting...")
        print("   Frontend will be available at: http://localhost:5173")
        print("   API server will be available at: http://localhost:8080")
        
        return ui_process
        
    except FileNotFoundError:
        print("❌ npm not found. Please install Node.js")
        return None


async def main():
    """Main launcher function."""
    print("🚀 TorchLoom System Launcher")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing dependencies.")
        return
    
    processes = []
    
    try:
        # Start NATS server
        nats_process = start_nats_server()
        if nats_process:
            processes.append(nats_process)
        
        print("\n" + "=" * 40)
        print("🎯 Starting torchLoom Components...")
        print("   Press Ctrl+C to stop all services")
        print("=" * 40)
        
        # Option 1: Start Weaver only (includes built-in UI server)
        if "--weaver-only" in sys.argv:
            print("🔧 Starting Weaver-only mode (with built-in UI server)")
            await start_weaver()
        
        # Option 2: Start both Weaver and separate UI development server
        else:
            print("🔧 Starting full development mode (Weaver + separate UI server)")
            
            # Start UI server in background
            ui_process = start_ui()
            if ui_process:
                processes.append(ui_process)
            
            # Start Weaver (this will block)
            await start_weaver()
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down torchLoom...")
    
    finally:
        # Clean up all processes
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception:
                pass
        
        print("✅ All services stopped")


def print_usage():
    """Print usage information."""
    print("""
TorchLoom System Launcher

Usage:
    python start_torchloom.py [options]

Options:
    --weaver-only    Start only the Weaver with built-in UI server
    --help           Show this help message

Examples:
    # Start full development environment (recommended)
    python start_torchloom.py

    # Start production-like mode with built-in UI
    python start_torchloom.py --weaver-only

    # Test the integration
    python test_ui_integration.py

After starting, you can access:
    - UI Frontend: http://localhost:5173 (dev mode) or http://localhost:8080 (weaver-only)
    - API Backend: http://localhost:8080/api
    - WebSocket: ws://localhost:8080/ws
""")


if __name__ == "__main__":
    if "--help" in sys.argv:
        print_usage()
    else:
        asyncio.run(main()) 