#!/usr/bin/env python3
"""
Debug version of frontend startup with detailed logging
"""

import subprocess
import sys
import requests
import time
import socket

def check_backend():
    """Check if backend is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def find_free_port(start_port=8501):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + 10):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        if result != 0:  # Port is free
            return port
    return None

def start_streamlit_with_debug():
    """Start Streamlit with detailed debugging"""
    print("🔍 Finding available port...")
    port = find_free_port(8501)
    if not port:
        print("❌ No free ports found in range 8501-8510")
        return
    
    print(f"✅ Using port {port}")
    
    # Try different binding strategies
    strategies = [
        # Strategy 1: Default (no address specified)
        [sys.executable, "-m", "streamlit", "run", "frontend.py", "--server.port", str(port)],
        
        # Strategy 2: Explicit localhost
        [sys.executable, "-m", "streamlit", "run", "frontend.py", "--server.port", str(port), "--server.address", "127.0.0.1"],
        
        # Strategy 3: All interfaces
        [sys.executable, "-m", "streamlit", "run", "frontend.py", "--server.port", str(port), "--server.address", "0.0.0.0"],
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n🚀 Trying Strategy {i}: {' '.join(strategy[2:])}")
        print(f"🌐 URL: http://localhost:{port}")
        print("⏳ Starting Streamlit (press Ctrl+C to try next strategy)...")
        
        try:
            # Start the process
            process = subprocess.Popen(strategy)
            
            # Wait a bit for startup
            time.sleep(3)
            
            # Check if process is still running
            if process.poll() is None:
                print(f"✅ Streamlit started successfully on port {port}")
                print(f"🌐 Open your browser to: http://localhost:{port}")
                print("Press Ctrl+C to stop...")
                
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\n🛑 Stopping Streamlit...")
                    process.terminate()
                    process.wait()
                    break
            else:
                print(f"❌ Strategy {i} failed - process exited")
                
        except KeyboardInterrupt:
            print(f"\n⏭️ Trying next strategy...")
            continue
        except Exception as e:
            print(f"❌ Strategy {i} error: {e}")
            continue
    
    print("\n🔧 All strategies exhausted. Manual troubleshooting:")
    print("1. Check Windows Firewall settings")
    print("2. Try running as Administrator")
    print("3. Check if antivirus is blocking the connection")
    print("4. Try: streamlit run frontend.py --server.port 8501 --server.address 127.0.0.1")

def main():
    """Main function"""
    print("🔧 H-MaMa Frontend Debug Startup")
    print("=" * 50)
    
    if not check_backend():
        print("❌ Backend not running. Please start it first:")
        print("   uvicorn src.hmama.serve.api:app --reload --host 0.0.0.0 --port 8000")
        return
    
    print("✅ Backend is running")
    start_streamlit_with_debug()

if __name__ == "__main__":
    main()
