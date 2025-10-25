#!/usr/bin/env python3
"""
Troubleshooting script for H-MaMa frontend issues
"""

import subprocess
import sys
import requests
import socket
import time

def check_port_available(port):
    """Check if a port is available"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result != 0

def check_backend():
    """Check backend status"""
    print("🔍 Checking backend...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and healthy")
            return True
        else:
            print(f"❌ Backend responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running")
        return False
    except Exception as e:
        print(f"❌ Backend check failed: {e}")
        return False

def check_frontend_port():
    """Check if frontend port is available"""
    print("🔍 Checking frontend port...")
    if check_port_available(8501):
        print("✅ Port 8501 is available")
        return True
    else:
        print("❌ Port 8501 is already in use")
        return False

def test_streamlit_install():
    """Test if Streamlit is properly installed"""
    print("🔍 Testing Streamlit installation...")
    try:
        result = subprocess.run([sys.executable, "-c", "import streamlit; print(streamlit.__version__)"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Streamlit is installed (version: {version})")
            return True
        else:
            print(f"❌ Streamlit test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Streamlit test error: {e}")
        return False

def start_simple_frontend():
    """Start frontend with minimal configuration"""
    print("🚀 Starting frontend with minimal configuration...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "frontend.py",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main troubleshooting function"""
    print("🔧 H-MaMa Frontend Troubleshooting")
    print("=" * 50)
    
    # Run all checks
    backend_ok = check_backend()
    port_ok = check_frontend_port()
    streamlit_ok = test_streamlit_install()
    
    print("\n📊 Summary:")
    print(f"Backend: {'✅' if backend_ok else '❌'}")
    print(f"Port 8501: {'✅' if port_ok else '❌'}")
    print(f"Streamlit: {'✅' if streamlit_ok else '❌'}")
    
    if not backend_ok:
        print("\n💡 Solution: Start the backend first:")
        print("   uvicorn src.hmama.serve.api:app --reload --host 0.0.0.0 --port 8000")
        return
    
    if not port_ok:
        print("\n💡 Solution: Kill the process using port 8501:")
        print("   netstat -ano | findstr :8501")
        print("   taskkill /PID <PID_NUMBER> /F")
        return
    
    if not streamlit_ok:
        print("\n💡 Solution: Install Streamlit:")
        print("   pip install streamlit")
        return
    
    print("\n✅ All checks passed! Starting frontend...")
    time.sleep(2)
    start_simple_frontend()

if __name__ == "__main__":
    main()
