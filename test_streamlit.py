#!/usr/bin/env python3
"""
Simple test to verify Streamlit can start
"""

import subprocess
import sys
import time
import webbrowser
import threading

def test_basic_streamlit():
    """Test basic Streamlit functionality"""
    print("🧪 Testing basic Streamlit functionality...")
    
    # Create a minimal test app
    test_app = """
import streamlit as st
st.title("🧪 Streamlit Test")
st.write("If you can see this, Streamlit is working!")
st.write("Backend status: Checking...")

import requests
try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    if response.status_code == 200:
        st.success("✅ Backend is connected!")
    else:
        st.error(f"❌ Backend error: {response.status_code}")
except Exception as e:
    st.error(f"❌ Backend not reachable: {e}")
"""
    
    # Write test app to file
    with open("test_app.py", "w") as f:
        f.write(test_app)
    
    print("✅ Test app created")
    print("🚀 Starting Streamlit test...")
    print("🌐 Opening browser in 3 seconds...")
    
    # Open browser after delay
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://localhost:8501")
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "test_app.py",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Test stopped")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        # Clean up
        try:
            import os
            os.remove("test_app.py")
        except:
            pass

if __name__ == "__main__":
    test_basic_streamlit()
