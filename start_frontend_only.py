#!/usr/bin/env python3
"""
Simple script to start only the Streamlit frontend
Use this when the backend is already running
"""

import subprocess
import sys
import requests

def check_backend_running():
    """Check if the FastAPI backend is running on localhost:8000"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Start the Streamlit frontend"""
    print("🔍 H-MaMa Fake News Detection Frontend")
    print("=" * 50)
    
    # Check if backend is running
    if not check_backend_running():
        print("❌ Backend not detected at http://localhost:8000")
        print("Please start the backend first:")
        print("   uvicorn src.hmama.serve.api:app --reload --host 0.0.0.0 --port 8000")
        return
    
    print("✅ Backend is running!")
    print("🎨 Starting Streamlit frontend...")
    print("🌐 Frontend will be available at: http://localhost:8501")
    print("=" * 50)
    
    try:
        # Start the frontend
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "frontend.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")

if __name__ == "__main__":
    main()
