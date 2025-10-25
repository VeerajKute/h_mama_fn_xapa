#!/usr/bin/env python3
"""
Startup script for the H-MaMa Fake News Detection Frontend
This script checks if the backend is running and starts the Streamlit frontend.
"""

import subprocess
import sys
import time
import requests
import os

def check_backend_running():
    """Check if the FastAPI backend is running on localhost:8000"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    try:
        # Change to the project directory
        project_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(project_dir)
        
        # Set PYTHONPATH to include the project root
        env = os.environ.copy()
        env['PYTHONPATH'] = project_dir
        
        # Start the backend
        backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "src.hmama.serve.api:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], env=env)
        
        # Wait for backend to be ready with retries
        print("⏳ Waiting for backend to start...")
        max_retries = 30  # 30 seconds total
        for i in range(max_retries):
            if check_backend_running():
                print("✅ Backend is ready!")
                return backend_process
            time.sleep(1)
            if i % 5 == 0 and i > 0:
                print(f"⏳ Still waiting... ({i}s)")
        
        print("❌ Backend failed to start within 30 seconds")
        return None
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_frontend():
    """Start the Streamlit frontend"""
    print("🎨 Starting Streamlit frontend...")
    try:
        # Start the frontend with localhost binding
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

def main():
    """Main function to orchestrate the startup"""
    print("🔍 H-MaMa Fake News Detection System")
    print("=" * 50)
    
    # Check if backend is already running
    if check_backend_running():
        print("✅ Backend is already running on http://localhost:8000")
    else:
        print("⚠️  Backend not detected. Starting backend...")
        backend_process = start_backend()
        
        if backend_process is None:
            print("❌ Failed to start backend. Please start it manually:")
            print("   uvicorn src.hmama.serve.api:app --reload --host 0.0.0.0 --port 8000")
            return
        
        # Check again if backend is now running
        if not check_backend_running():
            print("❌ Backend failed to start properly. Please check the logs.")
            return
    
    print("✅ Backend is ready!")
    print("🌐 Frontend will be available at: http://localhost:8501")
    print("=" * 50)
    
    # Start the frontend
    start_frontend()

if __name__ == "__main__":
    main()
