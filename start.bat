@echo off
echo 🔍 H-MaMa Fake News Detection System
echo ==================================================

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found. Please create one first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo ✅ Activating virtual environment...
call venv\Scripts\activate.bat

REM Install frontend dependencies
echo 📦 Installing frontend dependencies...
pip install -r requirements-frontend.txt

REM Start the system
echo 🚀 Starting system...
python run_frontend.py

pause
