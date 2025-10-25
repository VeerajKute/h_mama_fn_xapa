@echo off
echo 🔍 Starting H-MaMa Backend
echo ================================

REM Set the project directory as PYTHONPATH
set PYTHONPATH=%CD%
echo PYTHONPATH set to: %PYTHONPATH%

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    echo ✅ Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ❌ Virtual environment not found!
    pause
    exit /b 1
)

REM Start the backend
echo 🚀 Starting FastAPI backend...
echo Backend will be available at: http://localhost:8000
echo Press Ctrl+C to stop the backend
echo.

uvicorn src.hmama.serve.api:app --reload --host 0.0.0.0 --port 8000

pause
