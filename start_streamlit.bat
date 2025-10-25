@echo off
echo 🔍 H-MaMa Frontend Startup
echo ================================

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Try different Streamlit configurations
echo 🚀 Starting Streamlit frontend...
echo.

echo Trying configuration 1: Default settings
streamlit run frontend.py --server.port 8501
if %errorlevel% neq 0 (
    echo.
    echo Trying configuration 2: Explicit localhost
    streamlit run frontend.py --server.port 8501 --server.address 127.0.0.1
    if %errorlevel% neq 0 (
        echo.
        echo Trying configuration 3: Different port
        streamlit run frontend.py --server.port 8502
    )
)

pause
