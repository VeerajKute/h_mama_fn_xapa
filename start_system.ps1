# Comprehensive startup script for H-MaMa system
Write-Host "🔍 H-MaMa Fake News Detection System" -ForegroundColor Green
Write-Host "========================================"

# Set the project directory and PYTHONPATH
$projectDir = Get-Location
$env:PYTHONPATH = $projectDir
Write-Host "Project directory: $projectDir"
Write-Host "PYTHONPATH set to: $env:PYTHONPATH"

# Check if virtual environment exists
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please create one first: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "✅ Activating virtual environment..." -ForegroundColor Blue
.\venv\Scripts\Activate.ps1

# Install frontend dependencies if needed
Write-Host "📦 Checking frontend dependencies..." -ForegroundColor Blue
pip install -r requirements-frontend.txt -q

# Function to check if backend is running
function Test-Backend {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

# Check if backend is already running
if (Test-Backend) {
    Write-Host "✅ Backend is already running!" -ForegroundColor Green
} else {
    Write-Host "🚀 Starting backend..." -ForegroundColor Blue
    
    # Start backend in background
    $backendJob = Start-Job -ScriptBlock {
        param($projectDir)
        Set-Location $projectDir
        $env:PYTHONPATH = $projectDir
        uvicorn src.hmama.serve.api:app --reload --host 0.0.0.0 --port 8000
    } -ArgumentList $projectDir
    
    # Wait for backend to start
    Write-Host "⏳ Waiting for backend to start..." -ForegroundColor Yellow
    $timeout = 30
    $elapsed = 0
    
    do {
        Start-Sleep -Seconds 1
        $elapsed++
        if ($elapsed % 5 -eq 0) {
            Write-Host "⏳ Still waiting... ($elapsed s)" -ForegroundColor Yellow
        }
    } while (-not (Test-Backend) -and $elapsed -lt $timeout)
    
    if (Test-Backend) {
        Write-Host "✅ Backend started successfully!" -ForegroundColor Green
    } else {
        Write-Host "❌ Backend failed to start within $timeout seconds" -ForegroundColor Red
        Write-Host "Backend job output:" -ForegroundColor Yellow
        Receive-Job $backendJob
        Stop-Job $backendJob
        Remove-Job $backendJob
        exit 1
    }
}

# Start frontend
Write-Host "🎨 Starting Streamlit frontend..." -ForegroundColor Blue
Write-Host "Frontend will be available at: http://localhost:8501" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop both backend and frontend" -ForegroundColor Yellow
Write-Host ""

try {
    # Start Streamlit
    streamlit run frontend.py --server.port 8501 --server.address localhost
} catch {
    Write-Host "❌ Error starting frontend: $_" -ForegroundColor Red
} finally {
    # Clean up background job if it exists
    if ($backendJob) {
        Write-Host "🛑 Stopping backend..." -ForegroundColor Yellow
        Stop-Job $backendJob
        Remove-Job $backendJob
    }
}
