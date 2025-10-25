# PowerShell script to start the backend with correct Python path
Write-Host "🔍 Starting H-MaMa Backend" -ForegroundColor Green
Write-Host "================================"

# Set the project directory
$projectDir = Get-Location
Write-Host "Project directory: $projectDir"

# Set PYTHONPATH environment variable
$env:PYTHONPATH = $projectDir
Write-Host "PYTHONPATH set to: $env:PYTHONPATH"

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "✅ Activating virtual environment..." -ForegroundColor Blue
    .\venv\Scripts\Activate.ps1
} else {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    exit 1
}

# Start the backend
Write-Host "🚀 Starting FastAPI backend..." -ForegroundColor Blue
Write-Host "Backend will be available at: http://localhost:8000" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the backend" -ForegroundColor Yellow
Write-Host ""

uvicorn src.hmama.serve.api:app --reload --host 0.0.0.0 --port 8000
