# H-MaMa Fake News Detection System - PowerShell Launcher
Write-Host "🎯 H-MaMa Fake News Detection System" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host ""
Write-Host "Starting H-MaMa with one command..." -ForegroundColor Yellow
Write-Host ""

# Run the main Python script
python run_h_mama.py

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
