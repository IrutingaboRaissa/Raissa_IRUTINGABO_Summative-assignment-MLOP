# Skin Cancer Classification System Launcher
# Run this with: .\start.ps1

Write-Host "`n=== SKIN CANCER CLASSIFICATION SYSTEM ===" -ForegroundColor Cyan
Write-Host "Starting all services...`n" -ForegroundColor Yellow

# Kill existing Python processes
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

# Start API in new window
Write-Host "[1/2] Starting API (http://localhost:8000)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; python api.py"
Start-Sleep -Seconds 8

# Start UI in new window
Write-Host "[2/2] Starting UI (http://localhost:8502)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; streamlit run app.py --server.port 8502"
Start-Sleep -Seconds 5

# Open browser
Write-Host "`n=== SYSTEM READY ===" -ForegroundColor Green
Write-Host "Opening browser..." -ForegroundColor Yellow
Start-Process "http://localhost:8502"

Write-Host "`nServices running in separate windows." -ForegroundColor Cyan
Write-Host "Press any key to exit this window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
