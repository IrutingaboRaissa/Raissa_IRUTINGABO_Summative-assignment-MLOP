# Stop all Python processes
Write-Host "`nStopping all Python processes..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

# Start API
Write-Host "Starting API..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; Write-Host '=== API WINDOW - WATCH FOR DEBUG OUTPUT ===' -ForegroundColor Cyan; python api.py"
Start-Sleep -Seconds 8

# Test API
Write-Host "`nTesting API..." -ForegroundColor Cyan
$test = Test-NetConnection localhost -Port 8000 -InformationLevel Quiet -WarningAction SilentlyContinue
if ($test) {
    Write-Host "✓ API is running on port 8000" -ForegroundColor Green
    
    # Make test prediction
    Write-Host "`nMaking test prediction..." -ForegroundColor Yellow
    curl.exe -X POST "http://localhost:8000/predict" -F "file=@data\uploaded\ISIC_0024311.jpg" 2>$null | ConvertFrom-Json | ConvertTo-Json -Depth 2
} else {
    Write-Host "✗ API failed to start" -ForegroundColor Red
}

Write-Host "`n`nNow check the API window for debug output!" -ForegroundColor Yellow
Write-Host "Then refresh your Streamlit page and try uploading an image.`n" -ForegroundColor Cyan
