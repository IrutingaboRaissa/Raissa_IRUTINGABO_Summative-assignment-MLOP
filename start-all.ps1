# Start all services for Skin Cancer Detection App

Write-Host "🚀 Starting Skin Cancer Detection App..." -ForegroundColor Green

# Check if virtual environment exists
if (!(Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Start API in background
Write-Host "Starting API on http://localhost:8000..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; venv\Scripts\activate; cd api; python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

# Wait for API to start
Start-Sleep -Seconds 5

# Start UI in background
Write-Host "Starting UI on http://localhost:8501..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; venv\Scripts\activate; streamlit run ui/app.py"

# Wait for UI to start
Start-Sleep -Seconds 3

# Start Locust in background
Write-Host "Starting Locust on http://localhost:8089..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; venv\Scripts\activate; locust -f tests/locustfile.py --host=http://localhost:8000"

Write-Host "`n✅ All services started!" -ForegroundColor Green
Write-Host "📱 UI:      http://localhost:8501" -ForegroundColor Yellow
Write-Host "🔌 API:     http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "🐝 Locust:  http://localhost:8089" -ForegroundColor Yellow
Write-Host "`nPress Ctrl+C in each window to stop services" -ForegroundColor Gray
