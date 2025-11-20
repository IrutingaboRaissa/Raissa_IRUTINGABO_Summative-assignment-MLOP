# Complete Running Guide

## System Architecture

```
main.py (Entry Point)
    |
    └─> orchestrator.py
            |
            ├─> api.py (FastAPI - Port 8000)
            |     └─> Uses: src/prediction.py, src/model.py, src/preprocessing.py
            |
            └─> app.py (Streamlit UI - Port 8501)
                  └─> Calls API endpoints
```

## Prerequisites

### 1. Train the Model First (REQUIRED)

**The model MUST be trained before running the application!**

Open and run the Jupyter notebook:
```powershell
jupyter notebook
# Navigate to: notebook/skin_cancer_dataset.ipynb
# Run all cells (Kernel > Restart & Run All)
```

This will:
- Download the HAM10000 dataset from Kaggle
- Train the ResNet50 model
- Save model to: `models/skin_cancer_classifier.pth`
- Save metadata to: `models/model_metadata.pkl`

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

## Running the Complete Application

### Method 1: Using main.py (Recommended)

```powershell
# From project root directory
python main.py
```

This will start:
- **API** at `http://localhost:8000`
- **UI** at `http://localhost:8501`

### Method 2: Using Docker

```powershell
docker-compose up --build
```

### Method 3: Manual (For Development)

**Terminal 1 - Start API:**
```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start UI:**
```powershell
streamlit run app.py --server.port 8501
```

## What Each Component Does

### 1. **main.py** (Entry Point)
- Launches both API and UI
- Handles graceful shutdown with Ctrl+C

### 2. **api.py** (Backend API)
Endpoints:
- `GET /` - Root/welcome
- `GET /health` - Health check & uptime
- `GET /metrics` - Performance metrics
- `POST /predict` - Single image prediction
- `POST /upload/bulk` - Upload multiple images
- `POST /retrain` - Trigger model retraining
- `GET /retrain/status` - Check retraining status

### 3. **app.py** (Frontend UI)
Pages:
- **Model Monitoring** - Uptime, health status
- **Single Prediction** - Upload & predict one image
- **Visualizations** - Dataset insights
- **Bulk Upload & Retrain** - Upload data and trigger retraining

### 4. **src/** (Core Modules)
- `model.py` - Model architecture, training, retraining
- `preprocessing.py` - Data loading, augmentation, transforms
- `prediction.py` - Inference functions
- `orchestrator.py` - Service orchestration

### 5. **notebook/** (Development & Training)
- `skin_cancer_dataset.ipynb` - Complete ML pipeline with visualizations

## Verification Steps

### 1. Check Model Exists
```powershell
ls models/
# Should see: skin_cancer_classifier.pth, model_metadata.pkl
```

If not, **run the notebook first!**

### 2. Test API
```powershell
# After starting main.py
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "uptime_seconds": 123.45,
  "model_loaded": true,
  "device": "cuda" or "cpu"
}
```

### 3. Test UI
Open browser: `http://localhost:8501`

Should see the dashboard with navigation sidebar.

## Common Issues

### Issue 1: "Model file not found"
**Solution:** Run the notebook first to train the model.

### Issue 2: "Module not found"
**Solution:** Make sure you're in the project root directory and run:
```powershell
pip install -r requirements.txt
```

### Issue 3: "Port already in use"
**Solution:** Change ports in main.py:
```python
orchestrator = MLOpsOrchestrator(api_port=8001, ui_port=8502)
```

### Issue 4: API not loading model
**Solution:** Check that `models/` directory exists with the `.pth` file.

## Load Testing

After the application is running:

```powershell
# Terminal 3
locust -f locustfile.py --host=http://localhost:8000

# Open browser: http://localhost:8089
# Set: Users=10, Spawn rate=1
# Start swarming
```

## Complete Workflow

1. **Train Model** (One-time setup)
   ```powershell
   jupyter notebook
   # Run: notebook/skin_cancer_dataset.ipynb
   ```

2. **Start Application**
   ```powershell
   python main.py
   ```

3. **Use the System**
   - Open UI: `http://localhost:8501`
   - Make predictions
   - Upload bulk data
   - Trigger retraining

4. **Monitor Performance**
   - Check metrics in UI
   - View logs in terminal
   - Run load tests with Locust

5. **Stop Application**
   - Press `Ctrl+C` in terminal
   - All services shut down gracefully

## File Connections Summary

```
main.py
    imports: src.orchestrator.MLOpsOrchestrator
    runs: orchestrator.run()

orchestrator.py
    starts: api.py (subprocess)
    starts: app.py (subprocess)

api.py
    imports: src.prediction.ModelPredictor
    imports: src.model.retrain_model, save_model
    imports: src.preprocessing.create_dataloaders
    uses: models/skin_cancer_classifier.pth

app.py
    calls: api.py endpoints via HTTP
    displays: predictions, metrics, visualizations

notebook/skin_cancer_dataset.ipynb
    imports: src.model (SkinCancerClassifier, train_model, retrain_model)
    imports: src.preprocessing (SkinCancerDataset, transforms)
    imports: src.prediction (predict_single_image)
    creates: models/skin_cancer_classifier.pth
    creates: models/model_metadata.pkl
```

## Everything is Connected! ✓

Yes, everything flows through `main.py`:
1. `main.py` → starts orchestrator
2. Orchestrator → launches API & UI as subprocesses
3. API → loads model and serves predictions
4. UI → calls API for all operations
5. Both API & UI → use functions from `src/` modules
6. Notebook → uses same `src/` modules for training

**Single command to run everything:** `python main.py`
