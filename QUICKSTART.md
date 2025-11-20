# Quick Start Guide - Skin Cancer Classification System

This guide will get your system up and running in **15 minutes**.

## Prerequisites Checklist

- [ ] Python 3.9+ installed
- [ ] Git installed
- [ ] 4GB+ RAM available
- [ ] Internet connection (for first-time setup)

## Step-by-Step Setup

### 1. Clone and Navigate (2 minutes)

```powershell
git clone https://github.com/IrutingaboRaissa/Raissa_IRUTINGABO_Summative-assignment-MLOP.git
cd Raissa_IRUTINGABO_Summative-assignment-MLOP
```

### 2. Install Dependencies (3 minutes)

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### 3. Verify Model Exists (1 minute)

```powershell
# Check if model file exists
ls .\models\skin_cancer_classifier.pth

# If missing, you need to train the model first
# See section "Training Your First Model" below
```

### 4. Start the API (1 minute)

```powershell
# Open Terminal 1
python api.py

# Wait for message: "Uvicorn running on http://127.0.0.1:8000"
# Keep this terminal open!
```

### 5. Start the UI (1 minute)

```powershell
# Open Terminal 2 (NEW window)
.\venv\Scripts\Activate.ps1
streamlit run app.py

# Wait for message: "You can now view your Streamlit app in your browser"
# Browser should open automatically to http://localhost:8501
```

### 6. Test the System (2 minutes)

1. Open browser to: http://localhost:8501
2. Click "Single Prediction" in sidebar
3. Upload a test image from `data/test/` (if available) or any skin lesion image
4. Click "Predict"
5. View the diagnosis results!

## Training Your First Model (Optional - 30-60 minutes)

If model file doesn't exist, train it using the notebook:

1. Open Jupyter:
   ```powershell
   jupyter notebook
   ```

2. Navigate to `notebook/skin_cancer_dataset.ipynb`

3. Click **Cell > Run All**

4. Wait for training to complete (~30 mins on CPU, ~10 mins on GPU)

5. Model will be saved to `models/skin_cancer_classifier.pth`

6. Return to Step 4 above to start the API

## Common Issues & Solutions

### Issue: "ModuleNotFoundError"
**Solution:** Make sure virtual environment is activated and requirements installed:
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: "Model file not found"
**Solution:** Train the model using the notebook (see "Training Your First Model" above)

### Issue: "Port already in use"
**Solution:** 
```powershell
# For API (port 8000)
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process

# For UI (port 8501)
Get-Process -Id (Get-NetTCPConnection -LocalPort 8501).OwningProcess | Stop-Process
```

### Issue: "API is not responding"
**Solution:** Check API terminal for errors. Most common:
- Model file missing (train the model)
- Wrong Python version (use 3.9+)
- Dependencies not installed

## Next Steps

Once system is running:

1. **Explore Features:**
   - Model Status Dashboard
   - Single Image Prediction
   - Batch Prediction
   - Data Visualizations
   - Upload & Retrain

2. **Run Load Tests:** See [LOAD_TESTING.md](LOAD_TESTING.md)

3. **Deploy to Cloud:** See main README for deployment instructions

4. **Record Demo Video:** Test all features, then record your demonstration

## Quick Commands Reference

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Start API
python api.py

# Start UI (new terminal)
streamlit run app.py

# Run load test
locust -f locustfile.py --host=http://localhost:8000

# Stop all Python processes
Get-Process python | Stop-Process -Force
```

## System Architecture

```
┌─────────────┐     HTTP      ┌──────────────┐
│   Browser   │ ───────────> │  Streamlit   │
│  (User UI)  │               │  (Port 8501) │
└─────────────┘               └──────────────┘
                                      │
                                      │ REST API
                                      ▼
                              ┌──────────────┐
                              │   FastAPI    │
                              │  (Port 8000) │
                              └──────────────┘
                                      │
                                      │
                                      ▼
                              ┌──────────────┐
                              │  PyTorch     │
                              │  Model (.pth)│
                              └──────────────┘
```

## Support

For issues or questions:
1. Check the main [README.md](README.md) for detailed documentation
2. Review error messages in terminal
3. Ensure all prerequisites are met
4. Verify Python version: `python --version` (should be 3.9+)

---

**Ready to go? Start with Step 1!** 🚀
