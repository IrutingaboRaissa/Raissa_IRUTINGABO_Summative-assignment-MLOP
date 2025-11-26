from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
import uvicorn
import time
import os
from datetime import datetime
import shutil
from pathlib import Path

from src.prediction import ModelPredictor
from src.preprocessing import create_dataloaders_from_uploaded
from src.model import retrain_model, save_model
import pickle

# Initialize FastAPI app
app = FastAPI(
    title="Skin Cancer Classification API",
    description="API for skin cancer image classification with retraining capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL_PATH = "models/skin_cancer_classifier.pth"
METADATA_PATH = "models/model_metadata.pkl"
NUM_CLASSES = 7  # HAM10000 dataset has 7 classes
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
START_TIME = datetime.now()

# Auto-retrain configuration
AUTO_RETRAIN_ENABLED = True  # Enable automatic retraining
AUTO_RETRAIN_THRESHOLD = 10  # Trigger retraining after this many uploads
upload_count = 0  # Track number of uploads since last retrain

# Load class names from metadata
CLASS_NAMES = []
try:
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
        CLASS_NAMES = metadata.get('class_names', [])
        print(f"\n{'='*60}")
        print(f"LOADED CLASS NAMES: {CLASS_NAMES}")
        print(f"Number of classes: {len(CLASS_NAMES)}")
        print(f"{'='*60}\n")
except Exception as e:
    print(f"\n{'='*60}")
    print(f"ERROR: Could not load class names: {e}")
    CLASS_NAMES = [f"Class {i}" for i in range(NUM_CLASSES)]
    print(f"Using fallback: {CLASS_NAMES}")
    print(f"{'='*60}\n")

# Initialize predictor
try:
    predictor = ModelPredictor(MODEL_PATH, NUM_CLASSES, DEVICE)
    print(f"Model loaded successfully on {DEVICE}")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    predictor = None

# Retraining status
retraining_status = {
    "is_retraining": False,
    "last_retrain": None,
    "status_message": "No retraining performed yet"
}


class PredictionResponse(BaseModel):
    predicted_class: int
    class_name: str  # Added: disease class name (e.g., 'nv', 'mel', 'bcc')
    confidence: float
    probabilities: dict
    inference_time: float


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model_loaded: bool
    device: str


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Skin Cancer Classification API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint for monitoring model uptime"""
    uptime = (datetime.now() - START_TIME).total_seconds()
    
    return {
        "status": "healthy" if predictor is not None else "model_not_loaded",
        "uptime_seconds": uptime,
        "model_loaded": predictor is not None,
        "device": DEVICE
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    """
    Predict skin cancer class for a single image
    
    Args:
        file: Image file (JPEG, PNG)
    
    Returns:
        Prediction results with class, confidence, and probabilities
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type (check content_type if available, otherwise check filename)
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Also check file extension as backup
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"File must be an image. Allowed: {allowed_extensions}")
    
    try:
        print(f"\n{'='*60}")
        print(f"PREDICT ENDPOINT CALLED - file: {file.filename}")
        print(f"{'='*60}")
        
        # Read image bytes
        image_bytes = await file.read()
        print(f"Image bytes read: {len(image_bytes)} bytes")
        
        # Save uploaded image to predictions folder
        predictions_dir = Path("data/predictions")
        predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.filename).suffix
        saved_filename = f"{timestamp}_{file.filename}"
        saved_path = predictions_dir / saved_filename
        
        with open(saved_path, "wb") as buffer:
            buffer.write(image_bytes)
        
        # Make prediction
        start_time = time.time()
        result = predictor.predict(image_bytes)
        inference_time = time.time() - start_time
        
        # Add class name to result
        predicted_class_idx = result['predicted_class']
        print(f"API DEBUG - predicted_class_idx: {predicted_class_idx}")
        print(f"API DEBUG - CLASS_NAMES: {CLASS_NAMES}")
        print(f"API DEBUG - len(CLASS_NAMES): {len(CLASS_NAMES)}")
        
        if 0 <= predicted_class_idx < len(CLASS_NAMES):
            result['class_name'] = CLASS_NAMES[predicted_class_idx]
            print(f"API DEBUG - Set class_name to: {result['class_name']}")
        else:
            result['class_name'] = f"Class {predicted_class_idx}"
            print(f"API DEBUG - Out of range, using: {result['class_name']}")
        
        # Convert probabilities from 'Class 0', 'Class 1' format to actual class names
        if 'probabilities' in result:
            old_probs = result['probabilities']
            new_probs = {}
            for i in range(len(CLASS_NAMES)):
                class_key = f'Class {i}'
                if class_key in old_probs:
                    new_probs[CLASS_NAMES[i]] = old_probs[class_key]
            result['probabilities'] = new_probs
        
        result['inference_time'] = inference_time
        result['saved_path'] = str(saved_path)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict skin cancer class for multiple images
    
    Args:
        files: List of image files
    
    Returns:
        List of predictions for each image
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            image_bytes = await file.read()
            start_time = time.time()
            result = predictor.predict(image_bytes)
            inference_time = time.time() - start_time
            
            # Add class name to result
            predicted_class_idx = result['predicted_class']
            print(f"\n[BATCH] Processing {file.filename}")
            print(f"[BATCH] predicted_class_idx: {predicted_class_idx}")
            print(f"[BATCH] CLASS_NAMES: {CLASS_NAMES}")
            print(f"[BATCH] len(CLASS_NAMES): {len(CLASS_NAMES)}")
            
            if 0 <= predicted_class_idx < len(CLASS_NAMES):
                class_name = CLASS_NAMES[predicted_class_idx]
                print(f"[BATCH] Set class_name to: {class_name}")
            else:
                class_name = f"Class {predicted_class_idx}"
                print(f"[BATCH] Out of range, using: {class_name}")
            
            # Convert probabilities from 'Class 0', 'Class 1' format to actual class names
            old_probs = result['probabilities']
            new_probs = {}
            for i in range(len(CLASS_NAMES)):
                class_key = f'Class {i}'
                if class_key in old_probs:
                    new_probs[CLASS_NAMES[i]] = old_probs[class_key]
            
            results.append({
                "filename": file.filename,
                "predicted_class": result['predicted_class'],
                "class_name": class_name,
                "confidence": result['confidence'],
                "probabilities": new_probs,
                "inference_time": inference_time
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"predictions": results, "total_images": len(files)}


@app.post("/upload/bulk", tags=["Data Management"])
async def upload_bulk_data(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None, auto_retrain: bool = True):
    """
    Upload bulk data for retraining
    
    Args:
        files: List of image files to add to training data
        auto_retrain: Whether to automatically trigger retraining (default: True)
    
    Returns:
        Upload status and retraining trigger information
    """
    global upload_count
    
    upload_dir = Path("data/uploaded")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_files = []
    
    for file in files:
        try:
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            uploaded_files.append(str(file_path))
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to upload {file.filename}: {str(e)}"}
            )
    
    # Update upload count
    upload_count += len(uploaded_files)
    
    # Check if auto-retrain should be triggered
    should_auto_retrain = False
    retrain_triggered = False
    current_count = upload_count  # Store count before reset
    
    if AUTO_RETRAIN_ENABLED and auto_retrain:
        if upload_count >= AUTO_RETRAIN_THRESHOLD:
            should_auto_retrain = True
            upload_count = 0  # Reset counter after threshold reached
    
    response_data = {
        "message": "Files uploaded successfully",
        "uploaded_files": uploaded_files,
        "total_uploaded": len(uploaded_files),
        "cumulative_uploads": current_count,  # Show count before reset
        "current_count_after_reset": upload_count,  # Show count after reset
        "auto_retrain_enabled": AUTO_RETRAIN_ENABLED,
        "auto_retrain_threshold": AUTO_RETRAIN_THRESHOLD
    }
    
    # Trigger automatic retraining if threshold reached
    if should_auto_retrain and background_tasks:
        background_tasks.add_task(retrain_task)
        response_data["retrain_triggered"] = True
        response_data["retrain_message"] = f"Automatic retraining triggered (threshold: {AUTO_RETRAIN_THRESHOLD} uploads reached)"
        retrain_triggered = True
    else:
        response_data["retrain_triggered"] = False
        if AUTO_RETRAIN_ENABLED:
            remaining = AUTO_RETRAIN_THRESHOLD - upload_count
            response_data["retrain_message"] = f"Upload {remaining} more file(s) to trigger automatic retraining"
        else:
            response_data["retrain_message"] = "Automatic retraining is disabled. Use /retrain endpoint to retrain manually."
    
    return response_data


async def retrain_task():
    """Background task for model retraining"""
    global predictor, retraining_status
    
    try:
        retraining_status["is_retraining"] = True
        retraining_status["status_message"] = "Retraining in progress..."
        
        # Check if predictor and model exist
        if predictor is None or predictor.model is None:
            raise Exception("Model not loaded. Cannot retrain.")
        
        # Load uploaded data from data/uploaded directory
        new_data_loader = create_dataloaders_from_uploaded('data/uploaded', batch_size=32)
        
        if new_data_loader is None:
            raise Exception("No uploaded data found in data/uploaded directory")
        
        # Retrain model
        device = torch.device(DEVICE)
        retrained_model, metrics = retrain_model(
            predictor.model,
            new_data_loader,
            num_epochs=10,
            learning_rate=0.00005,
            device=device
        )
        
        # Save retrained model
        save_model(
            retrained_model,
            None,  # optimizer not needed for saving
            NUM_CLASSES,
            metrics['final_accuracy'],
            metrics['retrain_epochs'],
            MODEL_PATH
        )
        
        # Reload predictor
        predictor = ModelPredictor(MODEL_PATH, NUM_CLASSES, DEVICE)
        
        retraining_status["is_retraining"] = False
        retraining_status["last_retrain"] = datetime.now().isoformat()
        retraining_status["status_message"] = f"Retraining completed successfully. Accuracy: {metrics['final_accuracy']:.2f}%"
        
    except Exception as e:
        retraining_status["is_retraining"] = False
        retraining_status["status_message"] = f"Retraining failed: {str(e)}"


@app.post("/retrain", tags=["Model Management"])
async def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Trigger model retraining with new data
    
    Returns:
        Retraining status
    """
    if retraining_status["is_retraining"]:
        return JSONResponse(
            status_code=409,
            content={"error": "Retraining already in progress"}
        )
    
    background_tasks.add_task(retrain_task)
    
    return {
        "message": "Retraining task started",
        "status": "in_progress"
    }


@app.get("/retrain/status", tags=["Model Management"])
async def retrain_status():
    """Get current retraining status"""
    return retraining_status


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get model performance metrics"""
    uptime = (datetime.now() - START_TIME).total_seconds()
    
    return {
        "uptime_seconds": uptime,
        "uptime_hours": uptime / 3600,
        "model_loaded": predictor is not None,
        "device": DEVICE,
        "retraining_status": retraining_status
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
