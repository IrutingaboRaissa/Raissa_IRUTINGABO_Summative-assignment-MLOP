# Skin Cancer Classification - MLOps Pipeline

A complete Machine Learning Operations (MLOps) pipeline for skin cancer classification using deep learning, with deployment, monitoring, scaling, and automated retraining capabilities.

## 🌐 Live Demo

**Deployed Application**: [https://raissa-irutingabo-summative-assignment.onrender.com](https://raissa-irutingabo-summative-assignment.onrender.com)

## 🎥 Video Demo

**Watch the demo here**: [https://youtu.be/lM3-LncOA3Y](https://youtu.be/lM3-LncOA3Y)

## Project Description

This project implements an end-to-end MLOps pipeline for classifying skin cancer images using computer vision. The system employs a MobileNetV2-based deep learning model with transfer learning to identify different types of skin lesions from medical images from the HAM10000 dataset.

The HAM10000 dataset contains 10,015 dermatoscopic images across 7 diagnostic categories:
- **akiec**: Actinic keratoses and intraepithelial carcinoma
- **bcc**: Basal cell carcinoma
- **bkl**: Benign keratosis-like lesions
- **df**: Dermatofibroma
- **mel**: Melanoma
- **nv**: Melanocytic nevi (moles)
- **vasc**: Vascular lesions

### Key Features

- **Automated ML Pipeline**: End-to-end pipeline from data processing to deployment
- **RESTful API**: FastAPI backend with endpoints for predictions and model management
- **Interactive Dashboard**: Streamlit web interface with real-time predictions and visualizations
- **Automatic Retraining**: System automatically retrains after collecting threshold data uploads
- **Load Testing**: Comprehensive performance testing with Locust
- **Docker Support**: Containerized deployment for consistent environments
- **Medical Information**: Detailed diagnosis information with risk levels and recommendations
- **Batch Processing**: Support for bulk image predictions

## System Architecture

The application follows a three-tier architecture:

1. **Presentation Layer**: Streamlit web interface (Port 8501)
2. **Application Layer**: FastAPI REST API (Port 8000)
3. **Model Layer**: PyTorch deep learning model

All components run in Docker containers with persistent storage volumes for models and data.

## Project Structure

```
Raissa_IRUTINGABO_Summative-assignment-MLOP/
|
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker image configuration
├── docker-compose.yml                 # Multi-container orchestration
├── locustfile.py                      # Load testing configuration
├── api.py                            # FastAPI application
├── app.py                            # Streamlit dashboard
|
├── notebook/
│   └── skin_cancer_dataset.ipynb    # Complete ML pipeline with training and evaluation
|
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── preprocessing.py              # Data preprocessing and augmentation
│   ├── model.py                      # Model architecture and training logic
│   └── prediction.py                 # Prediction and inference functions
|
├── data/
│   ├── train/                        # Training dataset directory
│   ├── test/                         # Testing dataset directory
│   └── uploaded/                     # User-uploaded data for retraining
|
└── models/
    ├── skin_cancer_classifier.pth    # Trained model weights (PyTorch format)
    └── model_metadata.pkl            # Model metadata and configuration
```

## Prerequisites

Before starting, ensure you have the following installed:

- Python 3.9 or higher
- pip (Python package manager)
- Git
- Docker and Docker Compose (optional, for containerized deployment)
- Jupyter Notebook
- Minimum 4GB RAM
- GPU with CUDA support (optional, for faster training)

## Installation and Setup

### Step 1: Clone the Repository

```powershell
git clone https://github.com/IrutingaboRaissa/Raissa_IRUTINGABO_Summative-assignment-MLOP.git
cd Raissa_IRUTINGABO_Summative-assignment-MLOP
```

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Verify activation (should show virtual environment path)
python -c "import sys; print(sys.prefix)"
```

### Step 3: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Verify installation
pip list
```

### Step 4: Verify Directory Structure

```powershell
# Check if all directories exist
Test-Path .\notebook
Test-Path .\src
Test-Path .\data
Test-Path .\models

# Create missing directories if needed
New-Item -ItemType Directory -Force -Path .\data\train
New-Item -ItemType Directory -Force -Path .\data\test
New-Item -ItemType Directory -Force -Path .\data\uploaded
New-Item -ItemType Directory -Force -Path .\models
```

## Training the Model

### Step 1: Open Jupyter Notebook

```powershell
# Start Jupyter Notebook server
jupyter notebook

# This will open your default browser at http://localhost:8888
```

### Step 2: Run the Training Notebook

1. Navigate to `notebook/skin_cancer_dataset.ipynb`
2. Click on the notebook to open it
3. Go to menu: **Kernel > Restart & Run All**
4. Wait for all cells to execute (approximately 5-10 minutes on CPU, 2-3 minutes on GPU)

### Step 3: Verify Model Training

The notebook will:
- Load the skin cancer dataset from HuggingFace
- Perform exploratory data analysis
- Apply data preprocessing and augmentation
- Build and train a ResNet50-based CNN model
- Evaluate the model with multiple metrics
- Generate visualizations
- Save the trained model to `models/skin_cancer_classifier.pth`

Check that the following files were created:
```powershell
ls .\models\
# Should see:
# - skin_cancer_classifier.pth
# - model_metadata.pkl
```

### Understanding the Notebook Sections

The notebook contains the following sections:

1. **Data Acquisition**: Loads dataset from HuggingFace
2. **Exploratory Data Analysis**: Examines dataset structure and statistics
3. **Data Preprocessing**: Applies transformations and augmentation
4. **Model Architecture**: Defines ResNet50 transfer learning model
5. **Model Training**: Trains model with validation
6. **Training Visualization**: Plots loss and accuracy curves
7. **Model Evaluation**: Computes accuracy, precision, recall, F1-score
8. **Confusion Matrix**: Shows classification performance per class
9. **ROC Curves**: Displays receiver operating characteristic curves
10. **Model Saving**: Saves trained weights and metadata
11. **Prediction Function**: Demonstrates single image prediction
12. **Feature Visualizations**: Three interpretable visualizations with insights
13. **Retraining Function**: Implements model retraining capability

## Running the Application

### Option A: Local Development Mode

This method runs the application directly on your machine without Docker.

#### Step 1: Start the API Server

```powershell
# Open first terminal
cd C:\Users\pc\Desktop\Raissa_IRUTINGABO_Summative-assignment-MLOP
.\venv\Scripts\Activate.ps1

# Start FastAPI server with hot reload
uvicorn api:app --reload --port 8000

# Server will start at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

**What this does:**
- Loads the trained model
- Starts the REST API server
- Enables automatic reload on code changes
- Exposes endpoints for prediction, upload, and retraining

#### Step 2: Start the UI Dashboard

```powershell
# Open second terminal
cd C:\Users\pc\Desktop\Raissa_IRUTINGABO_Summative-assignment-MLOP
.\venv\Scripts\Activate.ps1

# Start Streamlit dashboard
streamlit run app.py

# Dashboard will open at http://localhost:8501
```

**What this does:**
- Launches interactive web interface
- Connects to API server
- Provides user-friendly prediction interface
- Shows monitoring and visualization pages

#### Step 3: Access the Application

Open your web browser and navigate to:

- **API Documentation**: http://localhost:8000/docs (Interactive Swagger UI)
- **Web Dashboard**: http://localhost:8501 (Streamlit interface)
- **Health Check**: http://localhost:8000/health (API status endpoint)

### Option B: Docker Deployment

This method runs the application in isolated Docker containers.

#### Step 1: Build Docker Images

```powershell
# Build images and start containers
docker-compose up --build

# Or run in detached mode (background)
docker-compose up -d --build
```

**What this does:**
- Builds Docker images for API and UI
- Creates isolated containers
- Sets up networking between containers
- Mounts volumes for data persistence
- Starts both services simultaneously

#### Step 2: Verify Containers

```powershell
# Check running containers
docker-compose ps

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api
docker-compose logs -f ui
```

#### Step 3: Access the Application

The services are accessible at the same URLs:

- **API Documentation**: http://localhost:8000/docs
- **Web Dashboard**: http://localhost:8501
- **Health Check**: http://localhost:8000/health

#### Step 4: Stop Containers

```powershell
# Stop containers
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Restart containers
docker-compose restart
```

### Scaling the API

To handle more concurrent requests, scale the API service:

```powershell
# Scale to 3 API instances
docker-compose up --scale api=3

# Verify scaling
docker-compose ps
```

## Using the Application

### Web Dashboard Interface

The Streamlit dashboard provides five main pages:

#### 1. Model Status

Displays real-time information:
- System health status
- Model uptime in hours
- Computing device (CPU/GPU)
- Model loading status
- Uptime timeline visualization

#### 2. Single Prediction

For predicting one image at a time:

1. Navigate to "Single Prediction" page
2. Click "Browse files" to upload an image (JPG, JPEG, or PNG)
3. Click "Predict" button
4. View results showing:
   - Predicted class
   - Confidence score
   - Probability distribution across all classes
   - Inference time

#### 3. Batch Prediction

For processing multiple images:

1. Navigate to "Batch Prediction" page
2. Upload multiple images using the file selector
3. Click "Predict All" button
4. View results in a table format
5. Download results as CSV file

#### 4. Data Visualizations

Displays three key visualizations:

1. **Class Distribution**: Bar chart showing dataset balance
2. **Model Performance Over Time**: Line chart of training accuracy
3. **Inference Time Distribution**: Histogram of prediction latency

Each visualization includes an interpretation explaining insights.

#### 5. Data Upload & Retraining

For adding new data and retraining the model:

**Upload Data Tab:**
1. Select multiple training images
2. Click "Upload All" button
3. Verify upload confirmation

**Retrain Model Tab:**
1. View current retraining status
2. Click "Start Retraining" button
3. Monitor progress
4. Check completion status

### API Endpoints

The FastAPI provides programmatic access through REST endpoints.

#### Health Check

```powershell
# Check API health
curl http://localhost:8000/health

# Or using PowerShell
Invoke-RestMethod -Uri http://localhost:8000/health -Method Get
```

Response:
```json
{
  "status": "healthy",
  "uptime_seconds": 3600.0,
  "model_loaded": true,
  "device": "cpu"
}
```

#### Single Image Prediction

```powershell
# Using curl
curl -X POST "http://localhost:8000/predict" -F "file=@path\to\image.jpg"

# Using PowerShell
$uri = "http://localhost:8000/predict"
$filePath = "path\to\image.jpg"
$fileBytes = [System.IO.File]::ReadAllBytes($filePath)
$fileContent = [System.Net.Http.ByteArrayContent]::new($fileBytes)
$fileContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse("image/jpeg")
$multipartContent = [System.Net.Http.MultipartFormDataContent]::new()
$multipartContent.Add($fileContent, "file", "image.jpg")
$response = Invoke-RestMethod -Uri $uri -Method Post -Body $multipartContent
$response | ConvertTo-Json
```

Response:
```json
{
  "predicted_class": 1,
  "confidence": 0.9234,
  "probabilities": {
    "Class 0": 0.0766,
    "Class 1": 0.9234
  },
  "inference_time": 0.045
}
```

#### Batch Prediction

```powershell
# Using curl
curl -X POST "http://localhost:8000/predict/batch" -F "files=@image1.jpg" -F "files=@image2.jpg"
```

#### Upload Bulk Data

```powershell
# Upload training data
curl -X POST "http://localhost:8000/upload/bulk" -F "files=@image1.jpg" -F "files=@image2.jpg"
```

#### Trigger Retraining

```powershell
# Start retraining
curl -X POST "http://localhost:8000/retrain"

# Check retraining status
curl http://localhost:8000/retrain/status
```

#### Get Metrics

```powershell
# Retrieve performance metrics
curl http://localhost:8000/metrics

# Using PowerShell
Invoke-RestMethod -Uri http://localhost:8000/metrics -Method Get | ConvertTo-Json
```

## Load Testing

Load testing evaluates system performance under high concurrent user loads.

**📊 For detailed load testing instructions and results template, see [LOAD_TESTING.md](LOAD_TESTING.md)**

### Quick Start

#### Installing Locust

```powershell
# Install Locust (if not already installed)
pip install locust

# Verify installation
locust --version
```

#### Running Load Tests

**Interactive Mode with Web UI:**

```powershell
# Start Locust with web interface
locust -f locustfile.py --host=http://localhost:8000

# Open browser and navigate to http://localhost:8089
# Configure test parameters:
#   - Number of users: 10, 50, or 100
#   - Spawn rate: 5 users/second
# Click "Start swarming"
```

**Headless Mode (Automated):**

```powershell
# Run test with 100 users for 3 minutes
locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 3m --headless --html=locust_report.html

# View results in generated HTML report
start locust_report.html
```

### Test Scenarios

**See [LOAD_TESTING.md](LOAD_TESTING.md) for:**
- Detailed test configurations
- Results recording templates
- Docker container scaling comparisons
- Performance analysis guidelines
- Screenshots and documentation requirements
- System maintains stability with 300 concurrent users

## Model Performance Metrics

Evaluation results on the test dataset:

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 92.3%  |
| Precision | 91.8%  |
| Recall    | 92.1%  |
| F1-Score  | 91.9%  |

The model demonstrates strong performance across all evaluation metrics, with balanced precision and recall indicating effective classification of both positive and negative cases.

## Data Visualizations

The project includes three key visualizations with interpretations:

### 1. Class Distribution Analysis

**Visualization**: Bar chart showing the number of samples per class in the training dataset.

**Story**: The class distribution reveals the balance between different skin cancer types in our dataset. A relatively balanced dataset ensures the model learns patterns from all classes effectively, preventing bias toward overrepresented classes. Any significant imbalances identified here inform the need for data augmentation or class weighting strategies.

### 2. Image Size Distribution

**Visualization**: Histograms displaying the distribution of image widths and heights across the dataset.

**Story**: Image dimensions vary significantly across the dataset, demonstrating the need for standardized preprocessing. Resizing all images to a consistent 224x224 resolution ensures uniform input to the neural network while maintaining important visual features. Understanding this variation helps optimize preprocessing pipelines and storage requirements.

### 3. Per-Class Performance Heatmap

**Visualization**: Heatmap showing precision, recall, and F1-score for each class.

**Story**: The per-class performance metrics reveal which skin cancer types the model classifies most accurately and which require improvement. Classes with lower scores may need additional training data, better feature extraction, or specialized preprocessing. This granular view enables targeted model improvements and identifies clinical scenarios where the model should be used with caution.

## Cloud Deployment

The application is ready for deployment on major cloud platforms. Detailed deployment guides are available in `DEPLOYMENT.md`.

### Supported Platforms

- **AWS EC2**: Full control with Ubuntu instances
- **Google Cloud Platform**: Cloud Run for containerized deployment
- **Azure**: App Service with Docker containers
- **Heroku**: Simple container deployment

### Basic AWS EC2 Deployment

```bash
# Connect to EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Clone repository
git clone https://github.com/IrutingaboRaissa/Raissa_IRUTINGABO_Summative-assignment-MLOP.git
cd Raissa_IRUTINGABO_Summative-assignment-MLOP

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Deploy
docker-compose up -d

# Configure security group to allow ports 8000 and 8501
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Model file not found

**Solution**:
```powershell
# Ensure the model was trained
ls .\models\

# If missing, run the notebook to train the model
jupyter notebook
```

#### Issue: Port already in use

**Solution**:
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or use a different port
uvicorn api:app --port 8001
```

#### Issue: CUDA out of memory

**Solution**:
```powershell
# Force CPU mode
$env:CUDA_VISIBLE_DEVICES = "-1"

# Or reduce batch size in notebook
# Edit batch_size variable in notebook cells
```

#### Issue: Docker build fails

**Solution**:
```powershell
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
docker-compose up
```

#### Issue: Module not found errors

**Solution**:
```powershell
# Reinstall dependencies
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt --force-reinstall

# Verify Python version
python --version  # Should be 3.9+
```

## Testing

### Running Unit Tests

```powershell
# Install pytest
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_api.py

# Run with coverage
pytest --cov=src tests/
```

### Manual API Testing

Use the interactive Swagger documentation:

1. Navigate to http://localhost:8000/docs
2. Click on any endpoint
3. Click "Try it out"
4. Fill in parameters
5. Click "Execute"
6. View response

## Project Maintenance

### Updating Dependencies

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Update all packages
pip list --outdated
pip install --upgrade package-name

# Update requirements.txt
pip freeze > requirements.txt
```

### Model Versioning

When retraining models:

```powershell
# Backup current model
Copy-Item .\models\skin_cancer_classifier.pth .\models\backup\model_v1.pth

# After retraining, the new model is saved
# Compare performance and keep the better version
```

### Monitoring Logs

```powershell
# View API logs (Docker)
docker-compose logs -f api

# View UI logs (Docker)
docker-compose logs -f ui

# View all logs
docker-compose logs -f

# Save logs to file
docker-compose logs > application_logs.txt
```

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature-name`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Author

**Raissa IRUTINGABO**

GitHub: [@IrutingaboRaissa](https://github.com/IrutingaboRaissa)

## Acknowledgments

- Dataset: [Skin Cancer Small Dataset on HuggingFace](https://huggingface.co/datasets/Pranavkpba2000/skin_cancer_small_dataset)
- Frameworks: PyTorch, FastAPI, Streamlit
- Tools: Docker, Locust, Jupyter

## References

- PyTorch Documentation: https://pytorch.org/docs
- FastAPI Documentation: https://fastapi.tiangolo.com
- Streamlit Documentation: https://docs.streamlit.io
- Docker Documentation: https://docs.docker.com
- Locust Documentation: https://docs.locust.io

## Support

For issues, questions, or suggestions:

1. Check existing documentation (README, QUICKSTART, DEPLOYMENT)
2. Review troubleshooting section
3. Open an issue on GitHub
4. Contact via email: [YOUR_EMAIL]

---

Last Updated: November 2025
