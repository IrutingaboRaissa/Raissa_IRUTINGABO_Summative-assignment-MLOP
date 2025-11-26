# Deployment Guide

## 🚀 Quick Start - Run Everything Locally

### Prerequisites
- Docker and Docker Compose installed
- Git with Git LFS support
- 8GB+ RAM recommended

### Step 1: Clone and Setup
```bash
# Clone the repository
git clone https://github.com/IrutingaboRaissa/Raissa_IRUTINGABO_Summative-assignment-MLOP.git
cd Raissa_IRUTINGABO_Summative-assignment-MLOP

# Pull Git LFS files (models)
git lfs pull

# Verify model files exist
ls -lh models/
```

### Step 2: Run with Docker Compose (Easiest Method)
```bash
# Build and start all services (API, UI, Locust)
docker-compose up --build

# Or run in detached mode
docker-compose up -d

# Check running containers
docker-compose ps

# View logs
docker-compose logs -f
```

**Services will be available at:**
- 🌐 **Streamlit UI**: http://localhost:8501
- 🔌 **FastAPI Backend**: http://localhost:8000
- 📊 **API Documentation**: http://localhost:8000/docs
- 🐝 **Locust Load Testing**: http://localhost:8089

### Step 3: Run Without Docker (Manual Setup)

#### Terminal 1: Start API
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run API
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# API available at: http://localhost:8000
```

#### Terminal 2: Start UI
```bash
# Activate virtual environment (if not already)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run Streamlit
streamlit run ui/app.py

# UI available at: http://localhost:8501
```

#### Terminal 3: Start Locust (Load Testing)
```bash
# Activate virtual environment (if not already)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run Locust
locust -f tests/locustfile.py --host=http://localhost:8000

# Locust UI available at: http://localhost:8089
```

### Step 4: Test the Application

#### Test via UI (Streamlit)
1. Open http://localhost:8501
2. Upload a skin lesion image
3. Click "Analyze Image"
4. View prediction results

#### Test via API (FastAPI)
```bash
# Health check
curl http://localhost:8000/health

# Predict with curl
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"

# Or use the interactive docs
# Open: http://localhost:8000/docs
```

#### Load Testing with Locust
1. Open http://localhost:8089
2. Set number of users (e.g., 10)
3. Set spawn rate (e.g., 2 users/second)
4. Enter host: http://localhost:8000
5. Click "Start Swarming"
6. Monitor performance metrics

### Step 5: Stop Services

```bash
# If using Docker Compose
docker-compose down

# If using Docker Compose with volume cleanup
docker-compose down -v

# If running manually, press Ctrl+C in each terminal
```

### Troubleshooting Quick Start

**Issue: Port already in use**
```bash
# Find and kill process using port 8000
# On Linux/Mac:
lsof -ti:8000 | xargs kill -9

# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Change ports in docker-compose.yml if needed
```

**Issue: Model not found**
```bash
# Ensure Git LFS is installed
git lfs install

# Pull LFS files
git lfs pull

# Verify model files
ls -lh models/*.pth
```

**Issue: Docker build fails**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

**Issue: Out of memory**
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory (set to 6GB+)

# Or reduce batch size in code
```

### Performance Testing Results Example

After running Locust, you should see metrics like:
- **RPS (Requests Per Second)**: ~50-100 for API
- **Response Time**: ~200-500ms for predictions
- **Failure Rate**: <1%

### Video Tutorial

For a complete walkthrough, watch this setup video:
[Link to setup video tutorial - if available]

---

## 🌐 Live Deployment

**Current Deployment**: [https://raissa-irutingabo-summative-assignment.onrender.com](https://raissa-irutingabo-summative-assignment.onrender.com)

**Platform**: Render.com (Free Tier)
**Status**: ✅ Live and Operational
**Deployment Method**: Docker Container with Git LFS for model files

---

## Cloud Deployment Options

### Option 1: AWS EC2 Deployment

#### Step 1: Launch EC2 Instance
1. Go to AWS Console > EC2 > Launch Instance
2. Choose Ubuntu Server 22.04 LTS
3. Instance type: t2.medium (minimum) or t2.large (recommended)
4. Configure security group:
   - SSH (22)
   - HTTP (80)
   - Custom TCP (8000) for API
   - Custom TCP (8501) for UI

#### Step 2: Connect and Setup
```bash
# Connect to EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Reboot
sudo reboot
```

#### Step 3: Deploy Application
```bash
# Clone repository
git clone https://github.com/IrutingaboRaissa/Raissa_IRUTINGABO_Summative-assignment-MLOP.git
cd Raissa_IRUTINGABO_Summative-assignment-MLOP

# Build and run
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f
```

#### Step 4: Configure Nginx (Optional)
```bash
sudo apt install nginx -y

# Create Nginx config
sudo nano /etc/nginx/sites-available/skin-cancer-app

# Add configuration:
server {
    listen 80;
    server_name your-domain.com;

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/skin-cancer-app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Option 2: Google Cloud Platform (Cloud Run)

#### Step 1: Setup GCP Project
```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

#### Step 2: Build and Push Container
```bash
# Build for Cloud Run
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/skin-cancer-api

# Deploy API
gcloud run deploy skin-cancer-api \
  --image gcr.io/YOUR_PROJECT_ID/skin-cancer-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### Option 3: Azure App Service

#### Step 1: Create Resources
```bash
# Install Azure CLI
# https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# Login
az login

# Create resource group
az group create --name skin-cancer-rg --location eastus

# Create container registry
az acr create --resource-group skin-cancer-rg \
  --name skincancerregistry --sku Basic
```

#### Step 2: Build and Deploy
```bash
# Build and push
az acr build --registry skincancerregistry \
  --image skin-cancer-app:v1 .

# Create app service plan
az appservice plan create --name skin-cancer-plan \
  --resource-group skin-cancer-rg \
  --is-linux

# Deploy web app
az webapp create --resource-group skin-cancer-rg \
  --plan skin-cancer-plan \
  --name skin-cancer-app \
  --deployment-container-image-name skincancerregistry.azurecr.io/skin-cancer-app:v1
```

### Option 4: Heroku

#### Step 1: Setup
```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create skin-cancer-mlops
```

#### Step 2: Deploy
```bash
# Create heroku.yml
# Already configured in project

# Set stack to container
heroku stack:set container -a skin-cancer-mlops

# Deploy
git push heroku master

# Open app
heroku open -a skin-cancer-mlops
```

## Performance Optimization

### 1. Model Optimization
- Use model quantization
- Implement model caching
- Use ONNX for faster inference

### 2. API Optimization
- Enable response caching
- Implement connection pooling
- Use async operations

### 3. Container Optimization
- Use multi-stage Docker builds
- Minimize image size
- Use Alpine base images

## Monitoring Setup

### Prometheus & Grafana

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## Scaling Strategies

### Horizontal Scaling
```bash
# Scale API containers
docker-compose up --scale api=5

# With Kubernetes
kubectl scale deployment skin-cancer-api --replicas=5
```

### Load Balancing
- Use Nginx or HAProxy
- AWS ELB/ALB
- GCP Load Balancer

## Security Best Practices

1. **Use HTTPS**: Configure SSL/TLS certificates
2. **API Authentication**: Implement JWT or API keys
3. **Rate Limiting**: Prevent abuse
4. **Input Validation**: Validate all uploads
5. **Environment Variables**: Store secrets securely
6. **Regular Updates**: Keep dependencies updated

## Backup & Recovery

### Database Backup
```bash
# Backup uploaded data
tar -czf backup-$(date +%Y%m%d).tar.gz data/

# Backup models
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/
```

### Automated Backups
```bash
# Add to crontab
0 2 * * * /path/to/backup-script.sh
```

## Troubleshooting

### Common Issues

1. **Container won't start**
```bash
docker-compose logs api
docker-compose down && docker-compose up --build
```

2. **Out of memory**
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory
```

3. **Model not loading**
```bash
# Check model file exists
ls -lh models/
# Check permissions
chmod 644 models/*.pth
```

4. **API unreachable**
```bash
# Check if running
docker ps
# Check firewall
sudo ufw status
sudo ufw allow 8000
```

## Cost Optimization

### AWS
- Use Spot Instances
- Enable auto-scaling
- Use S3 for model storage

### GCP
- Use preemptible instances
- Implement Cloud Functions for light workloads
- Use Cloud Storage

### Azure
- Use reserved instances
- Implement Azure Functions
- Use Blob Storage

## CI/CD Pipeline

### GitHub Actions Example

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build and push Docker image
        run: |
          docker build -t skin-cancer-app .
          docker push your-registry/skin-cancer-app
      
      - name: Deploy to production
        run: |
          # Add deployment commands
```
