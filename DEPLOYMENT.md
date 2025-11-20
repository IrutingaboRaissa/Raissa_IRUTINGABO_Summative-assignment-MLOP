# Deployment Guide

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
