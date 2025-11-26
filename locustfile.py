# Locust load testing script for Skin Cancer Classification API
# Tests ALL API endpoints comprehensively

from locust import HttpUser, task, between
import random
import os
from io import BytesIO
from PIL import Image
import json


class ComprehensiveAPIUser(HttpUser):
    """Comprehensive testing of ALL API endpoints"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Called when a user starts"""
        print("🚀 New user session started")
        # Create test images for predictions
        self.test_image = self.create_test_image()
        self.test_images_batch = [self.create_test_image() for _ in range(3)]
    
    def create_test_image(self, size=(224, 224)):
        """Create a realistic test image in memory"""
        # Create image with some variation (simulates real skin lesion images)
        img = Image.new('RGB', size, color=(
            random.randint(100, 200),  # Reddish skin tones
            random.randint(80, 150),
            random.randint(70, 130)
        ))
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG', quality=95)
        img_bytes.seek(0)
        return img_bytes.getvalue()
    
    # ========== ROOT & HEALTH ENDPOINTS ==========
    
    @task(5)
    def root_endpoint(self):
        """Test root endpoint GET /"""
        with self.client.get("/", catch_response=True, name="GET /") as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "message" in data:
                        response.success()
                    else:
                        response.failure("Missing 'message' in response")
                except:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(10)
    def health_check(self):
        """Test health check endpoint GET /health"""
        with self.client.get("/health", catch_response=True, name="GET /health") as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    required_fields = ["status", "uptime_seconds", "model_loaded", "device"]
                    if all(field in data for field in required_fields):
                        if data["model_loaded"]:
                            response.success()
                        else:
                            response.failure("Model not loaded!")
                    else:
                        response.failure(f"Missing fields. Got: {data.keys()}")
                except Exception as e:
                    response.failure(f"JSON error: {e}")
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(8)
    def get_metrics(self):
        """Test metrics endpoint GET /metrics"""
        with self.client.get("/metrics", catch_response=True, name="GET /metrics") as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    required_fields = ["total_predictions", "avg_inference_time", 
                                     "cache_hit_rate", "model_version"]
                    if all(field in data for field in required_fields):
                        response.success()
                    else:
                        response.failure(f"Missing metrics fields")
                except Exception as e:
                    response.failure(f"JSON error: {e}")
            else:
                response.failure(f"Status: {response.status_code}")
    
    # ========== PREDICTION ENDPOINTS ==========
    
    @task(20)
    def predict_single_image(self):
        """Test single image prediction POST /predict"""
        files = {'file': ('test_image.jpg', BytesIO(self.test_image), 'image/jpeg')}
        
        with self.client.post("/predict", files=files, catch_response=True, 
                             name="POST /predict") as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    required_fields = ["prediction", "confidence", "probabilities", 
                                     "inference_time", "all_classes"]
                    
                    if all(field in data for field in required_fields):
                        # Validate prediction format
                        if isinstance(data["confidence"], (int, float)):
                            if 0 <= data["confidence"] <= 1:
                                response.success()
                            else:
                                response.failure(f"Invalid confidence: {data['confidence']}")
                        else:
                            response.failure("Confidence not a number")
                    else:
                        response.failure(f"Missing fields. Got: {data.keys()}")
                except Exception as e:
                    response.failure(f"Error: {e}")
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(5)
    def batch_predict(self):
        """Test batch prediction POST /predict/batch"""
        files = [
            ('files', ('test1.jpg', BytesIO(self.test_images_batch[0]), 'image/jpeg')),
            ('files', ('test2.jpg', BytesIO(self.test_images_batch[1]), 'image/jpeg')),
            ('files', ('test3.jpg', BytesIO(self.test_images_batch[2]), 'image/jpeg'))
        ]
        
        with self.client.post("/predict/batch", files=files, catch_response=True,
                             name="POST /predict/batch") as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "predictions" in data and len(data["predictions"]) == 3:
                        # Check each prediction has required fields
                        valid = all("prediction" in pred and "confidence" in pred 
                                  for pred in data["predictions"])
                        if valid:
                            response.success()
                        else:
                            response.failure("Invalid prediction format")
                    else:
                        response.failure(f"Expected 3 predictions, got {len(data.get('predictions', []))}")
                except Exception as e:
                    response.failure(f"Error: {e}")
            else:
                response.failure(f"Status: {response.status_code}")
    
    # ========== UPLOAD & RETRAIN ENDPOINTS ==========
    
    @task(3)
    def bulk_upload(self):
        """Test bulk upload POST /upload/bulk"""
        # Upload 5 images with labels
        files = []
        for i in range(5):
            img_bytes = BytesIO(self.create_test_image())
            files.append(('files', (f'upload_{i}.jpg', img_bytes, 'image/jpeg')))
        
        # Add labels
        labels = ['nv', 'mel', 'bkl', 'bcc', 'akiec']  # HAM10000 classes
        data = {'labels': ','.join(labels)}
        
        with self.client.post("/upload/bulk", files=files, data=data, 
                             catch_response=True, name="POST /upload/bulk") as response:
            if response.status_code == 200:
                try:
                    resp_data = response.json()
                    if "uploaded_count" in resp_data and resp_data["uploaded_count"] == 5:
                        response.success()
                    else:
                        response.failure(f"Upload count mismatch: {resp_data}")
                except Exception as e:
                    response.failure(f"Error: {e}")
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(1)
    def trigger_retrain(self):
        """Test retrain trigger POST /retrain"""
        payload = {
            "epochs": 2,  # Small number for testing
            "learning_rate": 0.001
        }
        
        with self.client.post("/retrain", json=payload, catch_response=True,
                             name="POST /retrain") as response:
            if response.status_code == 200 or response.status_code == 202:
                try:
                    data = response.json()
                    if "message" in data or "status" in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except Exception as e:
                    response.failure(f"Error: {e}")
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(2)
    def check_retrain_status(self):
        """Test retrain status GET /retrain/status"""
        with self.client.get("/retrain/status", catch_response=True,
                            name="GET /retrain/status") as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "status" in data:
                        response.success()
                    else:
                        response.failure("Missing status field")
                except Exception as e:
                    response.failure(f"Error: {e}")
            else:
                response.failure(f"Status: {response.status_code}")
    
    # ========== API DOCUMENTATION ENDPOINTS ==========
    
    @task(4)
    def api_docs(self):
        """Test API documentation GET /docs"""
        with self.client.get("/docs", catch_response=True, name="GET /docs") as response:
            if response.status_code == 200:
                if "swagger" in response.text.lower() or "openapi" in response.text.lower():
                    response.success()
                else:
                    response.failure("Invalid docs page")
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(3)
    def openapi_spec(self):
        """Test OpenAPI specification GET /openapi.json"""
        with self.client.get("/openapi.json", catch_response=True,
                            name="GET /openapi.json") as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "openapi" in data and "paths" in data:
                        response.success()
                    else:
                        response.failure("Invalid OpenAPI spec")
                except Exception as e:
                    response.failure(f"Error: {e}")
            else:
                response.failure(f"Status: {response.status_code}")


# ========== SPECIALIZED USER CLASSES ==========

class LightLoadUser(HttpUser):
    """Simulates light load - casual users"""
    wait_time = between(3, 7)
    
    @task(5)
    def health_check(self):
        self.client.get("/health")
    
    @task(3)
    def predict(self):
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        self.client.post("/predict", files=files)
    
    @task(1)
    def metrics(self):
        self.client.get("/metrics")


class HeavyLoadUser(HttpUser):
    """Simulates heavy load - rapid requests"""
    wait_time = between(0.1, 0.5)
    
    @task
    def rapid_predictions(self):
        img = Image.new('RGB', (224, 224), color='blue')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        self.client.post("/predict", files=files)


class StressTestUser(HttpUser):
    """Stress test - pushes system limits"""
    wait_time = between(0, 0.1)
    
    @task(10)
    def continuous_predict(self):
        img = Image.new('RGB', (224, 224), color=(100, 150, 200))
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'file': ('stress.jpg', img_bytes, 'image/jpeg')}
        self.client.post("/predict", files=files)
    
    @task(1)
    def health_check(self):
        self.client.get("/health")


"""
========================================
COMPLETE USAGE GUIDE
========================================

1. BASIC TEST (All endpoints):
   locust -f locustfile.py --host=http://localhost:8000
   Open: http://localhost:8089
   Users: 10, Spawn rate: 2

2. LIGHT LOAD:
   locust -f locustfile.py --host=http://localhost:8000 LightLoadUser --users 5 --spawn-rate 1

3. HEAVY LOAD:
   locust -f locustfile.py --host=http://localhost:8000 HeavyLoadUser --users 50 --spawn-rate 10

4. STRESS TEST:
   locust -f locustfile.py --host=http://localhost:8000 StressTestUser --users 100 --spawn-rate 20

5. HEADLESS MODE (No UI):
   locust -f locustfile.py --host=http://localhost:8000 --users 20 --spawn-rate 5 --run-time 3m --headless

6. GENERATE HTML REPORT:
   locust -f locustfile.py --host=http://localhost:8000 --users 50 --spawn-rate 5 --run-time 5m --html=report.html --headless

7. TEST SPECIFIC ENDPOINT ONLY:
   # Modify tasks: Comment out @task decorators for endpoints you don't want to test

8. DOCKER CONTAINER SCALING TEST:
   # Terminal 1: Scale API containers
   docker-compose up --scale api=3
   
   # Terminal 2: Run load test
   locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10

========================================
METRICS TO MONITOR IN LOCUST UI
========================================

✓ Total Requests
✓ Requests per Second (RPS)
✓ Response Time (min/avg/max/median)
✓ 50th/95th/99th Percentile
✓ Failure Rate (should be 0%)
✓ Request/Failures breakdown by endpoint

========================================
API ENDPOINTS TESTED
========================================

✅ GET  /                    - Root/Welcome
✅ GET  /health              - Health check & uptime
✅ GET  /metrics             - Performance metrics
✅ POST /predict             - Single image prediction
✅ POST /predict/batch       - Batch prediction
✅ POST /upload/bulk         - Bulk image upload
✅ POST /retrain             - Trigger retraining
✅ GET  /retrain/status      - Retraining status
✅ GET  /docs                - API documentation
✅ GET  /openapi.json        - OpenAPI specification

ALL 10 ENDPOINTS COVERED! 🎉
"""
