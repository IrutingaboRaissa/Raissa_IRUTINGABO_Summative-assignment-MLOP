# Locust load testing script for Skin Cancer Classification API
# Simulates flood of requests to test model performance

from locust import HttpUser, task, between
import random
import os
from io import BytesIO
from PIL import Image


class SkinCancerUser(HttpUser):
    """Simulates a user interacting with the API"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Called when a user starts"""
        # Create a dummy image for testing
        self.test_image = self.create_test_image()
    
    def create_test_image(self, size=(224, 224)):
        """Create a test image in memory"""
        img = Image.new('RGB', size, color=(random.randint(0, 255), 
                                            random.randint(0, 255), 
                                            random.randint(0, 255)))
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes.getvalue()
    
    @task(10)  # Weight: 10 (most common task)
    def predict_single_image(self):
        """Test single image prediction endpoint"""
        files = {'file': ('test_image.jpg', BytesIO(self.test_image), 'image/jpeg')}
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code: {response.status_code}")
    
    @task(5)  # Weight: 5
    def health_check(self):
        """Test health check endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code: {response.status_code}")
    
    @task(3)  # Weight: 3
    def get_metrics(self):
        """Test metrics endpoint"""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code: {response.status_code}")
    
    @task(1)  # Weight: 1 (less frequent)
    def batch_predict(self):
        """Test batch prediction endpoint"""
        files = [
            ('files', ('test1.jpg', BytesIO(self.test_image), 'image/jpeg')),
            ('files', ('test2.jpg', BytesIO(self.test_image), 'image/jpeg')),
            ('files', ('test3.jpg', BytesIO(self.test_image), 'image/jpeg'))
        ]
        
        with self.client.post("/predict/batch", files=files, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code: {response.status_code}")


# Custom scenarios for different load patterns

class LightLoadUser(HttpUser):
    """Simulates light load - few concurrent users"""
    wait_time = between(3, 5)
    
    @task
    def predict(self):
        test_image = Image.new('RGB', (224, 224), color='red')
        img_bytes = BytesIO()
        test_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        self.client.post("/predict", files=files)


class HeavyLoadUser(HttpUser):
    """Simulates heavy load - many rapid requests"""
    wait_time = between(0.1, 0.5)
    
    @task
    def predict(self):
        test_image = Image.new('RGB', (224, 224), color='blue')
        img_bytes = BytesIO()
        test_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        self.client.post("/predict", files=files)


"""
Usage Instructions:

1. Install Locust:
   pip install locust

2. Run basic test:
   locust -f locustfile.py --host=http://localhost:8000

3. Run with specific parameters:
   locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10

4. Run headless (no web UI):
   locust -f locustfile.py --host=http://localhost:8000 --users 50 --spawn-rate 5 --run-time 5m --headless

5. Test different load scenarios:
   # Light load
   locust -f locustfile.py --host=http://localhost:8000 LightLoadUser --users 10 --spawn-rate 2
   
   # Heavy load
   locust -f locustfile.py --host=http://localhost:8000 HeavyLoadUser --users 200 --spawn-rate 20

6. Access Web UI:
   Open http://localhost:8089 in browser
   Configure number of users and spawn rate
   View real-time statistics and charts

7. Record results:
   locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 5m --html=locust_report.html

Metrics to Monitor:
- Response time (min, max, avg, median)
- Requests per second (RPS)
- Failure rate
- Number of concurrent users
- P50, P95, P99 latency percentiles
"""
