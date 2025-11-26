"""
Locust load testing for Skin Cancer Detection API
"""
from locust import HttpUser, task, between
import os
import random

class SkinCancerUser(HttpUser):
    """Simulates a user interacting with the Skin Cancer Detection API"""
    
    wait_time = between(1, 3)
    
    @task(3)
    def health_check(self):
        """Test the health endpoint"""
        self.client.get("/health")
    
    @task(2)
    def api_docs(self):
        """Access API documentation"""
        self.client.get("/docs")
    
    @task(1)
    def openapi_spec(self):
        """Get OpenAPI specification"""
        self.client.get("/openapi.json")
