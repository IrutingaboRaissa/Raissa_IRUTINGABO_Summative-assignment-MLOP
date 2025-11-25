"""
Comprehensive Deployment Testing Script
Tests all components before production deployment
"""

import requests
import time
import sys
from pathlib import Path
from datetime import datetime

class DeploymentTester:
    def __init__(self, api_url="http://localhost:8000", ui_url="http://localhost:8502"):
        self.api_url = api_url
        self.ui_url = ui_url
        self.passed = []
        self.failed = []
        
    def print_header(self, text):
        print("\n" + "="*80)
        print(f"  {text}")
        print("="*80)
    
    def print_test(self, name, status, message=""):
        symbol = "✓" if status else "✗"
        status_text = "PASSED" if status else "FAILED"
        print(f"{symbol} {name}: {status_text}")
        if message:
            print(f"  → {message}")
        
        if status:
            self.passed.append(name)
        else:
            self.failed.append(name)
    
    def test_api_health(self):
        """Test 1: API Health Check"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            status = response.status_code == 200
            self.print_test("API Health Check", status, 
                          f"Response: {response.json()}" if status else f"Status: {response.status_code}")
            return status
        except Exception as e:
            self.print_test("API Health Check", False, str(e))
            return False
    
    def test_api_metrics(self):
        """Test 2: API Metrics Endpoint"""
        try:
            response = requests.get(f"{self.api_url}/metrics", timeout=5)
            status = response.status_code == 200
            if status:
                data = response.json()
                msg = f"Model: {data.get('model_info', {}).get('architecture', 'N/A')}"
            else:
                msg = f"Status: {response.status_code}"
            self.print_test("API Metrics Endpoint", status, msg)
            return status
        except Exception as e:
            self.print_test("API Metrics Endpoint", False, str(e))
            return False
    
    def test_model_prediction(self):
        """Test 3: Model Prediction"""
        try:
            # Find a test image - try multiple locations
            search_paths = [
                Path("data/test"),
                Path("archive/HAM10000_images_part_1"),
                Path("archive/HAM10000_images_part_2"),
                Path("data/train")
            ]
            
            image_files = []
            for test_image_path in search_paths:
                if test_image_path.exists():
                    image_files = list(test_image_path.glob("*.jpg"))
                    if image_files:
                        break
            
            if not image_files:
                self.print_test("Model Prediction", False, "No test images found in any location")
                return False
            
            # Test prediction with first image
            with open(image_files[0], 'rb') as f:
                files = {'file': (image_files[0].name, f, 'image/jpeg')}
                response = requests.post(f"{self.api_url}/predict", files=files, timeout=10)
            
            status = response.status_code == 200
            if status:
                data = response.json()
                msg = f"Predicted: {data.get('predicted_class', 'N/A')} ({data.get('confidence', 0)*100:.1f}%)"
            else:
                msg = f"Status: {response.status_code}"
            self.print_test("Model Prediction", status, msg)
            return status
        except Exception as e:
            self.print_test("Model Prediction", False, str(e))
            return False
    
    def test_ui_availability(self):
        """Test 4: UI Availability"""
        try:
            response = requests.get(self.ui_url, timeout=5)
            status = response.status_code == 200
            self.print_test("UI Availability", status, 
                          f"UI running at {self.ui_url}" if status else f"Status: {response.status_code}")
            return status
        except Exception as e:
            self.print_test("UI Availability", False, str(e))
            return False
    
    def test_file_structure(self):
        """Test 5: Required Files"""
        required_files = [
            "api.py",
            "app.py",
            "main.py",
            "requirements.txt",
            "Dockerfile",
            "docker-compose.yml",
            "README.md",
            "DEPLOYMENT.md",
            "LOAD_TESTING.md",
            "src/model.py",
            "src/preprocessing.py",
            "src/prediction.py",
            "models/best_model.pth"
        ]
        
        missing = []
        for file in required_files:
            if not Path(file).exists():
                missing.append(file)
        
        status = len(missing) == 0
        msg = "All files present" if status else f"Missing: {', '.join(missing)}"
        self.print_test("Required Files Check", status, msg)
        return status
    
    def test_model_files(self):
        """Test 6: Model Files"""
        model_dir = Path("models")
        if not model_dir.exists():
            self.print_test("Model Files", False, "models/ directory not found")
            return False
        
        model_files = list(model_dir.glob("*.pth"))
        status = len(model_files) > 0
        msg = f"Found {len(model_files)} model file(s)" if status else "No model files found"
        self.print_test("Model Files", status, msg)
        return status
    
    def test_dependencies(self):
        """Test 7: Python Dependencies"""
        try:
            import torch
            import torchvision
            import fastapi
            import streamlit
            import locust
            
            status = True
            msg = f"PyTorch {torch.__version__}, FastAPI, Streamlit, Locust installed"
            self.print_test("Python Dependencies", status, msg)
            return status
        except ImportError as e:
            self.print_test("Python Dependencies", False, f"Missing: {str(e)}")
            return False
    
    def test_response_time(self):
        """Test 8: Response Time Performance"""
        try:
            # Warm up request (first request is always slower)
            requests.get(f"{self.api_url}/health", timeout=5)
            time.sleep(0.5)
            
            # Actual test - average of 3 requests
            times = []
            for _ in range(3):
                start = time.time()
                response = requests.get(f"{self.api_url}/health", timeout=5)
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
                time.sleep(0.1)
            
            avg_time = sum(times) / len(times)
            status = avg_time < 1000  # Should respond in less than 1000ms average
            msg = f"{avg_time:.0f}ms average (threshold: 1000ms)"
            self.print_test("Response Time", status, msg)
            return status
        except Exception as e:
            self.print_test("Response Time", False, str(e))
            return False
    
    def test_data_directories(self):
        """Test 9: Data Directories"""
        required_dirs = ["data/train", "data/test", "data/uploaded"]
        missing = []
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except:
                    missing.append(dir_path)
        
        status = len(missing) == 0
        msg = "All directories exist" if status else f"Cannot create: {', '.join(missing)}"
        self.print_test("Data Directories", status, msg)
        return status
    
    def test_docker_files(self):
        """Test 10: Docker Configuration"""
        dockerfile = Path("Dockerfile")
        compose = Path("docker-compose.yml")
        
        status = dockerfile.exists() and compose.exists()
        msg = "Docker files present" if status else "Missing Docker configuration files"
        self.print_test("Docker Configuration", status, msg)
        return status
    
    def run_all_tests(self):
        """Run all deployment tests"""
        self.print_header("DEPLOYMENT READINESS TEST SUITE")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"API URL: {self.api_url}")
        print(f"UI URL: {self.ui_url}")
        
        # Infrastructure Tests
        self.print_header("INFRASTRUCTURE TESTS")
        self.test_file_structure()
        self.test_model_files()
        self.test_data_directories()
        self.test_docker_files()
        self.test_dependencies()
        
        # Runtime Tests
        self.print_header("RUNTIME TESTS")
        self.test_api_health()
        self.test_api_metrics()
        self.test_ui_availability()
        self.test_model_prediction()
        self.test_response_time()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        total = len(self.passed) + len(self.failed)
        pass_rate = (len(self.passed) / total * 100) if total > 0 else 0
        
        self.print_header("TEST SUMMARY")
        print(f"Total Tests: {total}")
        print(f"Passed: {len(self.passed)} ({pass_rate:.1f}%)")
        print(f"Failed: {len(self.failed)}")
        
        if self.failed:
            print("\nFailed Tests:")
            for test in self.failed:
                print(f"  - {test}")
        
        print("\n" + "="*80)
        
        if len(self.failed) == 0:
            print("SUCCESS: All tests passed! System is ready for deployment.")
            print("\nNext steps:")
            print("  1. Commit all changes: git add . && git commit -m 'Ready for deployment'")
            print("  2. Push to GitHub: git push origin streamlit")
            print("  3. Record video demo")
            print("  4. Deploy using Docker: docker-compose up -d")
        else:
            print("WARNING: Some tests failed. Please fix issues before deployment.")
            print("\nTo fix:")
            print("  - Ensure services are running: .\\start.ps1")
            print("  - Check missing files and dependencies")
            print("  - Run tests again: python test_deployment.py")
        
        print("="*80)
        
        return len(self.failed) == 0

def main():
    print("\n" + "="*80)
    print("  SKIN CANCER DETECTION SYSTEM - DEPLOYMENT TEST")
    print("="*80)
    
    tester = DeploymentTester()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
