"""
Script to check all API endpoints and diagnose issues
"""
import requests
import json
import os

API_URL = "http://localhost:8000"
# For deployed version:
# API_URL = "https://raissa-irutingabo-summative-assignment.onrender.com"

def check_endpoint(method, endpoint, data=None, files=None):
    """Check if an endpoint is working"""
    url = f"{API_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, files=files, timeout=10)
        
        print(f"✅ {method} {endpoint}")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            try:
                print(f"   Response: {json.dumps(response.json(), indent=2)}")
            except:
                print(f"   Response: {response.text[:200]}")
        else:
            print(f"   Error: {response.text[:200]}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"❌ {method} {endpoint}")
        print(f"   Error: Cannot connect to API")
        return False
    except Exception as e:
        print(f"❌ {method} {endpoint}")
        print(f"   Error: {str(e)}")
        return False

print("="*70)
print("CHECKING API ENDPOINTS")
print("="*70)
print(f"\nAPI URL: {API_URL}\n")

# Check root endpoint
print("\n1. ROOT ENDPOINT")
print("-"*70)
check_endpoint("GET", "/")

# Check health endpoint
print("\n2. HEALTH ENDPOINT")
print("-"*70)
check_endpoint("GET", "/health")

# Check metrics endpoint
print("\n3. METRICS ENDPOINT")
print("-"*70)
check_endpoint("GET", "/metrics")

# Check prediction endpoint (without actual image)
print("\n4. PREDICTION ENDPOINT")
print("-"*70)
print("Note: Skipping actual prediction (would need image file)")
print("To test: curl -X POST 'http://localhost:8000/predict' -F 'file=@image.jpg'")

# Check retrain status
print("\n5. RETRAIN STATUS ENDPOINT")
print("-"*70)
check_endpoint("GET", "/retrain/status")

# Check retrain endpoint (without triggering)
print("\n6. RETRAIN ENDPOINT")
print("-"*70)
print("Note: Skipping actual retrain (would take several minutes)")
print("To test: curl -X POST 'http://localhost:8000/retrain'")

# Check if model file exists locally
print("\n7. MODEL FILE CHECK")
print("-"*70)
model_paths = [
    "models/skin_cancer_classifier.pth",
    "models/skin_cancer_model.pth",
    "models/model.pth"
]

found_model = False
for path in model_paths:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"✅ Found: {path} ({size_mb:.1f} MB)")
        found_model = True
    else:
        print(f"❌ Not found: {path}")

if not found_model:
    print("\n⚠️ WARNING: No model file found!")
    print("   You need to train the model first.")
    print("   Run the training notebook: jupyter notebook")

# Check API documentation
print("\n8. API DOCUMENTATION")
print("-"*70)
check_endpoint("GET", "/docs")
check_endpoint("GET", "/openapi.json")

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

if not found_model:
    print("\n❌ ISSUE: Model file not found")
    print("\n   SOLUTION:")
    print("   1. Open Jupyter Notebook")
    print("   2. Navigate to your training notebook")
    print("   3. Run all cells to train the model")
    print("   4. Verify model is saved to models/ directory")
    print("   5. Restart the API")
else:
    print("\n✅ Model file exists")
    print("\n   If API still shows model_loaded=false:")
    print("   1. Check API logs for error messages")
    print("   2. Verify model path in API code")
    print("   3. Check file permissions")
    print("   4. Try restarting the API")

print("\n" + "="*70)
