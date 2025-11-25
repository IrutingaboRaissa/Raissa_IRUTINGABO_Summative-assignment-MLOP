import requests
from pathlib import Path

print("\n=== TESTING PREDICTION ===")

# Find a test image
img_path = None
for loc in ["archive", "data/test", "data/train"]:
    path = Path(loc)
    if path.exists():
        images = list(path.rglob("*.jpg"))
        if images:
            img_path = images[0]
            break

if not img_path:
    print("ERROR: No test images found")
    exit(1)

print(f"Using image: {img_path.name}")

# Test prediction
try:
    with open(img_path, 'rb') as f:
        response = requests.post(
            'http://localhost:8000/predict',
            files={'file': (img_path.name, f, 'image/jpeg')},
            timeout=10
        )
    
    if response.status_code == 200:
        result = response.json()
        print("\nPREDICTION SUCCESSFUL!")
        print(f"  Predicted Class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']*100:.2f}%")
        print(f"  Processing Time: {result['processing_time_seconds']:.3f}s")
        print("\nSTATUS: ✓ PASSED")
    else:
        print(f"ERROR: Status {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"ERROR: {e}")
