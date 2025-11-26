"""
Diagnose why the API can't load the model
"""
import os
import sys
import torch
from pathlib import Path

print("="*70)
print("MODEL LOADING DIAGNOSTICS")
print("="*70)

# Check 1: Model file exists
print("\n1. CHECKING MODEL FILE EXISTENCE")
print("-"*70)

model_paths = [
    "models/skin_cancer_classifier.pth",
    "models/best_model.pth",
    "../models/skin_cancer_classifier.pth"
]

for path in model_paths:
    exists = os.path.exists(path)
    if exists:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"✅ {path} - {size_mb:.2f} MB")
    else:
        print(f"❌ {path} - NOT FOUND")

# Check 2: Try loading model
print("\n2. TESTING MODEL LOADING")
print("-"*70)

model_path = "models/skin_cancer_classifier.pth"

if os.path.exists(model_path):
    print(f"Attempting to load: {model_path}")
    
    try:
        # Try loading with weights_only=False (for PyTorch 2.6+)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✅ Model loaded successfully!")
        
        print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            print("✅ 'model_state_dict' found")
            print(f"   Number of parameters: {len(checkpoint['model_state_dict'])}")
        
        if 'num_classes' in checkpoint:
            print(f"✅ num_classes: {checkpoint['num_classes']}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"\nError type: {type(e).__name__}")
        
        # Try alternative loading methods
        print("\nTrying alternative loading methods...")
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            print("✅ Loaded with weights_only=True")
        except Exception as e2:
            print(f"❌ weights_only=True failed: {e2}")
            
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print("✅ Loaded with default settings")
        except Exception as e3:
            print(f"❌ Default loading failed: {e3}")
else:
    print(f"❌ {model_path} does not exist!")

# Check 3: Import API module and check model loading code
print("\n3. CHECKING API CODE")
print("-"*70)

try:
    sys.path.insert(0, os.getcwd())
    
    # Check if api.py or main.py exists
    if os.path.exists("api.py"):
        print("✅ api.py exists")
        with open("api.py", "r") as f:
            content = f.read()
            if "torch.load" in content:
                print("✅ Found torch.load() in api.py")
                # Extract the line
                for line in content.split('\n'):
                    if 'torch.load' in line and not line.strip().startswith('#'):
                        print(f"   Line: {line.strip()}")
            else:
                print("❌ No torch.load() found in api.py")
    else:
        print("❌ api.py not found")
        
    if os.path.exists("main.py"):
        print("✅ main.py exists")
    
except Exception as e:
    print(f"❌ Error checking API code: {e}")

# Check 4: Current directory
print("\n4. DIRECTORY INFORMATION")
print("-"*70)
print(f"Current directory: {os.getcwd()}")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Check 5: Provide fix
print("\n5. RECOMMENDED FIX")
print("-"*70)
print("""
The issue is likely in your API code (api.py or similar).

The model loading line should be:

    checkpoint = torch.load(
        model_path, 
        map_location=device,
        weights_only=False  # Required for PyTorch 2.6+ with pickle/numpy arrays
    )

Make sure:
1. The path is correct (relative to where API runs)
2. weights_only=False is set
3. Error handling is in place
""")

print("\n" + "="*70)
