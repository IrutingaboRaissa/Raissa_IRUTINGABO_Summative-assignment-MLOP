"""
Quick script to check if model files exist
"""
import os
import sys

print("="*70)
print("CHECKING MODEL FILES")
print("="*70)

# Define possible model locations
model_locations = [
    "models/skin_cancer_classifier.pth",
    "models/skin_cancer_model.pth",
    "models/model.pth",
    "../models/skin_cancer_classifier.pth",
    "src/../models/skin_cancer_classifier.pth"
]

found_models = []
missing_models = []

print("\nSearching for model files...\n")

for model_path in model_locations:
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        print(f"✅ FOUND: {model_path}")
        print(f"   Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
        found_models.append((model_path, size_mb))
    else:
        print(f"❌ NOT FOUND: {model_path}")
        missing_models.append(model_path)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if found_models:
    print(f"\n✅ Found {len(found_models)} model file(s):")
    for path, size in found_models:
        print(f"   - {path} ({size:.2f} MB)")
    
    print("\n🎉 MODEL EXISTS! Your API should work.")
    print("\nNext steps:")
    print("1. Start API: python main.py api")
    print("2. Check health: curl http://localhost:8000/health")
    print("3. Should show: 'model_loaded': true")
else:
    print(f"\n❌ No model files found!")
    print("\n⚠️ YOU NEED TO TRAIN THE MODEL FIRST!")
    print("\nSteps to fix:")
    print("1. Open Jupyter Notebook: jupyter notebook")
    print("2. Navigate to your training notebook")
    print("3. Run all cells to train the model")
    print("4. Verify model is saved to models/ directory")
    print("5. Run this script again to verify")

# Check if models directory exists
print("\n" + "="*70)
print("DIRECTORY CHECK")
print("="*70)

if os.path.exists("models"):
    print("\n✅ models/ directory exists")
    print("\nContents of models/ directory:")
    try:
        files = os.listdir("models")
        if files:
            for file in files:
                file_path = os.path.join("models", file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"   - {file} ({size:.2f} MB)")
        else:
            print("   (empty directory)")
    except Exception as e:
        print(f"   Error reading directory: {e}")
else:
    print("\n❌ models/ directory does NOT exist")
    print("   Creating it now...")
    os.makedirs("models", exist_ok=True)
    print("   ✅ Created models/ directory")

print("\n" + "="*70)

# Check if we're in the right directory
print("\nCurrent working directory:")
print(f"   {os.getcwd()}")

print("\n" + "="*70)
