# Fix: Model Not Loaded on Render.com

## Problem
Deployed API returns: `{"detail": "Model not loaded"}`

## Root Cause
The model file (`skin_cancer_classifier.pth`) is **too large** for Git and wasn't uploaded to Render.

## Solution: Use Git LFS

### Step 1: Install and Configure Git LFS

```bash
# Install Git LFS (if not already)
git lfs install

# Track large files
git lfs track "*.pth"
git lfs track "*.pkl"
git lfs track "*.h5"

# Add .gitattributes
git add .gitattributes
```

### Step 2: Add Model Files

```bash
# Check if model exists
ls -lh models/

# Add model to Git LFS
git add models/skin_cancer_classifier.pth
git add models/model_metadata.pkl  # if exists

# Commit
git commit -m "Add model files via Git LFS"

# Push
git push origin main
```

### Step 3: Configure Render for Git LFS

1. Go to Render Dashboard
2. Select your service
3. Go to **Environment** tab
4. Add environment variable:
   ```
   Key: GIT_LFS_ENABLED
   Value: 1
   ```
5. Click **Save Changes**

### Step 4: Manual Deploy

1. Go to **Manual Deploy**
2. Click **"Deploy latest commit"**
3. Wait 5-10 minutes for build

### Step 5: Verify

After deployment, check:
```bash
curl https://raissa-irutingabo-summative-assignment.onrender.com/health
```

Should return:
```json
{
  "status": "healthy",
  "model_loaded": true,  // ✅ Should be true!
  "device": "cpu"
}
```

## Alternative: Use Smaller Model or External Storage

If Git LFS doesn't work on Render free tier:

### Option A: Upload Model to Cloud Storage

```python
# In your API code, download model on startup
import requests
import os

def load_model():
    model_path = "models/skin_cancer_classifier.pth"
    
    if not os.path.exists(model_path):
        # Download from Dropbox/Google Drive
        print("Downloading model...")
        model_url = "YOUR_DIRECT_DOWNLOAD_URL"
        response = requests.get(model_url)
        
        os.makedirs("models", exist_ok=True)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded!")
    
    # Load model
    model = torch.load(model_path)
    return model
```

### Option B: Use Render Persistent Disk

1. Render Dashboard → Your Service
2. **Disks** tab
3. Add Disk (costs $1/month for 1GB)
4. Mount at `/data`
5. Upload model via SFTP

### Option C: Use Hugging Face Hub

```python
# Upload model to Hugging Face
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="models/skin_cancer_classifier.pth",
    path_in_repo="skin_cancer_classifier.pth",
    repo_id="your-username/skin-cancer-model",
    repo_type="model"
)

# Download in API
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="your-username/skin-cancer-model",
    filename="skin_cancer_classifier.pth"
)
```

## Check Render Logs

To see what's happening:

1. Render Dashboard → Your Service
2. **Logs** tab
3. Look for:
   ```
   ❌ Model file not found: models/skin_cancer_classifier.pth
   ```
   or
   ```
   ✅ Model loaded successfully
   ```

## Quick Test Commands

```bash
# 1. Check if Git LFS is working
git lfs ls-files

# Should show:
# models/skin_cancer_classifier.pth

# 2. Check file was pushed
git lfs pull

# 3. Verify file size locally
ls -lh models/*.pth

# 4. Test deployed API
curl https://raissa-irutingabo-summative-assignment.onrender.com/health
```

## If All Else Fails

**Deploy API locally and use ngrok** for testing:

```bash
# Terminal 1: Run API
python main.py api

# Terminal 2: Expose with ngrok
ngrok http 8000

# Update app.py with ngrok URL
API_URL = "https://your-ngrok-url.ngrok.io"
```

This way you can demo everything working locally but accessible online!
