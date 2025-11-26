# Fix: Model Not Loaded Error

## Problem
API returns: `{"detail": "Model not loaded"}` with HTTP 503

## Solution for Render.com Deployment

### Step 1: Check if model file exists locally

```bash
ls -lh models/
# Should see: skin_cancer_classifier.pth (around 100-500MB)
```

If file doesn't exist, **run the training notebook first!**

### Step 2: Configure Git LFS (Large File Storage)

```bash
# Install Git LFS
git lfs install

# Track .pth files
git lfs track "*.pth"
git lfs track "*.pkl"

# Add .gitattributes
git add .gitattributes

# Check that model is tracked
git lfs ls-files
# Should show: models/skin_cancer_classifier.pth
```

### Step 3: Push model files to GitHub

```bash
# Add model files
git add models/

# Commit
git commit -m "Add model files via Git LFS"

# Push
git push origin main

# Verify upload
git lfs ls-files
```

### Step 4: Redeploy on Render

1. Go to Render Dashboard
2. Find your service
3. Click "Manual Deploy" → "Deploy latest commit"
4. Wait 5-10 minutes for build

### Step 5: Verify on Render

Check logs for:
