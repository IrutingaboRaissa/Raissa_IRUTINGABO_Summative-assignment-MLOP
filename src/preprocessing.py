"""
Data preprocessing module for skin cancer classification
"""
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import kagglehub


# Image transformations for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Image transformations for validation/testing
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class SkinCancerDataset(Dataset):
    """Custom Dataset for Skin Cancer Images from Kaggle HAM10000"""
    
    def __init__(self, data, transform=None):
        """
        Args:
            data: pandas DataFrame with 'path' and 'label_encoded' columns
            transform: torchvision transforms
        """
        self.data = data
        self.transform = transform
        
        # Check if it's a pandas DataFrame
        self.is_dataframe = isinstance(data, pd.DataFrame)
        
        if self.is_dataframe:
            self.data = data.reset_index(drop=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Load image from DataFrame (Kaggle HAM10000 dataset)
        img_path = self.data.iloc[idx]['path']
        label = self.data.iloc[idx]['label_encoded']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def preprocess_image(image_path, transform=val_transform):
    """
    Preprocess a single image for prediction
    
    Args:
        image_path: path to image or PIL Image
        transform: transformation pipeline
    
    Returns:
        preprocessed image tensor
    """
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def load_dataset_from_kaggle():
    """
    Load HAM10000 dataset from Kaggle using kagglehub
    
    Returns:
        path to the downloaded dataset
    """
    # Download latest version
    path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    
    print("Path to dataset files:", path)
    return path


def create_dataloaders_from_uploaded(uploaded_dir='data/uploaded', batch_size=32):
    """
    Create PyTorch DataLoaders from uploaded images in directory
    
    Args:
        uploaded_dir: directory containing uploaded images
        batch_size: batch size for training
    
    Returns:
        DataLoader for uploaded data
    """
    # Scan uploaded directory for images
    uploaded_path = Path(uploaded_dir)
    if not uploaded_path.exists():
        return None
    
    image_files = list(uploaded_path.glob('*.jpg')) + list(uploaded_path.glob('*.png'))
    
    if len(image_files) == 0:
        return None
    
    # Create DataFrame (assuming labels from filenames or metadata)
    # For retraining, you'll need to implement label extraction
    data = []
    for img_path in image_files:
        # Extract label from filename or metadata file
        # For now, using placeholder - implement based on your upload format
        data.append({
            'path': str(img_path),
            'label_encoded': 0  # TODO: Extract actual label
        })
    
    df = pd.DataFrame(data)
    dataset = SkinCancerDataset(df, transform=train_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return loader
