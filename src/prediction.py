"""
Prediction module for skin cancer classification
"""
import torch
from PIL import Image
import numpy as np
from src.preprocessing import val_transform
from src.model import SkinCancerClassifier, load_model


def predict_single_image(model, image_path, transform, device, num_classes):
    """
    Predict class for a single image
    
    Args:
        model: trained PyTorch model
        image_path: path to image file or PIL Image
        transform: image transformation pipeline
        device: torch device
        num_classes: number of classes
    
    Returns:
        predicted_class: int
        confidence: float
        probabilities: dict
    """
    model.eval()
    
    # Load and preprocess image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)
    
    predicted_class = predicted.item()
    confidence_score = confidence.item()
    
    # Get all class probabilities
    prob_dict = {f'Class {i}': probabilities[0][i].item() for i in range(num_classes)}
    
    return predicted_class, confidence_score, prob_dict


def batch_predict(model, image_paths, transform, device, num_classes):
    """
    Predict classes for multiple images
    
    Args:
        model: trained PyTorch model
        image_paths: list of image paths
        transform: image transformation pipeline
        device: torch device
        num_classes: number of classes
    
    Returns:
        predictions: list of (class, confidence, probabilities)
    """
    model.eval()
    predictions = []
    
    for img_path in image_paths:
        try:
            pred_class, confidence, probs = predict_single_image(
                model, img_path, transform, device, num_classes
            )
            predictions.append({
                'image': img_path,
                'predicted_class': pred_class,
                'confidence': confidence,
                'probabilities': probs
            })
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            predictions.append({
                'image': img_path,
                'error': str(e)
            })
    
    return predictions


def predict_from_bytes(model, image_bytes, transform, device, num_classes):
    """
    Predict from image bytes (useful for API)
    
    Args:
        model: trained PyTorch model
        image_bytes: image as bytes
        transform: image transformation pipeline
        device: torch device
        num_classes: number of classes
    
    Returns:
        predicted_class, confidence, probabilities
    """
    from io import BytesIO
    
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    return predict_single_image(model, image, transform, device, num_classes)


class ModelPredictor:
    """Wrapper class for model prediction"""
    
    def __init__(self, model_path, num_classes, device='cpu'):
        """
        Initialize predictor
        
        Args:
            model_path: path to saved model
            num_classes: number of classes
            device: torch device
        """
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.model = load_model(model_path, num_classes, self.device)
        self.transform = val_transform
    
    def predict(self, image_input):
        """
        Make prediction
        
        Args:
            image_input: image path, PIL Image, or bytes
        
        Returns:
            dict with prediction results
        """
        if isinstance(image_input, bytes):
            pred_class, confidence, probs = predict_from_bytes(
                self.model, image_input, self.transform, self.device, self.num_classes
            )
        else:
            pred_class, confidence, probs = predict_single_image(
                self.model, image_input, self.transform, self.device, self.num_classes
            )
        
        return {
            'predicted_class': pred_class,
            'confidence': confidence,
            'probabilities': probs
        }
    
    def batch_predict(self, image_paths):
        """Predict for multiple images"""
        return batch_predict(
            self.model, image_paths, self.transform, self.device, self.num_classes
        )
