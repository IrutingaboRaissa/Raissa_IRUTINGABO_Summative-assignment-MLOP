"""
Model architecture and training module for skin cancer classification
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
import pickle
import os


class SkinCancerClassifier(nn.Module):
    """CNN Model for Skin Cancer Classification using Transfer Learning with MobileNetV2"""
    
    def __init__(self, num_classes):
        super(SkinCancerClassifier, self).__init__()
        # Load pretrained MobileNetV2 (7x faster than ResNet50, optimized for speed)
        self.model = models.mobilenet_v2(pretrained=True)
        
        # Freeze early layers (feature extractor)
        for param in list(self.model.parameters())[:-30]:
            param.requires_grad = False
        
        # Replace final classifier layer
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, device='cpu'):
    """
    Train the model
    
    Args:
        model: PyTorch model
        train_loader: training data loader
        val_loader: validation data loader
        num_epochs: number of training epochs
        learning_rate: learning rate
        device: torch device
    
    Returns:
        model, training_history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            scheduler.step(val_loss)
            
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 60)
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print('-' * 60)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")
    
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'training_time': training_time
    }
    
    return model, history


def save_model(model, optimizer, num_classes, accuracy, epoch, model_path='../models/skin_cancer_classifier.pth'):
    """Save trained model"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_classes': num_classes,
        'accuracy': accuracy,
        'epoch': epoch,
    }, model_path)
    
    print(f"Model saved to {model_path}")


def load_model(model_path, num_classes, device='cpu'):
    """
    Load trained model
    
    Args:
        model_path: path to saved model
        num_classes: number of classes
        device: torch device
    
    Returns:
        loaded model
    """
    model = SkinCancerClassifier(num_classes).to(device)
    # PyTorch 2.6+ requires weights_only=False for pickle files with numpy arrays
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def retrain_model(model, new_data_loader, num_epochs=5, learning_rate=0.001, device='cpu'):
    """
    Retrain model with new data
    
    Args:
        model: existing trained model
        new_data_loader: DataLoader with new training data
        num_epochs: number of epochs for retraining
        learning_rate: learning rate
        device: torch device
    
    Returns:
        retrained model, metrics
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting model retraining for {num_epochs} epochs...")
    
    model.train()
    retrain_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for images, labels in new_data_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = epoch_loss / len(new_data_loader)
        accuracy = 100. * correct / total
        retrain_losses.append(avg_loss)
        print(f"Retrain Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
    
    metrics = {
        'final_loss': retrain_losses[-1],
        'final_accuracy': accuracy,
        'retrain_epochs': num_epochs
    }
    
    print(f"\nRetraining complete!")
    print(f"Final Accuracy: {accuracy:.2f}%")
    
    return model, metrics
