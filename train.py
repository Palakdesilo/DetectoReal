# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from model import SimpleCNN

# === CONFIG ===
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DATA TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === DATA LOADING ===
def load_data():
    """Load training and validation data"""
    data_dir = "real_vs_fake"
    
    # Training data
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Validation data
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "validation"),
        transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

# === TRAINING FUNCTION ===
def train_model():
    """Train the CNN model"""
    print(f"Training on device: {DEVICE}")
    
    # Load data
    train_loader, val_loader = load_data()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize model
    model = SimpleCNN(num_classes=2)
    model.to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{EPOCHS}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Accuracy: {100*correct/total:.2f}%')
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'Training Loss: {running_loss/len(train_loader):.4f}, '
              f'Training Accuracy: {100*correct/total:.2f}%')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}, '
              f'Validation Accuracy: {100*val_correct/val_total:.2f}%')
        print('-' * 50)
    
    # Save model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved as 'model.pth'")
    
    return model

if __name__ == "__main__":
    print("Starting model training...")
    model = train_model()
    print("Training completed!") 