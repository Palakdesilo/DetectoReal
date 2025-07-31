# train.py

import torch
import torch.nn as nn
import torch.optim as optim

from model import SimpleCNN
from utils import get_data_loaders

# ==== CONFIGURATION ====
DATASET_PATH = './real_vs_fake'
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'model.pth'

# ==== DEVICE SETUP ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# ==== LOAD DATA ====
train_loader, val_loader, _ = get_data_loaders(DATASET_PATH, image_size=128, batch_size=BATCH_SIZE)

# ==== MODEL, LOSS, OPTIMIZER ====
model = SimpleCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==== TRAINING LOOP ====
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # ==== VALIDATION ====
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"📊 Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} - Val Accuracy: {accuracy:.2f}%")

# ==== SAVE MODEL ====
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"💾 Model saved to {MODEL_SAVE_PATH}")
