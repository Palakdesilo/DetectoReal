# test_model.py

import torch
from model import SimpleCNN
from utils import get_data_loaders

# === CONFIG ===
MODEL_PATH = 'model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD TEST DATA ===
_, _, test_loader = get_data_loaders(data_dir="./real_vs_fake", image_size=128, batch_size=32)

# === LOAD MODEL ===
model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === TEST EVALUATION ===
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")
