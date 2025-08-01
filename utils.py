# utils.py

import torch # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image
import os
import numpy as np

# === IMAGE TRANSFORMS ===
def get_transform(image_size=128):
    """Get the standard transform for image preprocessing"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# === DATA LOADING UTILITIES ===
def load_image(image_path):
    """Load and preprocess a single image"""
    try:
        image = Image.open(image_path).convert('RGB')
        transform = get_transform()
        tensor = transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def load_image_from_pil(pil_image):
    """Load and preprocess a PIL image"""
    try:
        transform = get_transform()
        tensor = transform(pil_image)
        return tensor.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error processing PIL image: {e}")
        return None

# === MODEL UTILITIES ===
def load_model(model_path, device=None):
    """Load a PyTorch model from file"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        from model import SimpleCNN
        model = SimpleCNN(num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

# === PREDICTION UTILITIES ===
def predict_single_image(model, image_tensor, device=None):
    """Make prediction on a single image tensor"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            return {
                'prediction': 'real' if predicted_class == 0 else 'fake',
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy()
            }
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

# === DATA VALIDATION ===
def validate_image_file(image_path):
    """Validate if an image file exists and is readable"""
    if not os.path.exists(image_path):
        return False, "File does not exist"
    
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True, "Valid image file"
    except Exception as e:
        return False, f"Invalid image file: {e}"

def get_image_info(image_path):
    """Get basic information about an image"""
    try:
        with Image.open(image_path) as img:
            return {
                'size': img.size,
                'mode': img.mode,
                'format': img.format,
                'filename': os.path.basename(image_path)
            }
    except Exception as e:
        return None

# === BATCH PROCESSING ===
def process_batch(image_paths, model, device=None):
    """Process multiple images in batch"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {}
    for image_path in image_paths:
        image_tensor = load_image(image_path)
        if image_tensor is not None:
            prediction = predict_single_image(model, image_tensor, device)
            if prediction:
                results[image_path] = prediction
    
    return results

# === MEMORY MANAGEMENT ===
def clear_gpu_memory():
    """Clear GPU memory if available"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        return {
            'gpu_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'gpu_cached': torch.cuda.memory_reserved() / 1024**3,  # GB
        }
    return {'gpu_available': False} 








# # utils.py

# import os
# from torchvision import datasets, transforms # type: ignore
# from torch.utils.data import DataLoader # type: ignore

# def get_data_loaders(data_dir="./real_vs_fake", image_size=128, batch_size=32):
#     """
#     Load train, validation, and test datasets from the given directory.

#     Returns:
#         Tuple: (train_loader, val_loader, test_loader)
#     """
#     train_dir = os.path.join(data_dir, "train")
#     val_dir = os.path.join(data_dir, "validation")
#     test_dir = os.path.join(data_dir, "test")

#     train_transform = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])

#     common_transform = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])

#     train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
#     val_dataset = datasets.ImageFolder(val_dir, transform=common_transform)
#     test_dataset = datasets.ImageFolder(test_dir, transform=common_transform)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     print("✅ Classes:", train_dataset.classes)
#     print(f"📦 Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

#     return train_loader, val_loader, test_loader