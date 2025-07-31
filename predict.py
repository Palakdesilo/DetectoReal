# predict.py

import torch # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
import os
import numpy as np
from model import SimpleCNN 

# === CONFIG ===
MODEL_PATH = 'model.pth'
CLASS_NAMES = ['fake', 'real']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ONCE ===
model = None

def load_model():
    """Load the model with proper error handling for Streamlit Cloud"""
    global model
    
    if model is not None:
        return model
    
    try:
        # Try to find the model file in different possible locations
        possible_paths = [
            MODEL_PATH,
            os.path.join(os.getcwd(), MODEL_PATH),
            os.path.join(os.path.dirname(__file__), MODEL_PATH),
            '/mount/src/detectoreal/model.pth',  # Streamlit Cloud path
            '/app/model.pth',  # Alternative Streamlit Cloud path
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(f"Model file not found. Tried paths: {possible_paths}")
        
        print(f"Loading model from: {model_path}")
        
        model = SimpleCNN(num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        print("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

# Initialize model
try:
    model = load_model()
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_image(image_path):
    if model is None:
        raise RuntimeError("Model not loaded. Please check if model.pth exists.")
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item()

    return predicted_class, confidence_score

def predict_image_from_pil(image):
    """
    Predict from a PIL Image object without saving to disk
    Returns: (prediction, confidence_score)
    """
    if model is None:
        raise RuntimeError("Model not loaded. Please check if model.pth exists.")
    
    image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item()

    return predicted_class, confidence_score

def get_detailed_analysis(image):
    """
    Get detailed analysis including confidence scores for both classes
    Returns: dict with prediction details
    """
    if model is None:
        raise RuntimeError("Model not loaded. Please check if model.pth exists.")
    
    image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        
        # Get confidence scores for both classes
        fake_confidence = probabilities[0][0].item()
        real_confidence = probabilities[0][1].item()
        
        # Determine prediction and confidence
        if fake_confidence > real_confidence:
            prediction = "fake"
            confidence = fake_confidence
        else:
            prediction = "real"
            confidence = real_confidence
        
        # Determine confidence level
        if confidence > 0.8:
            confidence_level = "High"
        elif confidence > 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Generate analysis message
        if prediction == "fake":
            if confidence > 0.8:
                message = "Strong indicators of AI generation detected"
            elif confidence > 0.6:
                message = "Moderate signs of AI generation found"
            else:
                message = "Weak indicators of AI generation"
        else:
            if confidence > 0.8:
                message = "Strong indicators of authentic image"
            elif confidence > 0.6:
                message = "Moderate signs of authentic image"
            else:
                message = "Weak indicators of authentic image"

    return {
        "prediction": prediction,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "fake_confidence": fake_confidence,
        "real_confidence": real_confidence,
        "message": message
    }
