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

def load_model(model_path=None):
    """Load the model with proper error handling for Streamlit Cloud"""
    global model
    
    if model is not None and model_path is None:
        return model
    
    try:
        # Use provided model_path or default to MODEL_PATH
        if model_path is None:
            model_path = MODEL_PATH
        
        # Try to find the model file in different possible locations
        possible_paths = [
            model_path,
            os.path.join(os.getcwd(), model_path),
            os.path.join(os.path.dirname(__file__), model_path),
            '/mount/src/detectoreal/model.pth',  # Streamlit Cloud path
            '/app/model.pth',  # Alternative Streamlit Cloud path
        ]
        
        found_path = None
        for path in possible_paths:
            if os.path.exists(path):
                found_path = path
                break
        
        if found_path is None:
            raise FileNotFoundError(f"Model file not found. Tried paths: {possible_paths}")
        
        print(f"Loading model from: {found_path}")
        
        model = SimpleCNN(num_classes=2)
        model.load_state_dict(torch.load(found_path, map_location=device))
        model.to(device)
        model.eval()
        
        print("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

def load_prediction_model(model_path=None):
    """Load the prediction model with optional model path"""
    try:
        return load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

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
    Get detailed analysis of an image including prediction, confidence, and processing time
    """
    import time
    
    start_time = time.time()
    
    try:
        # Get prediction
        prediction, confidence = predict_image_from_pil(image)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create detailed analysis
        analysis = {
            "prediction": prediction,
            "confidence": confidence,
            "processing_time": processing_time,
            "details": []
        }
        
        # Add confidence-based details
        if confidence > 0.9:
            analysis["details"].append(f"Very high confidence prediction ({confidence:.1%})")
        elif confidence > 0.7:
            analysis["details"].append(f"High confidence prediction ({confidence:.1%})")
        elif confidence > 0.5:
            analysis["details"].append(f"Moderate confidence prediction ({confidence:.1%})")
        else:
            analysis["details"].append(f"Low confidence prediction ({confidence:.1%})")
        
        # Add prediction-specific details
        if prediction == "fake":
            analysis["details"].append("AI-generated content detected")
            analysis["details"].append("Patterns consistent with synthetic image generation")
        else:
            analysis["details"].append("Natural image characteristics detected")
            analysis["details"].append("Patterns consistent with real photography")
        
        # Add processing details
        if processing_time < 0.1:
            analysis["details"].append("Fast processing time")
        elif processing_time < 0.5:
            analysis["details"].append("Normal processing time")
        else:
            analysis["details"].append("Slower processing time")
        
        return analysis
        
    except Exception as e:
        return {
            "prediction": "error",
            "confidence": 0.0,
            "processing_time": time.time() - start_time,
            "details": [f"Error during analysis: {str(e)}"]
        }
