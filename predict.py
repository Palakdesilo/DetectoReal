# predict.py

import torch # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
import os
import numpy as np
import random
from model import SimpleCNN 
from model_path_fix import ModelPathManager

# === DETERMINISTIC SETTINGS ===
def set_deterministic():
    """Set deterministic settings for reproducible results"""
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Set deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Apply deterministic settings
set_deterministic()

# === CONFIG ===
MODEL_PATH = 'model.pth'
CLASS_NAMES = ['fake', 'real']
# Force CPU usage for consistent results between local and cloud
device = torch.device("cpu")

# === LOAD MODEL ONCE ===
model = None
model_path_manager = ModelPathManager()

def load_model(model_path=None):
    """Load the model with proper error handling for Streamlit Cloud"""
    global model
    
    if model is not None and model_path is None:
        return model
    
    try:
        # Use the model path manager for robust loading
        model, status = model_path_manager.load_model(device)
        print(f"📊 Model loading status: {status}")
        return model
        
    except Exception as e:
        print(f"❌ Error in model loading: {e}")
        # Create a simple model as fallback
        print("🔄 Creating fallback model")
        model = SimpleCNN(num_classes=2)
        model.to(device)
        model.eval()
        return model

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
