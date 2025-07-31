# 🕵️‍♂️ DetectoReal

A deep learning-based web application that can distinguish between real and AI-generated images using a Convolutional Neural Network (CNN) built with PyTorch and Streamlit.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Training](#-training)
- [Testing](#-testing)
- [API Usage](#-api-usage)
- [Configuration](#-configuration)
- [Troubleshooting](#️-troubleshooting)
- [Model Performance](#-model-performance)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)
- [Support](#-support)

## 🎯 Overview

This project implements a binary classification model that can detect whether an image is real or AI-generated. The model uses a custom CNN architecture trained on a large dataset of real and AI-generated images. The application provides a user-friendly web interface built with Streamlit for easy image upload and prediction.

## ✨ Features

- **🎨 Modern UI**: Clean, responsive interface with custom styling and compact layout
- **📱 Drag & Drop**: Easy image upload with drag-and-drop support
- **⚡ Real-time Prediction**: Instant results with confidence indicators
- **🖼️ Multiple Formats**: Supports JPG, JPEG, and PNG images
- **📊 Visual Feedback**: Color-coded results (Green for Real, Red for Fake)
- **🔧 Easy Setup**: Simple installation and configuration
- **💾 Multiple Model Formats**: Support for PyTorch (.pth) and Keras (.keras/.h5) models
- **🛡️ Robust Error Handling**: Comprehensive error handling for image processing
- **⚙️ Direct Processing**: Images processed directly in memory without disk storage

## 📁 Project Structure

```
archive/
├── app.py                 # Main Streamlit web application
├── model.py              # CNN model architecture
├── predict.py            # Prediction functions
├── train.py              # Training script
├── utils.py              # Data loading utilities
├── retrain_model.py      # Model retraining with feedback
├── requirements.txt      # Python dependencies
├── model.pth            # Trained PyTorch model weights
├── real_vs_fake.keras   # Alternative Keras model
├── real_vs_fake.h5      # Alternative H5 model
├── real_vs_fake/        # Dataset directory
│   ├── train/
│   │   ├── real/        # 50,000 real training images
│   │   └── fake/        # 50,000 fake training images
│   ├── validation/
│   │   ├── real/        # 10,000 real validation images
│   │   └── fake/        # 10,000 fake validation images
│   └── test/
│       ├── real/        # 10,000 real test images
│       └── fake/        # 10,000 fake test images
└── myenv/               # Virtual environment
```

## 🔧 Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: Optional, for GPU acceleration (CUDA 11.0+ recommended)
- **Memory**: At least 4GB RAM (8GB recommended)
- **Storage**: 2GB free space for models and dependencies

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd archive
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv myenv
myenv\Scripts\activate

# macOS/Linux
python3 -m venv myenv
source myenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch, streamlit, PIL; print('✅ All dependencies installed successfully!')"
```

## 🎮 Usage

### Running the Web Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the application**:
   - Open your browser and go to `http://localhost:8501`
   - The application will load directly without password protection

3. **Upload and predict**:
   - Click "Browse files" or drag an image
   - Supported formats: JPG, JPEG, PNG
   - View the prediction result in the compact two-column layout

### Command Line Usage

For direct prediction without the web interface:

```bash
python predict.py
```

Then enter the full path to an image file when prompted.

## 🧠 Model Architecture

The model uses a **SimpleCNN** architecture with the following layers:

```
Input (3, 128, 128)
├── Conv2d(3→16, kernel=3, padding=1) + ReLU + MaxPool2d(2,2)
├── Conv2d(16→32, kernel=3, padding=1) + ReLU + MaxPool2d(2,2)
├── Conv2d(32→64, kernel=3, padding=1) + ReLU + MaxPool2d(2,2)
├── Flatten → Linear(64*16*16 → 128) + ReLU
└── Linear(128 → 2) → Output (Real/Fake)
```

### Model Specifications:
- **Input Size**: 128×128×3 RGB images
- **Output**: Binary classification (Real/Fake)
- **Parameters**: ~2.1M trainable parameters
- **Framework**: PyTorch (with Keras alternatives available)

## 📊 Dataset

The model is trained on a comprehensive dataset:

- **Training**: 100,000 images (50K real + 50K fake)
- **Validation**: 20,000 images (10K real + 10K fake)
- **Testing**: 20,000 images (10K real + 10K fake)
- **Total**: 140,000 images

### Dataset Structure:
```
real_vs_fake/
├── train/
│   ├── real/     # 50,000 real images
│   └── fake/     # 50,000 AI-generated images
├── validation/
│   ├── real/     # 10,000 real images
│   └── fake/     # 10,000 AI-generated images
└── test/
    ├── real/     # 10,000 real images
    └── fake/     # 10,000 AI-generated images
```

## 🎯 Training

### Training Configuration

```python
# Training Parameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = 128
```

### Start Training

```bash
python train.py
```

### Training Features:
- **Data Augmentation**: Random horizontal flip, rotation
- **Optimizer**: Adam optimizer
- **Loss Function**: CrossEntropyLoss
- **Device**: Automatic GPU/CPU detection
- **Progress Tracking**: Real-time loss and accuracy monitoring

## 🧪 Testing

### Evaluate Model Performance

```bash
python test_model.py
```

This will output the test accuracy on the held-out test set.

### Expected Performance:
- **Test Accuracy**: ~85-90% (varies based on training)
- **Inference Time**: <1 second per image (CPU)
- **Memory Usage**: ~500MB for model loading

## 🔌 API Usage

### Programmatic Prediction

```python
from predict import predict_image, predict_image_from_pil

# Predict from file path
result = predict_image("path/to/image.jpg")
print(f"Prediction: {result}")  # Output: "real" or "fake"

# Predict from PIL Image object
from PIL import Image
image = Image.open("path/to/image.jpg")
result = predict_image_from_pil(image)
print(f"Prediction: {result}")
```

### Batch Prediction

```python
import os
from predict import predict_image

def batch_predict(image_folder):
    results = {}
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, filename)
            prediction = predict_image(image_path)
            results[filename] = prediction
    return results
```

## 🔧 Configuration

### Model Path

Update the model path in `predict.py`:

```python
# Line 7 in predict.py
MODEL_PATH = 'path/to/your/model.pth'
```

### Image Size

Modify the input size in `predict.py`:

```python
# Line 18 in predict.py
transforms.Resize((256, 256)),  # Change from (128, 128)
```

### UI Customization

Modify the CSS styling in `app.py`:

```python
# Lines 7-35 in app.py
st.markdown("""
<style>
/* Custom CSS for styling */
</style>
""", unsafe_allow_html=True)
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size in retrain_model.py
   BATCH_SIZE = 16  # Instead of 32
   ```

2. **Model Loading Error**:
   ```bash
   # Ensure model.pth exists
   ls -la model.pth
   ```

3. **Streamlit Warnings**:
   ```bash
   # Use streamlit run instead of python
   streamlit run app.py  # ✅ Correct
   python app.py         # ❌ Wrong
   ```

4. **Import Errors**:
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

### Performance Optimization

1. **GPU Acceleration**:
   ```bash
   # Install CUDA version of PyTorch
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Memory Optimization**:
   ```python
   # In predict.py, add memory cleanup
   import gc
   gc.collect()
   torch.cuda.empty_cache()  # If using GPU
   ```

## 📈 Model Performance

### Metrics
- **Accuracy**: 85-90%
- **Precision**: ~87%
- **Recall**: ~86%
- **F1-Score**: ~86%

### Limitations
- Works best with high-quality images
- May struggle with heavily edited real photos
- Performance varies with image quality and size

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python test_model.py

# Format code
black *.py

# Lint code
flake8 *.py
```

## 🙏 Acknowledgments

- **PyTorch**: Deep learning framework
- **Streamlit**: Web application framework
- **PIL/Pillow**: Image processing
- **TorchVision**: Computer vision utilities

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**Note**: This model is for educational and research purposes. Always verify results with additional methods for critical applications. 