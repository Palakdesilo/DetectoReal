# 🤖 RLHF Image Classifier - Real vs AI-Generated Detection

A **Reinforcement Learning with Human Feedback (RLHF)** system for detecting real vs AI-generated images with continuous learning capabilities.

## 🚀 Key Features

### **Real-Time Learning**
- **Immediate Model Updates**: Learn from user feedback instantly
- **Data Augmentation**: Apply transformations (flip, rotate, color jitter, crop, affine)
- **Fine-tuning**: Update model weights without full retraining
- **Confidence-based Learning**: High-confidence mistakes penalized more

### **Advanced Similarity Matching**
- **CNN Feature Extraction**: Extract intermediate convolutional features
- **Vector Database**: Store and query similar images efficiently
- **Cosine Similarity**: Find similar images for improved predictions
- **Memory Management**: Automatic cleanup of old samples

### **Human-in-the-Loop Interface**
- **Streamlit Web App**: User-friendly interface
- **Feedback Collection**: Easy correction of wrong predictions
- **Real-time Updates**: See learning progress immediately

## 🏗️ Architecture

### **Core Components**

1. **RLHFImageClassifier**: Main classifier with learning capabilities
2. **VectorDatabase**: Stores CNN features for similarity matching
3. **FeedbackDataset**: Manages feedback samples for fine-tuning
4. **FeatureExtractor**: Extracts intermediate CNN features

### **Learning Pipeline**

```
User Upload → Model Prediction → User Feedback → Data Augmentation → Fine-tuning → Vector DB Update
```

## 📊 How It Works

### **1. Initial Prediction**
- Model predicts real/fake with confidence score
- Extracts CNN features for similarity matching
- Checks vector database for similar images

### **2. User Feedback**
- User clicks "❌ Incorrect" if prediction is wrong
- Selects correct classification from dropdown
- Clicks "🔧 Improve Model" to trigger learning

### **3. RLHF Learning Process**
- **Data Augmentation**: Creates 5 augmented versions
  - Horizontal flip
  - Random rotation (±15°)
  - Color jitter (brightness, contrast, saturation)
  - Random crop (80-100% scale)
  - Random affine (translation, rotation)

- **Feature Extraction**: Extracts intermediate CNN features
- **Fine-tuning**: Updates model weights on feedback data
- **Vector DB Update**: Stores features for future similarity matching

### **4. Similarity Matching**
- New images are compared to stored features
- Cosine similarity threshold: 0.85
- Uses corrected labels from similar images
- Improves prediction accuracy for similar images

## 🛠️ Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📁 File Structure

```
├── app.py                              # Main Streamlit application
├── real_time_learning_enhanced_simple.py  # RLHF system implementation
├── model.py                            # CNN model architecture
├── predict.py                          # Prediction utilities
├── requirements.txt                    # Python dependencies
└── README_RLHF.md                     # This documentation
```

## 🎯 Usage

### **Basic Workflow**

1. **Upload Image**: Drag and drop or select an image
2. **Get Prediction**: Model predicts real/fake with confidence
3. **Provide Feedback**: If wrong, click "❌ Incorrect"
4. **Select Correction**: Choose correct classification
5. **Improve Model**: Click "🔧 Improve Model"
6. **Verify Learning**: Upload similar image to test improvement

### **Advanced Features**

- **Similarity Matching**: Automatically finds similar images
- **Data Augmentation**: Creates 5 variations for robust learning
- **Fine-tuning**: Updates model without full retraining
- **Real-time Learning**: Immediate model updates

## 🔧 Technical Details

### **Model Architecture**
- **SimpleCNN**: 3 convolutional layers + 2 fully connected
- **Input**: 128x128 RGB images
- **Output**: 2 classes (real/fake)
- **Features**: 64×16×16 = 16,384 dimensional feature vectors

### **Learning Parameters**
- **Learning Rate**: 1e-4 (Adam optimizer)
- **Loss Function**: Cross-entropy with confidence regularization
- **Gradient Clipping**: max_norm=1.0
- **Batch Size**: All available feedback samples

### **Vector Database**
- **Max Size**: 1000 samples
- **Similarity Threshold**: 0.85
- **Storage**: Features + labels + confidence + timestamps
- **Cleanup**: FIFO (First In, First Out)

### **Data Augmentation**
```python
aug_transforms = [
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
]
```

## 📈 Performance Metrics

### **Basic System Info**
- **Vector DB Size**: Current database size
- **Feedback Dataset Size**: Number of stored samples

### **Improvement Tracking**
- **Before/After Predictions**: Compare model performance
- **Confidence Scores**: Track prediction confidence
- **Similarity Matches**: Monitor vector DB effectiveness

## 🔍 Example Usage

```python
from real_time_learning_enhanced_simple import RLHFImageClassifier

# Initialize classifier
classifier = RLHFImageClassifier()

# Make prediction
result = classifier.predict_with_learning(image)
print(f"Prediction: {result['prediction']}")

# Improve model with feedback
improvement = classifier.improve_model_with_feedback(image, "fake")
print(f"Learning success: {improvement['success']}")

# Test improvement
test_result = classifier.test_same_image_improvement(image, "fake")
print(f"Correct prediction: {test_result['correct_prediction']}")
```

## 🚀 Benefits

### **Continuous Improvement**
- **No Full Retraining**: Fine-tune existing model
- **Real-time Learning**: Immediate updates from feedback
- **Similarity Matching**: Learn from similar images
- **Data Augmentation**: Robust learning with variations

### **User Experience**
- **Simple Interface**: Easy feedback collection
- **Immediate Results**: See learning progress
- **Confidence Scores**: Understand prediction reliability

### **Technical Advantages**
- **Memory Efficient**: Vector database with size limits
- **Fast Inference**: Optimized feature extraction
- **Robust Learning**: Augmentation prevents overfitting
- **Scalable**: Modular architecture for easy extension

## 🔮 Future Enhancements

- **Advanced Augmentations**: More sophisticated transformations
- **Multi-modal Learning**: Combine with text descriptions
- **Distributed Training**: Scale across multiple GPUs
- **Advanced Similarity**: Use more sophisticated similarity metrics
- **Persistent Storage**: Save vector database to disk
- **Performance Metrics**: Detailed accuracy tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for continuous learning and improvement** 