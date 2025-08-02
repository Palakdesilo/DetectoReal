#!/usr/bin/env python3
"""
Real-Time Learning System for DetectoReal
Implements immediate learning from user feedback with feature extraction
and dynamic model improvement
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import os
import hashlib
import datetime
import threading
import time
import pickle
from collections import defaultdict, deque
import base64
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import FeatureHasher
import cv2
from model import SimpleCNN

class RealTimeLearningSystem:
    """
    Real-time learning system that immediately improves the model
    based on user feedback with feature extraction and similarity matching
    """
    
    def __init__(self, model_path='model.pth', learning_rate=1e-4, memory_size=1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        
        # Load base model
        self.model = self._load_model(model_path)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Feature extraction and memory
        self.feature_extractor = FeatureExtractor()
        self.memory = LearningMemory(memory_size)
        
        # Real-time learning components
        self.feedback_queue = deque(maxlen=100)
        self.learning_lock = threading.Lock()
        self.is_learning = False
        
        # Performance tracking
        self.performance_history = []
        self.learning_stats = {
            'total_feedback': 0,
            'successful_learnings': 0,
            'accuracy_improvements': 0,
            'last_learning_time': None
        }
        
        print("🚀 Real-Time Learning System initialized")
        print(f"📊 Memory size: {memory_size}")
        print(f"🎯 Learning rate: {learning_rate}")
    
    def _load_model(self, model_path):
        """Load the model with error handling"""
        try:
            model = SimpleCNN(num_classes=2)
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Loaded model from {model_path}")
            else:
                print(f"⚠️ Model file not found at {model_path}, using untrained model")
            
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            model = SimpleCNN(num_classes=2).to(self.device)
            return model
    
    def predict_with_learning(self, image, user_feedback=None, user_correction=None):
        """
        Main prediction function with real-time learning capability
        """
        try:
            # Extract features from image
            features = self.feature_extractor.extract_features(image)
            image_hash = self._get_image_hash(image)
            
            # Check memory for similar images
            memory_match = self.memory.find_similar_image(features, threshold=0.8)
            
            if memory_match and user_feedback is None:
                # Use memory-based prediction
                prediction = memory_match['prediction']
                confidence = memory_match['confidence']
                source = "memory"
            else:
                # Use model prediction
                prediction, confidence = self._model_predict(image)
                source = "model"
            
            # If user provides feedback, learn immediately
            if user_feedback is not None and user_correction is not None:
                self._immediate_learn(image, features, prediction, user_correction, confidence)
                
                # Update memory with corrected prediction
                self.memory.add_image(
                    features=features,
                    image_hash=image_hash,
                    prediction=user_correction,
                    confidence=0.9,  # High confidence for user corrections
                    timestamp=datetime.datetime.now()
                )
                
                # Re-predict with improved model
                prediction, confidence = self._model_predict(image)
                source = "improved_model"
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "source": source,
                "memory_match": memory_match is not None,
                "learning_triggered": user_feedback is not None
            }
            
        except Exception as e:
            print(f"❌ Error in predict_with_learning: {e}")
            return {
                "prediction": "error",
                "confidence": 0.0,
                "source": "error",
                "error": str(e)
            }
    
    def _model_predict(self, image):
        """Get prediction from the current model"""
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = ['fake', 'real'][predicted.item()]
            confidence_score = confidence.item()
        
        return predicted_class, confidence_score
    
    def _immediate_learn(self, image, features, model_prediction, user_correction, confidence):
        """
        Immediately learn from user feedback
        """
        with self.learning_lock:
            if self.is_learning:
                return  # Prevent concurrent learning
            
            self.is_learning = True
            
            try:
                # Create training data
                transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5]*3, [0.5]*3)
                ])
                
                image_tensor = transform(image).unsqueeze(0).to(self.device)
                
                # Create target based on user correction
                target = torch.tensor([0 if user_correction == 'fake' else 1], dtype=torch.long).to(self.device)
                
                # Calculate loss
                self.model.train()
                self.optimizer.zero_grad()
                
                output = self.model(image_tensor)
                loss = F.cross_entropy(output, target)
                
                # Add regularization based on confidence
                if confidence > 0.7:  # High confidence mistakes are penalized more
                    loss *= 1.5
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
                
                # Switch back to eval mode
                self.model.eval()
                
                # Update statistics
                self.learning_stats['total_feedback'] += 1
                self.learning_stats['successful_learnings'] += 1
                self.learning_stats['last_learning_time'] = datetime.datetime.now()
                
                # Save improved model
                self._save_improved_model()
                
                print(f"✅ Immediate learning completed: {model_prediction} → {user_correction}")
                print(f"📊 Loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"❌ Error in immediate learning: {e}")
            finally:
                self.is_learning = False
    
    def _save_improved_model(self):
        """Save the improved model"""
        try:
            torch.save(self.model.state_dict(), 'model.pth')
            print("💾 Improved model saved")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def _get_image_hash(self, image):
        """Create a hash of the image"""
        try:
            img_bytes = image.tobytes()
            return hashlib.md5(img_bytes).hexdigest()[:8]
        except:
            return "unknown"
    
    def test_same_image_improvement(self, image, expected_prediction):
        """
        Test if the same image now predicts correctly after learning
        """
        result = self.predict_with_learning(image)
        
        correct_prediction = (result['prediction'] == expected_prediction)
        
        return {
            "correct_prediction": correct_prediction,
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "source": result['source'],
            "memory_used": result['memory_match']
        }
    
    def get_learning_statistics(self):
        """Get learning statistics"""
        return {
            **self.learning_stats,
            "memory_size": len(self.memory),
            "performance_history": len(self.performance_history)
        }

class FeatureExtractor:
    """
    Advanced feature extraction for image similarity and learning
    """
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    
    def extract_features(self, image):
        """
        Extract comprehensive features from image
        Returns: feature vector
        """
        features = {}
        
        # 1. Basic image features
        features.update(self._extract_basic_features(image))
        
        # 2. Color histogram features
        features.update(self._extract_color_features(image))
        
        # 3. Texture features
        features.update(self._extract_texture_features(image))
        
        # 4. Edge features
        features.update(self._extract_edge_features(image))
        
        # Convert to feature vector
        feature_vector = self._features_to_vector(features)
        
        return feature_vector
    
    def _extract_basic_features(self, image):
        """Extract basic image features"""
        features = {}
        
        # Image size
        features['width'] = image.width
        features['height'] = image.height
        features['aspect_ratio'] = image.width / image.height
        
        # Convert to numpy for processing
        img_array = np.array(image)
        
        # Brightness
        features['brightness'] = np.mean(img_array)
        
        # Contrast
        features['contrast'] = np.std(img_array)
        
        return features
    
    def _extract_color_features(self, image):
        """Extract color histogram features"""
        features = {}
        
        img_array = np.array(image)
        
        # RGB histograms
        for i, color in enumerate(['red', 'green', 'blue']):
            hist = cv2.calcHist([img_array], [i], None, [8], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            
            for j, val in enumerate(hist):
                features[f'{color}_hist_{j}'] = val
        
        # HSV histograms
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        for i, channel in enumerate(['hue', 'saturation', 'value']):
            hist = cv2.calcHist([hsv], [i], None, [8], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            for j, val in enumerate(hist):
                features[f'{channel}_hist_{j}'] = val
        
        return features
    
    def _extract_texture_features(self, image):
        """Extract texture features using GLCM-like approach"""
        features = {}
        
        img_array = np.array(image.convert('L'))  # Convert to grayscale
        
        # Simple texture features
        features['texture_variance'] = np.var(img_array)
        features['texture_entropy'] = self._calculate_entropy(img_array)
        
        # Local binary pattern approximation
        features['lbp_uniform'] = self._calculate_lbp_uniform(img_array)
        
        return features
    
    def _extract_edge_features(self, image):
        """Extract edge-based features"""
        features = {}
        
        img_array = np.array(image.convert('L'))
        
        # Sobel edges
        sobel_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)
        
        # Edge magnitude
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        features['edge_density'] = np.mean(edge_magnitude)
        features['edge_variance'] = np.var(edge_magnitude)
        
        return features
    
    def _calculate_entropy(self, img_array):
        """Calculate image entropy"""
        hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        hist = hist[hist > 0]  # Remove zero probabilities
        return -np.sum(hist * np.log2(hist))
    
    def _calculate_lbp_uniform(self, img_array):
        """Calculate uniform LBP pattern count"""
        # Simplified LBP calculation
        height, width = img_array.shape
        lbp_count = 0
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = img_array[i, j]
                pattern = 0
                
                # 8-neighbor LBP
                neighbors = [
                    img_array[i-1, j-1], img_array[i-1, j], img_array[i-1, j+1],
                    img_array[i, j+1], img_array[i+1, j+1], img_array[i+1, j],
                    img_array[i+1, j-1], img_array[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        pattern |= (1 << k)
                
                # Count uniform patterns (simplified)
                if bin(pattern).count('1') <= 2 or bin(pattern).count('1') >= 6:
                    lbp_count += 1
        
        return lbp_count / (height * width)
    
    def _features_to_vector(self, features):
        """Convert features dictionary to feature vector"""
        # Use feature hashing for consistent vector size
        hasher = FeatureHasher(n_features=128, input_type='dict')
        feature_vector = hasher.transform([features]).toarray()[0]
        
        return feature_vector

class LearningMemory:
    """
    Memory system for storing and retrieving similar images
    """
    
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.images = deque(maxlen=max_size)
        self.features_cache = {}
        self.similarity_threshold = 0.8
    
    def add_image(self, features, image_hash, prediction, confidence, timestamp):
        """Add image to memory"""
        memory_entry = {
            'features': features,
            'image_hash': image_hash,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': timestamp
        }
        
        self.images.append(memory_entry)
        self.features_cache[image_hash] = features
    
    def find_similar_image(self, features, threshold=None):
        """
        Find similar image in memory
        Returns: memory entry if found, None otherwise
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        if not self.images:
            return None
        
        # Calculate similarities
        similarities = []
        for entry in self.images:
            similarity = cosine_similarity(
                features.reshape(1, -1), 
                entry['features'].reshape(1, -1)
            )[0][0]
            similarities.append((similarity, entry))
        
        # Find best match
        best_similarity, best_entry = max(similarities, key=lambda x: x[0])
        
        if best_similarity >= threshold:
            return best_entry
        
        return None
    
    def __len__(self):
        return len(self.images)
    
    def clear(self):
        """Clear memory"""
        self.images.clear()
        self.features_cache.clear()

# Example usage and testing
def test_real_time_learning():
    """Test the real-time learning system"""
    print("🧪 Testing Real-Time Learning System")
    print("=" * 50)
    
    # Initialize system
    rtl = RealTimeLearningSystem(learning_rate=1e-4, memory_size=100)
    
    # Create test image
    test_image = Image.new('RGB', (128, 128), color='red')
    
    # Step 1: Initial prediction
    print("\n1. Initial prediction...")
    result1 = rtl.predict_with_learning(test_image)
    print(f"   Prediction: {result1['prediction']} (confidence: {result1['confidence']:.3f})")
    print(f"   Source: {result1['source']}")
    
    # Step 2: User provides feedback (simulating wrong prediction)
    print("\n2. User provides feedback...")
    result2 = rtl.predict_with_learning(
        test_image, 
        user_feedback="This is wrong", 
        user_correction="fake"  # User says it's actually AI-generated
    )
    print(f"   Learning triggered: {result2['learning_triggered']}")
    print(f"   New prediction: {result2['prediction']} (confidence: {result2['confidence']:.3f})")
    print(f"   Source: {result2['source']}")
    
    # Step 3: Test same image again
    print("\n3. Testing same image again...")
    result3 = rtl.predict_with_learning(test_image)
    print(f"   Prediction: {result3['prediction']} (confidence: {result3['confidence']:.3f})")
    print(f"   Source: {result3['source']}")
    print(f"   Memory used: {result3['memory_match']}")
    
    # Step 4: Test improvement
    print("\n4. Testing improvement...")
    improvement = rtl.test_same_image_improvement(test_image, "fake")
    print(f"   Correct prediction: {improvement['correct_prediction']}")
    print(f"   Prediction: {improvement['prediction']}")
    print(f"   Confidence: {improvement['confidence']:.3f}")
    
    # Step 5: Get statistics
    print("\n5. Learning statistics...")
    stats = rtl.get_learning_statistics()
    print(f"   Total feedback: {stats['total_feedback']}")
    print(f"   Successful learnings: {stats['successful_learnings']}")
    print(f"   Memory size: {stats['memory_size']}")
    
    print("\n✅ Real-time learning test completed!")

if __name__ == "__main__":
    test_real_time_learning() 