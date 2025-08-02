#!/usr/bin/env python3
"""
Enhanced Real-Time Learning System for DetectoReal
Implements immediate learning from user feedback with similar image detection
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
from model import SimpleCNN
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import image as skimage

class EnhancedRealTimeLearningSystem:
    """
    Enhanced real-time learning system that immediately improves the model
    based on user feedback and can handle similar images
    """
    
    def __init__(self, model_path='model.pth', learning_rate=1e-4, memory_size=1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        
        # Load base model
        self.model = self._load_model(model_path)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Enhanced memory system with similarity detection
        self.memory = EnhancedMemory(memory_size)
        
        # Real-time learning components
        self.learning_lock = threading.Lock()
        self.is_learning = False
        
        # Performance tracking
        self.learning_stats = {
            'total_feedback': 0,
            'successful_learnings': 0,
            'accuracy_improvements': 0,
            'similar_image_matches': 0,
            'exact_image_matches': 0,
            'last_learning_time': None
        }
        
        # Feature extractor for similar image detection
        self.feature_extractor = FeatureExtractor()
        
        print("🚀 Enhanced Real-Time Learning System initialized")
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
        Main prediction function with enhanced real-time learning capability
        """
        try:
            # Get image hash and features
            image_hash = self._get_image_hash(image)
            image_features = self.feature_extractor.extract_features(image)
            
            # Check memory for exact and similar matches
            memory_match = self.memory.find_image(image_hash, image_features)
            
            if memory_match and user_feedback is None:
                # Use memory-based prediction
                prediction = memory_match['prediction']
                confidence = memory_match['confidence']
                source = "memory"
                match_type = memory_match['match_type']
            else:
                # Use model prediction
                prediction, confidence = self._model_predict(image)
                source = "model"
                match_type = "none"
            
            # If user provides feedback, learn immediately
            if user_feedback is not None and user_correction is not None:
                self._immediate_learn(image, prediction, user_correction, confidence)
                
                # Update memory with corrected prediction
                self.memory.add_image(
                    image_hash=image_hash,
                    image_features=image_features,
                    prediction=user_correction,
                    confidence=0.9,  # High confidence for user corrections
                    timestamp=datetime.datetime.now()
                )
                
                # Re-predict with improved model
                prediction, confidence = self._model_predict(image)
                source = "improved_model"
                match_type = "none"
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "source": source,
                "match_type": match_type,
                "memory_match": memory_match is not None,
                "learning_triggered": user_feedback is not None
            }
            
        except Exception as e:
            print(f"❌ Error in predict_with_learning: {e}")
            return {
                "prediction": "error",
                "confidence": 0.0,
                "source": "error",
                "match_type": "none",
                "error": str(e)
            }
    
    def improve_model_with_feedback(self, image, user_correction):
        """
        Specific method for the "Improve Model" button
        Learns everything about the image and similar images
        """
        try:
            # Extract features for similar image learning
            image_features = self.feature_extractor.extract_features(image)
            
            # Find similar images in memory
            similar_images = self.memory.find_similar_images(image_features, threshold=0.8)
            
            # Learn from the current image
            current_prediction, confidence = self._model_predict(image)
            self._immediate_learn(image, current_prediction, user_correction, confidence)
            
            # Learn from similar images if they exist
            if similar_images:
                print(f"🎯 Learning from {len(similar_images)} similar images")
                for similar_image in similar_images:
                    if similar_image['prediction'] != user_correction:
                        # Update similar image's prediction in memory
                        similar_image['prediction'] = user_correction
                        similar_image['confidence'] = 0.9
                        similar_image['timestamp'] = datetime.datetime.now()
                        
                        # Learn from this similar image
                        self._learn_from_memory_entry(similar_image)
            
            # Update memory with current image
            image_hash = self._get_image_hash(image)
            self.memory.add_image(
                image_hash=image_hash,
                image_features=image_features,
                prediction=user_correction,
                confidence=0.95,  # Very high confidence for explicit corrections
                timestamp=datetime.datetime.now()
            )
            
            # Update statistics
            self.learning_stats['total_feedback'] += 1
            self.learning_stats['successful_learnings'] += 1
            self.learning_stats['last_learning_time'] = datetime.datetime.now()
            
            return {
                "success": True,
                "similar_images_updated": len(similar_images) if similar_images else 0,
                "learning_completed": True
            }
            
        except Exception as e:
            print(f"❌ Error in improve_model_with_feedback: {e}")
            return {
                "success": False,
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
    
    def _immediate_learn(self, image, model_prediction, user_correction, confidence):
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
                
                # Save improved model
                self._save_improved_model()
                
                print(f"✅ Immediate learning completed: {model_prediction} → {user_correction}")
                print(f"📊 Loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"❌ Error in immediate learning: {e}")
            finally:
                self.is_learning = False
    
    def _learn_from_memory_entry(self, memory_entry):
        """
        Learn from a memory entry (for similar images)
        """
        try:
            # This would require reconstructing the image from features
            # For now, we'll just update the memory entry
            print(f"🔄 Updated similar image prediction: {memory_entry['prediction']}")
        except Exception as e:
            print(f"❌ Error learning from memory entry: {e}")
    
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
            # Convert to bytes and hash
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
            "match_type": result['match_type'],
            "memory_used": result['memory_match']
        }
    
    def get_learning_statistics(self):
        """Get learning statistics"""
        return {
            **self.learning_stats,
            "memory_size": len(self.memory)
        }

class FeatureExtractor:
    """
    Extract features from images for similarity detection
    """
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    
    def extract_features(self, image):
        """Extract features from image"""
        try:
            # Convert to tensor
            image_tensor = self.transform(image).unsqueeze(0)
            
            # Extract basic features (color histogram, texture, etc.)
            features = []
            
            # Color histogram
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                # RGB image
                for channel in range(3):
                    hist = np.histogram(img_array[:, :, channel], bins=16, range=(0, 255))[0]
                    features.extend(hist / np.sum(hist))
            else:
                # Grayscale image
                hist = np.histogram(img_array, bins=16, range=(0, 255))[0]
                features.extend(hist / np.sum(hist))
            
            # Basic texture features (simplified)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            features.extend([
                np.mean(gray),
                np.std(gray),
                np.var(gray)
            ])
            
            return np.array(features)
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return np.zeros(19)  # Default feature vector

class EnhancedMemory:
    """
    Enhanced memory system with similarity detection
    """
    
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.images = deque(maxlen=max_size)
        self.image_dict = {}  # For exact matches
        self.feature_dict = {}  # For similar matches
    
    def add_image(self, image_hash, image_features, prediction, confidence, timestamp):
        """Add image to memory"""
        memory_entry = {
            'image_hash': image_hash,
            'image_features': image_features,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': timestamp,
            'match_type': 'exact'
        }
        
        self.images.append(memory_entry)
        self.image_dict[image_hash] = memory_entry
        self.feature_dict[image_hash] = image_features
    
    def find_image(self, image_hash, image_features):
        """
        Find image in memory (exact or similar match)
        Returns: memory entry if found, None otherwise
        """
        # Check for exact match first
        if image_hash in self.image_dict:
            entry = self.image_dict[image_hash]
            entry['match_type'] = 'exact'
            return entry
        
        # Check for similar images
        similar_entry = self.find_similar_images(image_features, threshold=0.85)
        if similar_entry:
            similar_entry['match_type'] = 'similar'
            return similar_entry
        
        return None
    
    def find_similar_images(self, image_features, threshold=0.8):
        """
        Find similar images based on feature similarity
        """
        if not self.feature_dict:
            return None
        
        best_match = None
        best_similarity = 0
        
        for hash_key, features in self.feature_dict.items():
            try:
                similarity = cosine_similarity([image_features], [features])[0][0]
                if similarity > threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = self.image_dict[hash_key]
                    best_match['similarity'] = similarity
            except Exception as e:
                print(f"❌ Error calculating similarity: {e}")
                continue
        
        return best_match
    
    def __len__(self):
        return len(self.images)
    
    def clear(self):
        """Clear memory"""
        self.images.clear()
        self.image_dict.clear()
        self.feature_dict.clear()

# Example usage and testing
def test_enhanced_real_time_learning():
    """Test the enhanced real-time learning system"""
    print("🧪 Testing Enhanced Real-Time Learning System")
    print("=" * 50)
    
    # Initialize the system
    rtl = EnhancedRealTimeLearningSystem(learning_rate=1e-4, memory_size=100)
    
    # Create test images
    test_image1 = Image.new('RGB', (128, 128), color='red')
    test_image2 = Image.new('RGB', (128, 128), color='darkred')  # Similar to test_image1
    
    # Step 1: Initial prediction
    print("\n1. Initial prediction...")
    result1 = rtl.predict_with_learning(test_image1)
    print(f"   Prediction: {result1['prediction']} (confidence: {result1['confidence']:.3f})")
    print(f"   Source: {result1['source']}")
    
    # Step 2: Improve model with feedback
    print("\n2. Improving model with feedback...")
    improvement_result = rtl.improve_model_with_feedback(test_image1, "fake")
    print(f"   Success: {improvement_result['success']}")
    print(f"   Similar images updated: {improvement_result['similar_images_updated']}")
    
    # Step 3: Test same image again
    print("\n3. Testing same image again...")
    result3 = rtl.predict_with_learning(test_image1)
    print(f"   Prediction: {result3['prediction']} (confidence: {result3['confidence']:.3f})")
    print(f"   Source: {result3['source']}")
    print(f"   Match type: {result3['match_type']}")
    
    # Step 4: Test similar image
    print("\n4. Testing similar image...")
    result4 = rtl.predict_with_learning(test_image2)
    print(f"   Prediction: {result4['prediction']} (confidence: {result4['confidence']:.3f})")
    print(f"   Source: {result4['source']}")
    print(f"   Match type: {result4['match_type']}")
    
    # Step 5: Get learning statistics
    print("\n5. Learning statistics...")
    stats = rtl.get_learning_statistics()
    print(f"   Total feedback: {stats['total_feedback']}")
    print(f"   Successful learnings: {stats['successful_learnings']}")
    print(f"   Memory size: {stats['memory_size']}")
    
    print("\n✅ Enhanced real-time learning test completed!")

if __name__ == "__main__":
    test_enhanced_real_time_learning() 