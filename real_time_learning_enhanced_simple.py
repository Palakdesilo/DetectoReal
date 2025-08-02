#!/usr/bin/env python3
"""
Simplified Enhanced Real-Time Learning System for DetectoReal
Implements immediate learning from user feedback with similar image detection
(No OpenCV dependency)
"""

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
from torchvision import transforms # type: ignore
import numpy as np
import hashlib
import json
import time
import threading
from PIL import Image
import os
from collections import defaultdict
import pickle
# from sklearn.metrics.pairwise import cosine_similarity  # Optional import
from model import SimpleCNN
from predict import load_prediction_model

class RLHFImageClassifier:
    """
    Reinforcement Learning with Human Feedback (RLHF) Image Classifier
    with data augmentation, feature extraction, and vector database
    """
    
    def __init__(self, model_path='model.pth', learning_rate=1e-3, memory_size=1000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.learning_lock = threading.Lock()
        self.is_learning = False
        
        # Load or initialize model
        self.model = self._load_model(model_path)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Feature extractor for intermediate CNN features
        self.feature_extractor = self._create_feature_extractor()
        
        # Vector database for similarity matching
        self.vector_db = VectorDatabase(max_size=memory_size)
        
        # Feedback dataset for fine-tuning
        self.feedback_dataset = FeedbackDataset()
        
        # Load persistent data
        self._load_persistent_data()
        
        # No statistics tracking
        
        print(f"🚀 RLHF Image Classifier initialized on {self.device}")
    
    def _load_model(self, model_path):
        """Load the CNN model with robust error handling"""
        try:
            if os.path.exists(model_path):
                print(f"Loading model from: {model_path}")
                model = SimpleCNN(num_classes=2).to(self.device)
                
                # Try loading with different approaches
                try:
                    state_dict = torch.load(model_path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    print(f"✅ Model loaded from {model_path}")
                except Exception as load_error:
                    print(f"⚠️ Error loading state dict: {load_error}")
                    print("Trying to load as full model...")
                    try:
                        model = torch.load(model_path, map_location=self.device)
                        print(f"✅ Full model loaded from {model_path}")
                    except Exception as full_load_error:
                        print(f"❌ Error loading full model: {full_load_error}")
                        print("Initializing new model...")
                        model = SimpleCNN(num_classes=2).to(self.device)
            else:
                print(f"Model file not found: {model_path}")
                print("Initializing new model...")
                model = SimpleCNN(num_classes=2).to(self.device)
            
            model.eval()
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Initializing new model...")
            model = SimpleCNN(num_classes=2).to(self.device)
            model.eval()
            return model
    
    def _create_feature_extractor(self):
        """Create feature extractor for intermediate CNN features"""
        class FeatureExtractor(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                # Extract features from the last convolutional layer
                self.features = nn.Sequential(
                    self.model.conv1, self.model.pool,
                    self.model.conv2, self.model.pool,
                    self.model.conv3, self.model.pool
                )
            
            def forward(self, x):
                return self.features(x)
        
        return FeatureExtractor(self.model).to(self.device)
    
    def predict_with_learning(self, image, user_feedback=None, user_correction=None):
        """
        Make prediction with RLHF learning capability
        """
        try:
            # Get model prediction
            prediction = self._model_predict(image)
            
            # Extract features for similarity matching
            features = self._extract_features(image)
            
            # Check vector database for similar images
            similar_match = self.vector_db.find_similar(features, threshold=0.85)
            
            if similar_match:
                # Use similar image's corrected prediction
                corrected_prediction = similar_match['corrected_label']
                source = 'similar_match'
                match_type = 'similar'
                memory_match = True
                
                print(f"🔍 Found similar image in database: {corrected_prediction}")
            else:
                source = 'model_prediction'
                match_type = 'none'
                memory_match = False
            
            # Handle user feedback for learning
            if user_feedback is not None and user_correction is not None:
                self._learn_from_feedback(image, prediction, user_correction, features)
            
            return {
                "prediction": prediction,
                "source": source,
                "match_type": match_type,
                "memory_match": memory_match,
                "learning_triggered": user_feedback is not None
            }
            
        except Exception as e:
            print(f"❌ Error in predict_with_learning: {e}")
            import traceback
            traceback.print_exc()
            return {
                "prediction": "error",
                "source": "error",
                "match_type": "none",
                "error": str(e)
            }
    
    def improve_model_with_feedback(self, image, user_correction):
        """
        Improve model using RLHF with enhanced data augmentation and learning
        """
        try:
            # Get current prediction
            current_prediction = self._model_predict(image)
            
            # Extract features
            features = self._extract_features(image)
            
            # Create augmented dataset with more variations
            augmented_images = self._create_augmentations(image)
            
            # Add original image to feedback dataset
            self.feedback_dataset.add_sample(image, user_correction, features)
            
            # Add all augmented versions
            for aug_image in augmented_images:
                self.feedback_dataset.add_sample(aug_image, user_correction, features)
            
            # Enhanced fine-tuning with multiple iterations
            self._fine_tune_model()
            
            # Update vector database
            self.vector_db.add_sample(features, user_correction)
            
            # Test improvement immediately
            test_result = self._test_immediate_improvement(image, user_correction)
            
            print(f"✅ RLHF learning completed: {current_prediction} → {user_correction}")
            print(f"📊 Augmentations applied: {len(augmented_images)}")
            print(f"🎯 Learning verification: {test_result}")
            
            # If learning didn't work well, try stronger learning
            if not test_result.get('correct', False):
                print("⚠️ Learning may not be sufficient, trying stronger approach...")
                self._force_stronger_learning(image, user_correction)
                
                # If still not learning, try model reset
                final_test = self._test_immediate_improvement(image, user_correction)
                if not final_test.get('correct', False):
                    print("🚨 Learning completely failed, resetting model...")
                    self._reset_model_for_learning()
                    self._fine_tune_model()
            
            return {
                'success': True,
                'augmentations_applied': len(augmented_images),
                'learning_verified': test_result.get('correct', False),
                'similar_images_updated': 0
            }
            
        except Exception as e:
            print(f"❌ Error in improve_model_with_feedback: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def _force_stronger_learning(self, image, user_correction):
        """Force stronger learning for stubborn cases"""
        try:
            print("🔧 Applying stronger learning approach...")
            
            # Create more aggressive augmentations
            aggressive_augmentations = self._create_aggressive_augmentations(image)
            
            # Add to feedback dataset
            features = self._extract_features(image)
            for aug_image in aggressive_augmentations:
                self.feedback_dataset.add_sample(aug_image, user_correction, features)
            
            # Multiple fine-tuning passes with increasing learning rate
            original_lr = self.optimizer.param_groups[0]['lr']
            
            for i in range(5):  # More passes
                # Increase learning rate for stronger learning
                self.optimizer.param_groups[0]['lr'] = original_lr * (2 ** i)
                print(f"🔄 Strong learning pass {i+1}/5 (LR: {self.optimizer.param_groups[0]['lr']:.6f})")
                
                self._fine_tune_model()
                
                # Test after each pass
                test_result = self._test_immediate_improvement(image, user_correction)
                if test_result.get('correct', False):
                    print(f"🎉 Learning successful on pass {i+1}!")
                    break
            
            # Reset learning rate
            self.optimizer.param_groups[0]['lr'] = original_lr
            
            # Save the improved model after strong learning
            try:
                torch.save(self.model.state_dict(), 'model.pth')
                print("💾 Strong learning model saved")
            except Exception as e:
                print(f"⚠️ Warning: Could not save model: {e}")
            
            # Save persistent data
            self._save_persistent_data()
            
            # Final test
            test_result = self._test_immediate_improvement(image, user_correction)
            print(f"🎯 Strong learning result: {test_result}")
            
        except Exception as e:
            print(f"❌ Error in strong learning: {e}")
    
    def _create_aggressive_augmentations(self, image):
        """Create more aggressive augmentations for stubborn learning"""
        # Convert image to RGB first
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        aggressive_transforms = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomResizedCrop(size=(128, 128), scale=(0.7, 1.0)),
            transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        ]
        
        augmentations = []
        for transform in aggressive_transforms:
            try:
                aug_image = transform(image)
                augmentations.append(aug_image)
            except Exception as e:
                print(f"⚠️ Aggressive augmentation failed: {e}")
                continue
        
        return augmentations
    
    def _extract_features(self, image):
        """Extract intermediate CNN features"""
        try:
            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
            
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(image_tensor)
                # Flatten features
                features = features.view(features.size(0), -1)
                return features.cpu().numpy().flatten()
                
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return np.zeros(64 * 16 * 16)  # Default feature size
    
    def _create_augmentations(self, image):
        """Create augmented versions of the image"""
        augmentations = []
        
        # Convert image to RGB first
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Define augmentation transforms
        aug_transforms = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        ]
        
        for transform in aug_transforms:
            try:
                aug_image = transform(image)
                augmentations.append(aug_image)
            except Exception as e:
                print(f"⚠️ Augmentation failed: {e}")
                continue
        
        return augmentations
    
    def _learn_from_feedback(self, image, model_prediction, user_correction, features):
        """Learn from user feedback with enhanced immediate fine-tuning"""
        with self.learning_lock:
            if self.is_learning:
                return
            
            self.is_learning = True
            
            try:
                # Add to feedback dataset
                self.feedback_dataset.add_sample(image, user_correction, features)
                
                # Create augmented versions for robust learning
                augmented_images = self._create_augmentations(image)
                for aug_image in augmented_images:
                    self.feedback_dataset.add_sample(aug_image, user_correction, features)
                
                # Enhanced fine-tuning with multiple passes
                self._fine_tune_model()
                
                # Update vector database
                self.vector_db.add_sample(features, user_correction)
                
                # Save improved model
                self._save_improved_model()
                
                # Test immediate improvement
                test_result = self._test_immediate_improvement(image, user_correction)
                
                print(f"✅ Immediate RLHF learning: {model_prediction} → {user_correction}")
                print(f"📊 Augmentations: {len(augmented_images)}")
                print(f"🎯 Immediate test: {test_result}")
                
            except Exception as e:
                print(f"❌ Error in immediate learning: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.is_learning = False
    
    def _test_immediate_improvement(self, image, expected_label):
        """Test if the model learned correctly from feedback"""
        try:
            # Get new prediction
            new_prediction = self._model_predict(image)
            
            # Check if prediction improved
            correct_prediction = (new_prediction == expected_label)
            
            return {
                "correct": correct_prediction,
                "prediction": new_prediction,
                "expected": expected_label
            }
        except Exception as e:
            print(f"❌ Error testing immediate improvement: {e}")
            return {"correct": False, "error": str(e)}
    
    def _fine_tune_model(self):
        """Fine-tune model on feedback dataset with enhanced learning"""
        if len(self.feedback_dataset) < 1:
            return
        
        try:
            self.model.train()
            
            # Get feedback data
            images, labels = self.feedback_dataset.get_batch()
            
            # Preprocess images with error handling
            processed_images = []
            for img in images:
                try:
                    # Ensure image is RGB
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    processed_img = self._preprocess_image(img)
                    processed_images.append(processed_img)
                except Exception as e:
                    print(f"⚠️ Skipping image due to processing error: {e}")
                    continue
            
            if len(processed_images) == 0:
                print("⚠️ No valid images to process for fine-tuning")
                return
            
            # Convert to tensors
            image_tensors = torch.stack(processed_images).to(self.device)
            label_tensors = torch.tensor(labels[:len(processed_images)], dtype=torch.long).to(self.device)
            
            # More aggressive training for better learning
            num_iterations = min(15, len(processed_images) * 3)  # Much more iterations
            
            total_loss = 0
            for iteration in range(num_iterations):
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(image_tensors)
                loss = F.cross_entropy(outputs, label_tensors)
                
                # Add stronger regularization
                l2_lambda = 0.001  # Reduced L2 regularization
                l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
                loss = loss + l2_lambda * l2_norm
                
                # Add focal loss for better learning of hard examples
                probs = F.softmax(outputs, dim=1)
                focal_loss = self._focal_loss(probs, label_tensors)
                loss = loss + 0.5 * focal_loss
                
                # Backward pass
                loss.backward()
                
                # Stronger gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                # Update weights
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Print progress every 5 iterations
                if (iteration + 1) % 5 == 0:
                    print(f"🔄 Training iteration {iteration + 1}/{num_iterations}, loss: {loss.item():.4f}")
            
            # Switch back to eval mode
            self.model.eval()
            
            avg_loss = total_loss / num_iterations
            print(f"🎯 Fine-tuning completed: {num_iterations} iterations, avg loss: {avg_loss:.4f}")
            
            # Verify learning by testing on the same data
            with torch.no_grad():
                test_outputs = self.model(image_tensors)
                test_probs = F.softmax(test_outputs, dim=1)
                predicted_labels = torch.argmax(test_probs, dim=1)
                correct_predictions = (predicted_labels == label_tensors).sum().item()
                accuracy = correct_predictions / len(labels[:len(processed_images)])
                print(f"📊 Learning verification: {correct_predictions}/{len(labels[:len(processed_images)])} correct ({accuracy:.2%})")
            
            # Save the improved model immediately after learning
            try:
                torch.save(self.model.state_dict(), 'model.pth')
                print("💾 Improved model saved")
            except Exception as e:
                print(f"⚠️ Warning: Could not save model: {e}")
            
            # Save persistent data
            self._save_persistent_data()
            
        except Exception as e:
            print(f"❌ Error in fine-tuning: {e}")
            import traceback
            traceback.print_exc()
    
    def _focal_loss(self, probs, targets, alpha=1.0, gamma=2.0):
        """Focal loss for better learning of hard examples"""
        ce_loss = F.cross_entropy(probs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def _reset_model_for_learning(self):
        """Reset model weights for fresh learning"""
        try:
            print("🔄 Resetting model weights for fresh learning...")
            
            # Reinitialize the model
            self.model = SimpleCNN(num_classes=2).to(self.device)
            
            # Reset optimizer with higher learning rate
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=5e-3,  # Much higher learning rate
                weight_decay=1e-5
            )
            
            # Clear feedback dataset to start fresh
            self.feedback_dataset.clear()
            
            # Save the reset model
            try:
                torch.save(self.model.state_dict(), 'model.pth')
                print("💾 Reset model saved")
            except Exception as e:
                print(f"⚠️ Warning: Could not save reset model: {e}")
            
            # Save persistent data
            self._save_persistent_data()
            
            print("✅ Model reset complete - ready for fresh learning")
            
        except Exception as e:
            print(f"❌ Error resetting model: {e}")
    
    def clear(self):
        """Clear the feedback dataset"""
        self.images.clear()
        self.labels.clear()
        self.features.clear()
    
    def _save_persistent_data(self):
        """Save vector database and feedback dataset"""
        try:
            # Save vector database
            with open('vector_db.pkl', 'wb') as f:
                pickle.dump(self.vector_db, f)
            
            # Save feedback dataset
            with open('feedback_dataset.pkl', 'wb') as f:
                pickle.dump(self.feedback_dataset, f)
            
            print("💾 Persistent data saved")
        except Exception as e:
            print(f"⚠️ Warning: Could not save persistent data: {e}")
    
    def _load_persistent_data(self):
        """Load vector database and feedback dataset"""
        try:
            # Load vector database
            if os.path.exists('vector_db.pkl'):
                with open('vector_db.pkl', 'rb') as f:
                    self.vector_db = pickle.load(f)
                print("📂 Vector database loaded")
            
            # Load feedback dataset
            if os.path.exists('feedback_dataset.pkl'):
                with open('feedback_dataset.pkl', 'rb') as f:
                    self.feedback_dataset = pickle.load(f)
                print("📂 Feedback dataset loaded")
        except Exception as e:
            print(f"⚠️ Warning: Could not load persistent data: {e}")
    
    def _preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        return transform(image)
    
    def _model_predict(self, image):
        """Get model prediction"""
        try:
            # Ensure image is RGB
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_tensor = self._preprocess_image(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted = torch.argmax(probabilities, 1)
                
                prediction = "fake" if predicted.item() == 0 else "real"
                
                return prediction
                
        except Exception as e:
            print(f"❌ Error in model prediction: {e}")
            return "error"
    
    def _save_improved_model(self):
        """Save the improved model"""
        try:
            torch.save(self.model.state_dict(), 'model.pth')
            print("💾 Improved model saved")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def test_same_image_improvement(self, image, expected_prediction):
        """Test if the same image now predicts correctly after learning"""
        result = self.predict_with_learning(image)
        
        correct_prediction = (result['prediction'] == expected_prediction)
        
        return {
            "correct_prediction": correct_prediction,
            "prediction": result['prediction'],
            "source": result['source'],
            "match_type": result['match_type'],
            "memory_used": result['memory_match']
        }
    
    def get_learning_statistics(self):
        """Get basic system info (no detailed statistics)"""
        return {
            "vector_db_size": len(self.vector_db),
            "feedback_dataset_size": len(self.feedback_dataset)
        }

class VectorDatabase:
    """Vector database for similarity matching"""
    
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.features = []
        self.labels = []
        self.timestamps = []
    
    def add_sample(self, features, corrected_label):
        """Add a sample to the vector database"""
        if len(self.features) >= self.max_size:
            # Remove oldest sample
            self.features.pop(0)
            self.labels.pop(0)
            self.timestamps.pop(0)
        
        self.features.append(features)
        self.labels.append(corrected_label)
        self.timestamps.append(time.time())
    
    def find_similar(self, query_features, threshold=0.85):
        """Find similar images in the database"""
        if len(self.features) == 0:
            return None
        
        # Calculate similarities using numpy (no sklearn dependency)
        similarities = []
        for features in self.features:
            similarity = self._cosine_similarity(query_features, features)
            similarities.append(similarity)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= threshold:
            return {
                'corrected_label': self.labels[best_idx],
                'similarity': best_similarity,
                'timestamp': self.timestamps[best_idx]
            }
        
        return None
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors using numpy"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
            
            return dot_product / (norm1 * norm2)
        except Exception as e:
            print(f"❌ Error calculating cosine similarity: {e}")
            return 0
    
    def __len__(self):
        return len(self.features)

class FeedbackDataset:
    """Dataset for storing feedback samples"""
    
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.images = []
        self.labels = []
        self.features = []
    
    def add_sample(self, image, corrected_label, features):
        """Add a feedback sample"""
        if len(self.images) >= self.max_size:
            # Remove oldest sample
            self.images.pop(0)
            self.labels.pop(0)
            self.features.pop(0)
        
        self.images.append(image)
        self.labels.append(0 if corrected_label == 'fake' else 1)
        self.features.append(features)
    
    def get_batch(self, batch_size=None):
        """Get a batch of samples"""
        if batch_size is None:
            batch_size = len(self.images)
        
        indices = np.random.choice(len(self.images), min(batch_size, len(self.images)), replace=False)
        
        batch_images = [self.images[i] for i in indices]
        batch_labels = [self.labels[i] for i in indices]
        
        return batch_images, batch_labels
    
    def __len__(self):
        return len(self.images)

def test_rlhf_image_classifier():
    """Test the RLHF image classifier"""
    print("🧪 Testing RLHF Image Classifier...")
    
    # Initialize classifier
    classifier = RLHFImageClassifier()
    
    # Create a test image
    test_image = Image.new('RGB', (128, 128), color='red')
    
    # Test prediction
    result = classifier.predict_with_learning(test_image)
    print(f"📊 Initial prediction: {result}")
    
    # Test learning
    improvement_result = classifier.improve_model_with_feedback(test_image, "fake")
    print(f"🎯 Learning result: {improvement_result}")
    
    # Test improvement
    improvement = classifier.test_same_image_improvement(test_image, "fake")
    print(f"✅ Improvement test: {improvement}")
    
    # Get statistics
    stats = classifier.get_learning_statistics()
    print(f"📈 Statistics: {stats}")
    
    print("✅ RLHF Image Classifier test completed!")

if __name__ == "__main__":
    test_rlhf_image_classifier() 