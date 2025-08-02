#!/usr/bin/env python3
"""
Simplified Enhanced Real-Time Learning System for DetectoReal
Implements immediate learning from user feedback with similar image detection
Streamlit Cloud Compatible - Uses both session state and file persistence
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
import random
from collections import defaultdict
import pickle
import streamlit as st
# from sklearn.metrics.pairwise import cosine_similarity  # Optional import
from model import SimpleCNN
from predict import load_prediction_model

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

class RLHFImageClassifier:
    """
    Reinforcement Learning with Human Feedback (RLHF) Image Classifier
    with data augmentation, feature extraction, and vector database
    Streamlit Cloud Compatible - Uses both session state and file persistence
    """
    
    def __init__(self, model_path='model.pth', learning_rate=1e-3, memory_size=1000):
        # Force CPU usage for consistent results between local and cloud
        self.device = torch.device('cpu')
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.learning_lock = threading.Lock()
        self.is_learning = False
        
        # Define persistent storage paths
        self.learned_model_path = 'learned_model.pth'
        self.vector_db_path = 'vector_db.pkl'
        self.feedback_dataset_path = 'feedback_dataset.pkl'
        
        # Load or initialize model
        self.model = self._load_model(model_path)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Feature extractor for intermediate CNN features
        self.feature_extractor = self._create_feature_extractor()
        
        # Vector database for similarity matching
        self.vector_db = VectorDatabase(max_size=memory_size)
        
        # Feedback dataset for fine-tuning
        self.feedback_dataset = FeedbackDataset()
        
        # Load persistent data from both session state and files
        self._load_persistent_data()
        
        print(f"🚀 RLHF Image Classifier initialized on {self.device}")
    
    def _load_model(self, model_path):
        """Load the CNN model with robust error handling and learned state"""
        try:
            # Try to load learned model first, then fallback to original model
            model_paths = ['learned_model.pth', 'model.pth']
            
            for path in model_paths:
                if os.path.exists(path):
                    print(f"📊 RLHF loading model from: {path}")
                    model = SimpleCNN(num_classes=2).to(self.device)
                    state_dict = torch.load(path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    model.eval()
                    return model
            
            # If no model files found, create a fresh model
            print("🔄 Creating fresh RLHF model (no files found)")
            model = SimpleCNN(num_classes=2).to(self.device)
            model.eval()
            return model
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Creating fresh model due to loading error")
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
            
            # Save to files for cross-session persistence
            self._save_persistent_data_to_files()
            
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
            
            # Save the improved model to both session state and files
            try:
                st.session_state.model_state = self.model.state_dict()
                print("💾 Strong learning model saved to session state")
            except Exception as e:
                print(f"⚠️ Warning: Could not save model to session: {e}")
            
            # Save to files for cross-session persistence
            self._save_persistent_data_to_files()
            
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
        
        # Get image dimensions
        img_width, img_height = image.size
        min_dimension = min(img_width, img_height)
        
        # Adjust crop size based on image dimensions
        crop_size = min(128, min_dimension - 10)
        if crop_size < 64:
            crop_size = 64
        
        aggressive_transforms = [
            # More aggressive basic transforms (safe for all image sizes)
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=0.7),
            transforms.RandomRotation(degrees=45),
            
            # More aggressive color transforms (safe for all image sizes)
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomGrayscale(p=0.2),
        ]
        
        # Add crop transforms only if image is large enough
        if min_dimension >= 128:
            aggressive_transforms.extend([
                # More aggressive crop and resize (compress/expand)
                transforms.RandomResizedCrop(size=(128, 128), scale=(0.6, 1.0)),  # More compress
                transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.4)),  # More expand
                transforms.RandomCrop(size=(128, 128), padding=20),
            ])
        elif min_dimension >= 80:
            aggressive_transforms.extend([
                transforms.RandomResizedCrop(size=(crop_size, crop_size), scale=(0.6, 1.0)),
                transforms.RandomCrop(size=(crop_size, crop_size), padding=10),
            ])
        
        # Add aggressive affine transforms
        aggressive_transforms.extend([
            # More aggressive affine transforms
            transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.6, 1.4)),
            transforms.RandomAffine(degrees=45, translate=(0.25, 0.25), scale=(0.5, 1.5)),
            
            # More aggressive perspective transforms
            transforms.RandomPerspective(distortion_scale=0.4, p=0.7),
            
            # Extreme transforms for stubborn cases
            transforms.RandomAffine(degrees=60, translate=(0.3, 0.3), scale=(0.4, 1.6)),
        ])
        
        augmentations = []
        for i, transform in enumerate(aggressive_transforms):
            try:
                aug_image = transform(image)
                augmentations.append(aug_image)
            except Exception as e:
                print(f"⚠️ Aggressive augmentation {i+1} failed: {e}")
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
        """Create comprehensive augmented versions of the image"""
        augmentations = []
        
        # Convert image to RGB first
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get image dimensions
        img_width, img_height = image.size
        min_dimension = min(img_width, img_height)
        
        # Adjust crop size based on image dimensions
        crop_size = min(128, min_dimension - 10)  # Ensure crop size is smaller than image
        if crop_size < 64:  # If image is too small, skip crop augmentations
            crop_size = 64
        
        # Define comprehensive augmentation transforms
        aug_transforms = [
            # Basic transforms (safe for all image sizes)
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            
            # Color and brightness transforms (safe for all image sizes)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
        ]
        
        # Add crop transforms only if image is large enough
        if min_dimension >= 128:
            aug_transforms.extend([
                # Crop and resize transforms (compress/expand)
                transforms.RandomResizedCrop(size=(128, 128), scale=(0.7, 1.0)),  # Compress
                transforms.RandomResizedCrop(size=(128, 128), scale=(0.9, 1.2)),  # Expand
                transforms.RandomCrop(size=(128, 128), padding=10),
            ])
        elif min_dimension >= 80:  # Use smaller crop size for medium images
            aug_transforms.extend([
                transforms.RandomResizedCrop(size=(crop_size, crop_size), scale=(0.8, 1.0)),
                transforms.RandomCrop(size=(crop_size, crop_size), padding=5),
            ])
        
        # Add affine transforms (safe with proper error handling)
        aug_transforms.extend([
            # Affine transforms (rotate, translate, scale)
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.7, 1.3)),
            
            # Perspective transforms
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            
            # Elastic transforms (simulated with affine)
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ])
        
        for i, transform in enumerate(aug_transforms):
            try:
                aug_image = transform(image)
                augmentations.append(aug_image)
            except Exception as e:
                print(f"⚠️ Augmentation {i+1} failed: {e}")
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
                
                # Save improved model to both session state and files
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
            print(f"🔄 Starting fine-tuning with model ID: {id(self.model)}")
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
            
            # Fine-tune for 10-20 epochs as requested
            num_epochs = min(20, max(10, len(processed_images)))  # Ensure 10-20 epochs
            total_loss = 0
            total_iterations = 0
            
            print(f"🎯 Starting fine-tuning for {num_epochs} epochs...")
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                epoch_iterations = 0
                
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
                
                epoch_loss += loss.item()
                epoch_iterations += 1
                total_loss += loss.item()
                total_iterations += 1
                
                # Print progress every epoch
                print(f"🔄 Epoch {epoch + 1}/{num_epochs}, loss: {loss.item():.4f}")
            
            # Switch back to eval mode
            self.model.eval()
            
            avg_loss = total_loss / total_iterations
            print(f"🎯 Fine-tuning completed: {num_epochs} epochs, avg loss: {avg_loss:.4f}")
            print(f"✅ Model updated in place - Model ID: {id(self.model)}")
            
            # Verify learning by testing on the same data
            with torch.no_grad():
                test_outputs = self.model(image_tensors)
                test_probs = F.softmax(test_outputs, dim=1)
                predicted_labels = torch.argmax(test_probs, dim=1)
                correct_predictions = (predicted_labels == label_tensors).sum().item()
                accuracy = correct_predictions / len(labels[:len(processed_images)])
                print(f"📊 Learning verification: {correct_predictions}/{len(labels[:len(processed_images)])} correct ({accuracy:.2%})")
            
            # Save the improved model to both session state and files
            try:
                st.session_state.model_state = self.model.state_dict()
                print("💾 Improved model saved to session state")
            except Exception as e:
                print(f"⚠️ Warning: Could not save model to session: {e}")
            
            # Save to files for cross-session persistence
            self._save_persistent_data_to_files()
            
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
            
            # Load original model weights instead of creating new model
            if os.path.exists('model.pth'):
                print("Loading original model weights...")
                state_dict = torch.load('model.pth', map_location=self.device)
                self.model.load_state_dict(state_dict)
                print("✅ Original model weights loaded")
            else:
                print("Initializing new model weights...")
                # Initialize with random weights
                for param in self.model.parameters():
                    if param.dim() > 1:
                        torch.nn.init.xavier_uniform_(param)
                    else:
                        torch.nn.init.zeros_(param)
            
            # Reset optimizer with higher learning rate
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=5e-3,  # Much higher learning rate
                weight_decay=1e-5
            )
            
            # Clear feedback dataset to start fresh
            self.feedback_dataset.clear()
            
            # Save the reset model to both session state and files
            try:
                st.session_state.model_state = self.model.state_dict()
                print("💾 Reset model saved to session state")
            except Exception as e:
                print(f"⚠️ Warning: Could not save reset model to session: {e}")
            
            # Save to files for cross-session persistence
            self._save_persistent_data_to_files()
            
            print("✅ Model reset complete - ready for fresh learning")
            
        except Exception as e:
            print(f"❌ Error resetting model: {e}")
    
    def clear(self):
        """Clear the feedback dataset"""
        self.images.clear()
        self.labels.clear()
        self.features.clear()
    
    def _save_persistent_data_to_session(self):
        """Save persistent data to Streamlit session state"""
        try:
            # Save vector database to session state
            st.session_state.vector_db_data = {
                'features': self.vector_db.features,
                'labels': self.vector_db.labels,
                'timestamps': self.vector_db.timestamps
            }
            
            # Save feedback dataset to session state
            st.session_state.feedback_dataset_data = {
                'images': self.feedback_dataset.images,
                'labels': self.feedback_dataset.labels,
                'features': self.feedback_dataset.features
            }
            
            # Save model state to session state
            st.session_state.model_state = self.model.state_dict()
            
            print("💾 Persistent data saved to session state")
        except Exception as e:
            print(f"⚠️ Warning: Could not save persistent data to session: {e}")
    
    def _save_persistent_data_to_files(self):
        """Save persistent data to files for cross-session persistence"""
        try:
            # Save learned model to file
            torch.save(self.model.state_dict(), self.learned_model_path)
            print(f"💾 Learned model saved to {self.learned_model_path}")
            
            # Save vector database to file
            vector_db_data = {
                'features': self.vector_db.features,
                'labels': self.vector_db.labels,
                'timestamps': self.vector_db.timestamps
            }
            with open(self.vector_db_path, 'wb') as f:
                pickle.dump(vector_db_data, f)
            print(f"💾 Vector database saved to {self.vector_db_path}")
            
            # Save feedback dataset to file
            feedback_dataset_data = {
                'images': self.feedback_dataset.images,
                'labels': self.feedback_dataset.labels,
                'features': self.feedback_dataset.features
            }
            with open(self.feedback_dataset_path, 'wb') as f:
                pickle.dump(feedback_dataset_data, f)
            print(f"💾 Feedback dataset saved to {self.feedback_dataset_path}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save persistent data to files: {e}")
    
    def _load_persistent_data(self):
        """Load persistent data from both session state and files"""
        try:
            # Load from session state first (for current session)
            self._load_persistent_data_from_session()
            
            # Load from files (for cross-session persistence)
            self._load_persistent_data_from_files()
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load persistent data: {e}")
    
    def _load_persistent_data_from_session(self):
        """Load persistent data from Streamlit session state"""
        try:
            # Load vector database from session state
            if 'vector_db_data' in st.session_state:
                vector_data = st.session_state.vector_db_data
                self.vector_db.features = vector_data.get('features', [])
                self.vector_db.labels = vector_data.get('labels', [])
                self.vector_db.timestamps = vector_data.get('timestamps', [])
                print(f"📂 Vector database loaded from session: {len(self.vector_db.features)} samples")
            
            # Load feedback dataset from session state
            if 'feedback_dataset_data' in st.session_state:
                feedback_data = st.session_state.feedback_dataset_data
                self.feedback_dataset.images = feedback_data.get('images', [])
                self.feedback_dataset.labels = feedback_data.get('labels', [])
                self.feedback_dataset.features = feedback_data.get('features', [])
                print(f"📂 Feedback dataset loaded from session: {len(self.feedback_dataset.images)} samples")
                
            # Load model state from session state
            if 'model_state' in st.session_state:
                try:
                    self.model.load_state_dict(st.session_state.model_state)
                    print("📂 Model state loaded from session")
                except Exception as e:
                    print(f"⚠️ Could not load model state from session: {e}")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not load persistent data from session: {e}")
    
    def _load_persistent_data_from_files(self):
        """Load persistent data from files for cross-session persistence"""
        try:
            # Load vector database from file
            if os.path.exists(self.vector_db_path):
                with open(self.vector_db_path, 'rb') as f:
                    vector_data = pickle.load(f)
                self.vector_db.features = vector_data.get('features', [])
                self.vector_db.labels = vector_data.get('labels', [])
                self.vector_db.timestamps = vector_data.get('timestamps', [])
                print(f"📂 Vector database loaded from file: {len(self.vector_db.features)} samples")
            
            # Load feedback dataset from file
            if os.path.exists(self.feedback_dataset_path):
                with open(self.feedback_dataset_path, 'rb') as f:
                    feedback_data = pickle.load(f)
                self.feedback_dataset.images = feedback_data.get('images', [])
                self.feedback_dataset.labels = feedback_data.get('labels', [])
                self.feedback_dataset.features = feedback_data.get('features', [])
                print(f"📂 Feedback dataset loaded from file: {len(self.feedback_dataset.images)} samples")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not load persistent data from files: {e}")
    
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
                
                print(f"🎯 Prediction made with model ID: {id(self.model)}")
                return prediction
                
        except Exception as e:
            print(f"❌ Error in model prediction: {e}")
            return "error"
    
    def _save_improved_model(self):
        """Save the improved model to both session state and files"""
        try:
            # Save model state to session state
            st.session_state.model_state = self.model.state_dict()
            print("💾 Improved model saved to session state")
            
            # Save to files for cross-session persistence
            self._save_persistent_data_to_files()
            
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
    
    def clear_learned_data(self):
        """Clear all learned data and reset to original model"""
        try:
            print("🗑️ Clearing all learned data...")
            
            # Remove learned model file
            if os.path.exists(self.learned_model_path):
                os.remove(self.learned_model_path)
                print(f"🗑️ Removed learned model: {self.learned_model_path}")
            
            # Remove vector database file
            if os.path.exists(self.vector_db_path):
                os.remove(self.vector_db_path)
                print(f"🗑️ Removed vector database: {self.vector_db_path}")
            
            # Remove feedback dataset file
            if os.path.exists(self.feedback_dataset_path):
                os.remove(self.feedback_dataset_path)
                print(f"🗑️ Removed feedback dataset: {self.feedback_dataset_path}")
            
            # Clear session state
            if 'model_state' in st.session_state:
                del st.session_state.model_state
            if 'vector_db_data' in st.session_state:
                del st.session_state.vector_db_data
            if 'feedback_dataset_data' in st.session_state:
                del st.session_state.feedback_dataset_data
            
            # Reset model to original
            self.model = self._load_model('model.pth')
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
            
            # Clear in-memory data
            self.vector_db.features.clear()
            self.vector_db.labels.clear()
            self.vector_db.timestamps.clear()
            self.feedback_dataset.clear()
            
            print("✅ All learned data cleared successfully")
            
        except Exception as e:
            print(f"❌ Error clearing learned data: {e}")
    
    def has_learned_data(self):
        """Check if there is any learned data"""
        has_learned_model = os.path.exists(self.learned_model_path)
        has_vector_db = os.path.exists(self.vector_db_path)
        has_feedback_dataset = os.path.exists(self.feedback_dataset_path)
        
        return {
            'has_learned_model': has_learned_model,
            'has_vector_db': has_vector_db,
            'has_feedback_dataset': has_feedback_dataset,
            'has_any_learned_data': has_learned_model or has_vector_db or has_feedback_dataset
        }
    
    def get_learning_statistics(self):
        """Get detailed learning statistics"""
        learning_status = self.has_learned_data()
        
        return {
            "vector_db_size": len(self.vector_db),
            "feedback_dataset_size": len(self.feedback_dataset),
            "learning_status": learning_status,
            "model_device": str(self.device),
            "learning_rate": self.learning_rate,
            "model_id": id(self.model)
        }
    
    def ensure_same_model_instance(self):
        """Ensure we're always using the same model instance"""
        print(f"🔒 Ensuring same model instance - Model ID: {id(self.model)}")
        return self.model

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