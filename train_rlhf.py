#!/usr/bin/env python3
"""
RLHF Training Script for DetectoReal
Reinforcement Learning with Human Feedback for image authenticity detection
"""

import torch # type: ignore 
import torch.nn as nn # type: ignore 
import torch.optim as optim # type: ignore 
from torch.utils.data import Dataset, DataLoader # type: ignore 
import torch.nn.functional as F # type: ignore 
from torchvision import transforms # type: ignore 
from PIL import Image
import json
import os
import base64
from io import BytesIO
import numpy as np
import glob
import datetime
import argparse
from model import SimpleCNN
import copy
from enhanced_feedback import EnhancedFeedbackCollector, FeedbackAnalyzer

class RLHFTrainer:
    """
    Advanced RLHF trainer with multiple training strategies
    """
    
    def __init__(self, device='cpu', verbose=True):
        self.device = torch.device(device)
        self.verbose = verbose
        self.policy_model = None
        self.reward_model = None
        self.policy_optimizer = None
        self.reward_optimizer = None
        self.training_history = []
        
    def log(self, message):
        """Log message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def initialize_models(self, base_model_path='model.pth'):
        """Initialize policy and reward models from existing model"""
        self.log("🔄 Loading existing model for continuous improvement...")
        
        # Load existing model
        base_model = SimpleCNN(num_classes=2)
        try:
            base_model.load_state_dict(torch.load(base_model_path, map_location=self.device))
            self.log("✅ Loaded existing model successfully for improvement")
        except Exception as e:
            self.log(f"❌ Error loading existing model: {e}")
            return False
        
        # Initialize policy and reward models from existing model
        self.policy_model = PolicyModel(base_model).to(self.device)
        self.reward_model = RewardModel(base_model).to(self.device)
        
        # Initialize optimizers with different learning rates for incremental learning
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=5e-5, weight_decay=1e-5)  # Lower LR for incremental learning
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=5e-5, weight_decay=1e-5)
        
        self.log("✅ Models initialized for continuous improvement")
        return True
    
    def train_reward_model(self, feedback_data, epochs=20, batch_size=8):
        """Train the reward model on human feedback"""
        self.log("🎯 Training reward model...")
        
        if not feedback_data:
            self.log("❌ No feedback data available")
            return False
        
        # Create dataset and dataloader
        dataset = FeedbackDataset(feedback_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.reward_model.train()
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.reward_optimizer, patience=3)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (images, rewards) in enumerate(dataloader):
                images = images.to(self.device)
                rewards = rewards.to(self.device)
                
                self.reward_optimizer.zero_grad()
                predicted_rewards = self.reward_model(images)
                loss = criterion(predicted_rewards, rewards)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), max_norm=1.0)
                
                self.reward_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.training_history.append({
                'epoch': epoch + 1,
                'reward_loss': avg_loss,
                'type': 'reward'
            })
            
            self.log(f"Epoch {epoch+1}/{epochs} - Reward Loss: {avg_loss:.4f}")
            
            if patience_counter >= 5:
                self.log("🛑 Early stopping reward model training")
                break
        
        self.log("✅ Reward model training completed")
        return True
    
    def train_policy_model(self, feedback_data, epochs=10, batch_size=4, method='ppo'):
        """Train the policy model using different RL methods"""
        self.log(f"🤖 Training policy model with {method.upper()}...")
        
        if not feedback_data:
            self.log("❌ No feedback data available")
            return False
        
        # Create dataset
        dataset = FeedbackDataset(feedback_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.policy_model.train()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.policy_optimizer, patience=3)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (images, rewards) in enumerate(dataloader):
                images = images.to(self.device)
                rewards = rewards.to(self.device)
                
                # Get action probabilities
                action_probs = self.policy_model.get_action_probs(images)
                
                # Get reward predictions
                with torch.no_grad():
                    predicted_rewards = self.reward_model(images)
                
                if method == 'ppo':
                    # PPO-style policy loss
                    log_probs = torch.log(action_probs + 1e-8)
                    policy_loss = -(log_probs * predicted_rewards).mean()
                elif method == 'a2c':
                    # A2C-style policy loss
                    log_probs = torch.log(action_probs + 1e-8)
                    advantage = predicted_rewards - predicted_rewards.mean()
                    policy_loss = -(log_probs * advantage).mean()
                else:
                    # Simple policy gradient
                    log_probs = torch.log(action_probs + 1e-8)
                    policy_loss = -(log_probs * predicted_rewards).mean()
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
                
                self.policy_optimizer.step()
                
                total_loss += policy_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.training_history.append({
                'epoch': epoch + 1,
                'policy_loss': avg_loss,
                'method': method,
                'type': 'policy'
            })
            
            self.log(f"Epoch {epoch+1}/{epochs} - Policy Loss: {avg_loss:.4f}")
            
            if patience_counter >= 5:
                self.log("🛑 Early stopping policy model training")
                break
        
        self.log("✅ Policy model training completed")
        return True
    
    def train_rlhf(self, feedback_data, reward_epochs=20, policy_epochs=10, 
                   batch_size=8, policy_method='ppo'):
        """Complete RLHF training process with configurable parameters"""
        self.log("🚀 Starting RLHF training process...")
        
        # Step 1: Train reward model
        success = self.train_reward_model(feedback_data, reward_epochs, batch_size)
        if not success:
            return False
        
        # Step 2: Train policy model
        success = self.train_policy_model(feedback_data, policy_epochs, batch_size//2, policy_method)
        if not success:
            return False
        
        self.log("🎉 RLHF training completed successfully!")
        return True
    
    def save_models(self, policy_path='model.pth', reward_path='reward_model.pth'):
        """Save the improved models, updating existing model in place"""
        try:
            # Update the existing model with improved weights
            torch.save(self.policy_model.model.state_dict(), policy_path)
            torch.save(self.reward_model.state_dict(), reward_path)
            self.log(f"✅ Model continuously improved and saved: {policy_path}")
            return True
        except Exception as e:
            self.log(f"❌ Error saving improved model: {e}")
            return False
    
    def get_training_summary(self):
        """Get summary of training history"""
        reward_epochs = [h for h in self.training_history if h['type'] == 'reward']
        policy_epochs = [h for h in self.training_history if h['type'] == 'policy']
        
        final_reward_loss = None
        final_policy_loss = None
        
        if len(reward_epochs) >= 1:
            final_reward_loss = reward_epochs[-1].get('reward_loss')
        
        if len(policy_epochs) >= 1:
            final_policy_loss = policy_epochs[-1].get('policy_loss')
        
        return {
            'total_epochs': len(self.training_history),
            'reward_epochs': len(reward_epochs),
            'policy_epochs': len(policy_epochs),
            'final_reward_loss': final_reward_loss,
            'final_policy_loss': final_policy_loss
        }

class PolicyModel(nn.Module):
    """Policy model that gets updated through RLHF"""
    def __init__(self, base_model=None):
        super(PolicyModel, self).__init__()
        if base_model is None:
            self.model = SimpleCNN(num_classes=2)
        else:
            self.model = copy.deepcopy(base_model)
    
    def forward(self, x):
        return self.model(x)
    
    def get_action_probs(self, x):
        """Get action probabilities for RL"""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

class RewardModel(nn.Module):
    """Reward model that learns to predict human preferences"""
    def __init__(self, base_model=None):
        super(RewardModel, self).__init__()
        if base_model is None:
            self.base_model = SimpleCNN(num_classes=2)
        else:
            self.base_model = copy.deepcopy(base_model)
        
        # Add reward head
        self.reward_head = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        base_output = self.base_model(x)
        reward = self.reward_head(base_output)
        return reward

class FeedbackDataset(Dataset):
    """Dataset for training the reward model"""
    def __init__(self, feedback_data, transform=None):
        self.feedback_data = feedback_data
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    
    def __len__(self):
        return len(self.feedback_data)
    
    def __getitem__(self, idx):
        feedback = self.feedback_data[idx]
        
        # Decode image
        image_data = base64.b64decode(feedback['image_data'])
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image_tensor = self.transform(image)
        
        # Create reward based on feedback
        model_pred = feedback['model_prediction']
        user_corr = feedback['user_correction']
        confidence = feedback['confidence']
        
        # Calculate reward: positive if model was wrong (user corrected)
        if model_pred != user_corr:
            # Model was wrong - negative reward
            reward = -1.0 * confidence  # Higher confidence when wrong = worse
        else:
            # Model was correct - positive reward
            reward = confidence  # Higher confidence when correct = better
        
        return image_tensor, torch.tensor([reward], dtype=torch.float32)

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='RLHF Training for DetectoReal')
    parser.add_argument('--reward-epochs', type=int, default=20, help='Number of reward model epochs')
    parser.add_argument('--policy-epochs', type=int, default=10, help='Number of policy model epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--policy-method', choices=['ppo', 'a2c', 'simple'], default='ppo', 
                       help='Policy training method')
    parser.add_argument('--device', default='cpu', help='Device to use for training')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("🤖 RLHF Training System")
    print("=" * 50)
    print(f"Reward epochs: {args.reward_epochs}")
    print(f"Policy epochs: {args.policy_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Policy method: {args.policy_method}")
    print(f"Device: {args.device}")
    
    # Load feedback data
    collector = EnhancedFeedbackCollector()
    training_data = collector.get_training_ready_data()
    
    if not training_data:
        print("❌ No high-quality feedback data found. Please collect some feedback first.")
        return False
    
    print(f"📊 Found {len(training_data)} high-quality feedback samples")
    
    # Analyze feedback quality
    analyzer = FeedbackAnalyzer()
    summary = analyzer.get_training_recommendations()
    
    if "performance" in summary:
        perf = summary["performance"]
        print(f"Current model accuracy: {perf.get('accuracy', 0):.2%}")
    
    # Initialize RLHF trainer
    trainer = RLHFTrainer(device=args.device, verbose=args.verbose)
    
    # Initialize models
    if not trainer.initialize_models():
        return False
    
    # Run RLHF training
    success = trainer.train_rlhf(
        training_data,
        reward_epochs=args.reward_epochs,
        policy_epochs=args.policy_epochs,
        batch_size=args.batch_size,
        policy_method=args.policy_method
    )
    
    if success:
        # Save improved model in place
        trainer.save_models()
        
        # Show training summary
        summary = trainer.get_training_summary()
        print(f"\n📈 Continuous Improvement Summary:")
        print(f"Total epochs: {summary['total_epochs']}")
        if summary['final_reward_loss'] is not None:
            print(f"Final reward loss: {summary['final_reward_loss']:.4f}")
        if summary['final_policy_loss'] is not None:
            print(f"Final policy loss: {summary['final_policy_loss']:.4f}")
        
        print("✅ Model continuously improved with user feedback")
        
        return True
    
    return False

def run_rlhf_training():
    """Function to run RLHF training that can be imported"""
    return main()

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 RLHF training completed successfully!")
    else:
        print("\n❌ RLHF training failed.") 