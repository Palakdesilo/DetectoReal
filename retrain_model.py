import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import json
import os
import base64
from io import BytesIO
import numpy as np
from model import SimpleCNN
import glob

def load_feedback_data():
    """Load all feedback data from the feedback_data directory"""
    feedback_files = glob.glob("feedback_data/feedback_*.json")
    feedback_data = []
    
    for file_path in feedback_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                feedback_data.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return feedback_data

def decode_base64_image(image_base64):
    """Convert base64 string back to PIL Image"""
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data)).convert('RGB')
    return image

def prepare_training_data(feedback_data):
    """Prepare training data from feedback"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    images = []
    labels = []
    
    for feedback in feedback_data:
        try:
            # Decode image from base64
            image = decode_base64_image(feedback['image_data'])
            image_tensor = transform(image)
            
            # Get correct label from user correction
            label = 0 if feedback['user_correction'] == 'fake' else 1  # 0=fake, 1=real
            
            images.append(image_tensor)
            labels.append(label)
            
        except Exception as e:
            print(f"Error processing feedback: {e}")
            continue
    
    if images:
        return torch.stack(images), torch.tensor(labels, dtype=torch.long)
    else:
        return None, None

def retrain_model_with_feedback():
    """Retrain the model using collected feedback"""
    print("🔄 Starting model retraining with feedback data...")
    
    # Load feedback data
    feedback_data = load_feedback_data()
    
    if not feedback_data:
        print("❌ No feedback data found. Please collect some feedback first.")
        return False
    
    print(f"📊 Found {len(feedback_data)} feedback samples")
    
    # Prepare training data
    images, labels = prepare_training_data(feedback_data)
    
    if images is None:
        print("❌ No valid training data could be prepared from feedback.")
        return False
    
    # Load existing model
    try:
        model = SimpleCNN(num_classes=2)
        model.load_state_dict(torch.load('model.pth', map_location='cpu'))
        print("✅ Loaded existing model")
    except Exception as e:
        print(f"❌ Error loading existing model: {e}")
        return False
    
    # Set up training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    batch_size = min(8, len(images))  # Use smaller batch size for feedback data
    
    print(f"🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size].to(device)
            batch_labels = labels[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / (len(images) // batch_size + 1)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")
    
    # Save retrained model
    try:
        torch.save(model.state_dict(), 'model_retrained.pth')
        print("✅ Retrained model saved as 'model_retrained.pth'")
        
        # Optionally replace the original model
        backup_original = input("Do you want to replace the original model? (y/n): ").lower()
        if backup_original == 'y':
            # Backup original model
            if os.path.exists('model.pth'):
                os.rename('model.pth', 'model_backup.pth')
                print("📦 Original model backed up as 'model_backup.pth'")
            
            # Replace with retrained model
            os.rename('model_retrained.pth', 'model.pth')
            print("✅ Original model replaced with retrained version")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving retrained model: {e}")
        return False

def analyze_feedback():
    """Analyze collected feedback data"""
    feedback_data = load_feedback_data()
    
    if not feedback_data:
        print("❌ No feedback data found.")
        return
    
    print(f"\n📊 Feedback Analysis:")
    print(f"Total feedback samples: {len(feedback_data)}")
    
    # Analyze corrections
    corrections = {}
    for feedback in feedback_data:
        model_pred = feedback['model_prediction']
        user_corr = feedback['user_correction']
        
        key = f"{model_pred} -> {user_corr}"
        corrections[key] = corrections.get(key, 0) + 1
    
    print("\n🔍 Correction Patterns:")
    for pattern, count in corrections.items():
        print(f"  {pattern}: {count} times")
    
    # Analyze confidence levels
    confidences = [f['confidence'] for f in feedback_data]
    avg_confidence = np.mean(confidences)
    print(f"\n📈 Average confidence of incorrect predictions: {avg_confidence:.3f}")

if __name__ == "__main__":
    print("🤖 Model Retraining Tool")
    print("=" * 40)
    
    # Analyze existing feedback
    analyze_feedback()
    
    # Ask user if they want to retrain
    retrain = input("\nDo you want to retrain the model with feedback? (y/n): ").lower()
    
    if retrain == 'y':
        success = retrain_model_with_feedback()
        if success:
            print("\n🎉 Model retraining completed successfully!")
        else:
            print("\n❌ Model retraining failed.")
    else:
        print("\n👋 Retraining cancelled.") 