#!/usr/bin/env python3
"""
Automatic Learning System for Continuous Model Improvement
Implements the exact workflow you described:
1. User uploads image
2. Model predicts
3. User gives feedback
4. Model improves automatically
5. Same image predicts correctly next time
"""

import os
import json
import torch
import numpy as np
from datetime import datetime, timedelta
from PIL import Image
import hashlib
from collections import defaultdict
import threading
import time

from enhanced_feedback import EnhancedFeedbackCollector, FeedbackAnalyzer
from predict import load_prediction_model, predict_image_from_pil, get_detailed_analysis
from train_rlhf import run_rlhf_training

class AutomaticLearningSystem:
    """
    Automatic learning system that continuously improves the model
    based on user feedback
    """
    
    def __init__(self, feedback_threshold=10, auto_retrain=True, retrain_interval_hours=24):
        self.feedback_collector = EnhancedFeedbackCollector()
        self.feedback_analyzer = FeedbackAnalyzer()
        self.feedback_threshold = feedback_threshold  # Minimum feedback to trigger retraining
        self.auto_retrain = auto_retrain
        self.retrain_interval_hours = retrain_interval_hours
        self.last_retrain_time = None
        self.feedback_count = 0
        self.learning_history = []
        
        # Thread for background retraining
        self.retrain_thread = None
        self.retrain_lock = threading.Lock()
        
        # Load initial model
        self.model = load_prediction_model()
        
        print("🤖 Automatic Learning System initialized")
        print(f"📊 Feedback threshold: {self.feedback_threshold}")
        print(f"🔄 Auto-retrain: {self.auto_retrain}")
    
    def predict_and_collect_feedback(self, image, user_feedback=None, user_correction=None):
        """
        Main workflow: Predict image and collect feedback
        Returns: (prediction, confidence, feedback_saved, should_retrain)
        """
        try:
            # Step 1: Model makes prediction
            analysis = get_detailed_analysis(image)
            prediction = analysis["prediction"]
            confidence = analysis["confidence"]
            
            # Step 2: If user provides feedback, collect it
            feedback_saved = False
            if user_feedback is not None and user_correction is not None:
                feedback_saved = self._save_feedback(
                    image=image,
                    model_prediction=prediction,
                    user_correction=user_correction,
                    confidence=confidence,
                    feedback_type="correction"
                )
                
                if feedback_saved:
                    self.feedback_count += 1
                    print(f"✅ Feedback saved! Total feedback: {self.feedback_count}")
            
            # Step 3: Check if we should retrain
            should_retrain = self._should_retrain()
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "feedback_saved": feedback_saved,
                "should_retrain": should_retrain,
                "total_feedback": self.feedback_count
            }
            
        except Exception as e:
            print(f"❌ Error in predict_and_collect_feedback: {e}")
            return {
                "prediction": "error",
                "confidence": 0.0,
                "feedback_saved": False,
                "should_retrain": False,
                "error": str(e)
            }
    
    def _save_feedback(self, image, model_prediction, user_correction, confidence, feedback_type="correction"):
        """Save feedback with enhanced metadata"""
        try:
            # Create image hash for tracking
            image_hash = self._get_image_hash(image)
            
            # Save detailed feedback
            feedback_file = self.feedback_collector.save_detailed_feedback(
                image=image,
                model_prediction=model_prediction,
                user_correction=user_correction,
                confidence=confidence,
                feedback_type=feedback_type,
                user_confidence=8,  # High confidence correction
                additional_notes=f"Auto-learning feedback - Image hash: {image_hash}"
            )
            
            if feedback_file:
                # Record learning event
                self.learning_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "image_hash": image_hash,
                    "model_prediction": model_prediction,
                    "user_correction": user_correction,
                    "confidence": confidence,
                    "feedback_file": feedback_file
                })
                
                print(f"📝 Learning event recorded: {model_prediction} → {user_correction}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error saving feedback: {e}")
            return False
    
    def _get_image_hash(self, image):
        """Create a hash of the image for tracking"""
        try:
            # Convert to bytes and hash
            img_bytes = image.tobytes()
            return hashlib.md5(img_bytes).hexdigest()[:8]
        except:
            return "unknown"
    
    def _should_retrain(self):
        """Check if we should trigger automatic retraining"""
        if not self.auto_retrain:
            return False
        
        # Check feedback threshold
        if self.feedback_count < self.feedback_threshold:
            return False
        
        # Check time interval
        if self.last_retrain_time is not None:
            time_since_retrain = datetime.now() - self.last_retrain_time
            if time_since_retrain < timedelta(hours=self.retrain_interval_hours):
                return False
        
        return True
    
    def trigger_retraining(self, method="rlhf"):
        """
        Trigger model retraining based on collected feedback
        method: "rlhf" or "standard"
        """
        with self.retrain_lock:
            if self.retrain_thread and self.retrain_thread.is_alive():
                print("🔄 Retraining already in progress...")
                return False
            
            print(f"🚀 Starting automatic retraining with {method}...")
            
            # Start retraining in background thread
            self.retrain_thread = threading.Thread(
                target=self._retrain_model,
                args=(method,)
            )
            self.retrain_thread.start()
            
            return True
    
    def _retrain_model(self, method):
        """Background retraining process"""
        try:
            start_time = datetime.now()
            
            if method == "rlhf":
                success = run_rlhf_training()
                method_name = "RLHF"
            else:
                from retrain_model import retrain_model_with_feedback
                success = retrain_model_with_feedback()
                method_name = "Standard"
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if success:
                self.last_retrain_time = end_time
                self.feedback_count = 0  # Reset counter
                
                # Reload model
                self.model = load_prediction_model()
                
                print(f"✅ {method_name} retraining completed in {duration:.1f}s")
                print("🔄 Model updated and ready for new predictions!")
                
                # Record retraining event
                self.learning_history.append({
                    "timestamp": end_time.isoformat(),
                    "event": "retraining_completed",
                    "method": method_name,
                    "duration_seconds": duration,
                    "feedback_used": self.feedback_count
                })
                
            else:
                print(f"❌ {method_name} retraining failed")
                
        except Exception as e:
            print(f"❌ Error during retraining: {e}")
    
    def test_same_image_prediction(self, image, expected_prediction):
        """
        Test if the same image now predicts correctly after retraining
        Returns: (correct_prediction, confidence, improvement)
        """
        try:
            # Get current prediction
            analysis = get_detailed_analysis(image)
            current_prediction = analysis["prediction"]
            current_confidence = analysis["confidence"]
            
            # Check if prediction is correct
            correct_prediction = (current_prediction == expected_prediction)
            
            # Find previous prediction for this image
            image_hash = self._get_image_hash(image)
            previous_prediction = None
            
            for event in reversed(self.learning_history):
                if event.get("image_hash") == image_hash and "model_prediction" in event:
                    previous_prediction = event["model_prediction"]
                    break
            
            improvement = None
            if previous_prediction:
                improvement = {
                    "previous": previous_prediction,
                    "current": current_prediction,
                    "expected": expected_prediction,
                    "improved": (previous_prediction != expected_prediction and current_prediction == expected_prediction)
                }
            
            return {
                "correct_prediction": correct_prediction,
                "prediction": current_prediction,
                "confidence": current_confidence,
                "improvement": improvement
            }
            
        except Exception as e:
            print(f"❌ Error testing same image prediction: {e}")
            return {
                "correct_prediction": False,
                "prediction": "error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def get_learning_stats(self):
        """Get statistics about the learning process"""
        stats = {
            "total_feedback": self.feedback_count,
            "feedback_threshold": self.feedback_threshold,
            "auto_retrain": self.auto_retrain,
            "learning_events": len(self.learning_history),
            "last_retrain": self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            "retraining_in_progress": self.retrain_thread.is_alive() if self.retrain_thread else False
        }
        
        # Analyze learning history
        if self.learning_history:
            corrections = [e for e in self.learning_history if "user_correction" in e]
            retrainings = [e for e in self.learning_history if e.get("event") == "retraining_completed"]
            
            stats.update({
                "total_corrections": len(corrections),
                "total_retrainings": len(retrainings),
                "last_correction": corrections[-1]["timestamp"] if corrections else None,
                "last_retraining": retrainings[-1]["timestamp"] if retrainings else None
            })
        
        return stats
    
    def get_performance_improvement(self):
        """Analyze performance improvement over time"""
        try:
            analyzer = FeedbackAnalyzer()
            performance = analyzer.analyze_model_performance()
            
            if performance:
                return {
                    "current_accuracy": performance.get("accuracy", 0),
                    "total_feedback": performance.get("total_feedback", 0),
                    "high_quality_feedback": performance.get("high_quality_feedback", 0),
                    "recent_accuracy": performance.get("recent_accuracy", 0),
                    "improvement_trend": "positive" if performance.get("recent_accuracy", 0) > performance.get("accuracy", 0) else "stable"
                }
            
            return {"error": "No performance data available"}
            
        except Exception as e:
            return {"error": f"Error analyzing performance: {e}"}

# Example usage and testing
def test_automatic_learning_workflow():
    """Test the complete automatic learning workflow"""
    print("🧪 Testing Automatic Learning Workflow")
    print("=" * 50)
    
    # Initialize the system
    als = AutomaticLearningSystem(feedback_threshold=5, auto_retrain=True)
    
    # Create a test image (you would normally upload a real image)
    test_image = Image.new('RGB', (128, 128), color='red')
    
    # Step 1: Initial prediction
    print("\n1. Initial prediction...")
    result1 = als.predict_and_collect_feedback(test_image)
    print(f"   Prediction: {result1['prediction']} (confidence: {result1['confidence']:.3f})")
    
    # Step 2: User provides feedback (simulating wrong prediction)
    print("\n2. User provides feedback...")
    result2 = als.predict_and_collect_feedback(
        test_image, 
        user_feedback="This is wrong", 
        user_correction="fake"  # User says it's actually AI-generated
    )
    print(f"   Feedback saved: {result2['feedback_saved']}")
    print(f"   Should retrain: {result2['should_retrain']}")
    
    # Step 3: Trigger retraining (if enough feedback)
    if result2['should_retrain']:
        print("\n3. Triggering automatic retraining...")
        als.trigger_retraining(method="rlhf")
        
        # Wait a bit for retraining (in real scenario, this would be async)
        time.sleep(2)
    
    # Step 4: Test same image prediction
    print("\n4. Testing same image prediction...")
    test_result = als.test_same_image_prediction(test_image, "fake")
    print(f"   Correct prediction: {test_result['correct_prediction']}")
    print(f"   Current prediction: {test_result['prediction']}")
    
    if test_result['improvement']:
        print(f"   Improvement: {test_result['improvement']['previous']} → {test_result['improvement']['current']}")
        print(f"   Improved: {test_result['improvement']['improved']}")
    
    # Step 5: Get learning statistics
    print("\n5. Learning statistics...")
    stats = als.get_learning_stats()
    print(f"   Total feedback: {stats['total_feedback']}")
    print(f"   Learning events: {stats['learning_events']}")
    print(f"   Auto-retrain: {stats['auto_retrain']}")
    
    performance = als.get_performance_improvement()
    if "error" not in performance:
        print(f"   Current accuracy: {performance['current_accuracy']:.1%}")
        print(f"   Improvement trend: {performance['improvement_trend']}")
    
    print("\n✅ Automatic learning workflow test completed!")

if __name__ == "__main__":
    test_automatic_learning_workflow() 