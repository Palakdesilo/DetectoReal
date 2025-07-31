import json
import datetime
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np

class EnhancedFeedbackCollector:
    """
    Enhanced feedback collection system for RLHF
    """
    
    def __init__(self, feedback_dir="feedback_data"):
        self.feedback_dir = feedback_dir
        os.makedirs(feedback_dir, exist_ok=True)
    
    def save_detailed_feedback(self, image, model_prediction, user_correction, 
                              confidence, feedback_type="correction", 
                              user_confidence=None, additional_notes=""):
        """
        Save detailed feedback with additional metadata
        
        Args:
            image: PIL Image object
            model_prediction: Model's prediction ('fake' or 'real')
            user_correction: User's correction ('fake' or 'real')
            confidence: Model's confidence score
            feedback_type: Type of feedback ('correction', 'confirmation', 'uncertain')
            user_confidence: User's confidence in their correction (1-10)
            additional_notes: Any additional notes from user
        """
        
        # Encode image to base64
        image_base64 = self._encode_image_to_base64(image)
        
        # Create detailed feedback data
        feedback_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_prediction": model_prediction,
            "user_correction": user_correction,
            "confidence": confidence,
            "feedback_type": feedback_type,
            "user_confidence": user_confidence,
            "additional_notes": additional_notes,
            "image_data": image_base64,
            "metadata": {
                "image_size": image.size,
                "image_mode": image.mode,
                "correction_needed": model_prediction != user_correction,
                "confidence_level": self._get_confidence_level(confidence),
                "feedback_quality": self._calculate_feedback_quality(
                    confidence, user_confidence, feedback_type
                )
            }
        }
        
        # Save to file
        filename = f"feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        filepath = os.path.join(self.feedback_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        return filepath
    
    def _encode_image_to_base64(self, image):
        """Convert PIL image to base64 string"""
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def _get_confidence_level(self, confidence):
        """Get confidence level description"""
        if confidence > 0.8:
            return "high"
        elif confidence > 0.6:
            return "medium"
        else:
            return "low"
    
    def _calculate_feedback_quality(self, model_confidence, user_confidence, feedback_type):
        """Calculate quality score for feedback"""
        quality_score = 0
        
        # Base quality based on feedback type
        if feedback_type == "correction":
            quality_score += 5  # Corrections are most valuable
        elif feedback_type == "confirmation":
            quality_score += 3  # Confirmations are good
        else:
            quality_score += 1  # Uncertain feedback is least valuable
        
        # Adjust based on model confidence when wrong
        if model_confidence > 0.7:  # High confidence mistakes are more valuable
            quality_score += 2
        
        # Adjust based on user confidence
        if user_confidence and user_confidence > 7:
            quality_score += 2
        elif user_confidence and user_confidence < 4:
            quality_score -= 1
        
        return min(10, max(1, quality_score))  # Clamp between 1-10
    
    def get_feedback_statistics(self):
        """Get statistics about collected feedback"""
        feedback_files = [f for f in os.listdir(self.feedback_dir) 
                         if f.startswith('feedback_') and f.endswith('.json')]
        
        if not feedback_files:
            return {"total_feedback": 0, "message": "No feedback data found"}
        
        total_feedback = len(feedback_files)
        corrections = 0
        confirmations = 0
        uncertain = 0
        quality_scores = []
        
        for filename in feedback_files:
            filepath = os.path.join(self.feedback_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                feedback_type = data.get('feedback_type', 'correction')
                if feedback_type == 'correction':
                    corrections += 1
                elif feedback_type == 'confirmation':
                    confirmations += 1
                else:
                    uncertain += 1
                
                quality = data.get('metadata', {}).get('feedback_quality', 5)
                quality_scores.append(quality)
                
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        return {
            "total_feedback": total_feedback,
            "corrections": corrections,
            "confirmations": confirmations,
            "uncertain": uncertain,
            "average_quality": avg_quality,
            "correction_rate": corrections / total_feedback if total_feedback > 0 else 0
        }
    
    def get_training_ready_data(self):
        """Get feedback data ready for RLHF training"""
        feedback_files = [f for f in os.listdir(self.feedback_dir) 
                         if f.startswith('feedback_') and f.endswith('.json')]
        
        training_data = []
        
        for filename in feedback_files:
            filepath = os.path.join(self.feedback_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Only include high-quality feedback for training
                quality = data.get('metadata', {}).get('feedback_quality', 5)
                if quality >= 5:  # Only include medium-high quality feedback
                    training_data.append(data)
                    
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        
        return training_data

class FeedbackAnalyzer:
    """
    Analyze feedback data for insights
    """
    
    def __init__(self, feedback_dir="feedback_data"):
        self.feedback_dir = feedback_dir
    
    def analyze_model_performance(self):
        """Analyze model performance based on feedback"""
        feedback_files = [f for f in os.listdir(self.feedback_dir) 
                         if f.startswith('feedback_') and f.endswith('.json')]
        
        if not feedback_files:
            return {"message": "No feedback data available"}
        
        performance_data = {
            "total_predictions": len(feedback_files),
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "accuracy": 0,
            "confidence_analysis": {
                "high_confidence_correct": 0,
                "high_confidence_incorrect": 0,
                "low_confidence_correct": 0,
                "low_confidence_incorrect": 0
            },
            "error_patterns": {}
        }
        
        for filename in feedback_files:
            filepath = os.path.join(self.feedback_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                model_pred = data['model_prediction']
                user_corr = data['user_correction']
                confidence = data['confidence']
                
                # Check if prediction was correct
                if model_pred == user_corr:
                    performance_data['correct_predictions'] += 1
                    if confidence > 0.7:
                        performance_data['confidence_analysis']['high_confidence_correct'] += 1
                    else:
                        performance_data['confidence_analysis']['low_confidence_correct'] += 1
                else:
                    performance_data['incorrect_predictions'] += 1
                    if confidence > 0.7:
                        performance_data['confidence_analysis']['high_confidence_incorrect'] += 1
                    else:
                        performance_data['confidence_analysis']['low_confidence_incorrect'] += 1
                
                # Track error patterns
                if model_pred != user_corr:
                    error_key = f"{model_pred}_predicted_as_{user_corr}"
                    performance_data['error_patterns'][error_key] = \
                        performance_data['error_patterns'].get(error_key, 0) + 1
                
            except Exception as e:
                print(f"Error analyzing {filename}: {e}")
        
        # Calculate accuracy
        total = performance_data['total_predictions']
        if total > 0:
            performance_data['accuracy'] = performance_data['correct_predictions'] / total
        
        return performance_data
    
    def get_training_recommendations(self):
        """Get recommendations for model training based on feedback analysis"""
        performance = self.analyze_model_performance()
        
        if "message" in performance:
            return {"message": "No data available for recommendations"}
        
        recommendations = []
        
        # Analyze accuracy
        accuracy = performance['accuracy']
        if accuracy < 0.7:
            recommendations.append({
                "type": "accuracy",
                "priority": "high",
                "message": f"Model accuracy is low ({accuracy:.2%}). Consider retraining with more diverse data."
            })
        
        # Analyze confidence issues
        high_conf_incorrect = performance['confidence_analysis']['high_confidence_incorrect']
        if high_conf_incorrect > 0:
            recommendations.append({
                "type": "confidence",
                "priority": "medium",
                "message": f"Model made {high_conf_incorrect} high-confidence errors. Consider confidence calibration."
            })
        
        # Analyze error patterns
        error_patterns = performance['error_patterns']
        if error_patterns:
            most_common_error = max(error_patterns.items(), key=lambda x: x[1])
            recommendations.append({
                "type": "bias",
                "priority": "medium",
                "message": f"Most common error: {most_common_error[0]} ({most_common_error[1]} times). Consider bias correction."
            })
        
        return {
            "performance": performance,
            "recommendations": recommendations
        }

def create_feedback_summary():
    """Create a summary of all feedback data"""
    collector = EnhancedFeedbackCollector()
    analyzer = FeedbackAnalyzer()
    
    stats = collector.get_feedback_statistics()
    recommendations = analyzer.get_training_recommendations()
    
    summary = {
        "feedback_statistics": stats,
        "training_recommendations": recommendations,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    return summary

if __name__ == "__main__":
    # Test the feedback system
    summary = create_feedback_summary()
    print("📊 Feedback Summary:")
    print(json.dumps(summary, indent=2)) 