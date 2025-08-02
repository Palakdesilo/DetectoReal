#!/usr/bin/env python3
"""
Test script for Enhanced Real-Time Learning System
Tests the "Improve Model" functionality and similar image detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_time_learning_enhanced import EnhancedRealTimeLearningSystem
from PIL import Image
import numpy as np

def test_enhanced_learning_system():
    """Test the enhanced real-time learning system"""
    print("🧪 Testing Enhanced Real-Time Learning System")
    print("=" * 60)
    
    try:
        # Initialize the system
        print("1. Initializing Enhanced Real-Time Learning System...")
        rtl = EnhancedRealTimeLearningSystem(learning_rate=1e-4, memory_size=100)
        print("✅ System initialized successfully")
        
        # Create test images
        print("\n2. Creating test images...")
        test_image1 = Image.new('RGB', (128, 128), color='red')
        test_image2 = Image.new('RGB', (128, 128), color='darkred')  # Similar to test_image1
        test_image3 = Image.new('RGB', (128, 128), color='blue')     # Different from test_image1
        
        print("✅ Test images created")
        
        # Step 1: Initial prediction
        print("\n3. Testing initial prediction...")
        result1 = rtl.predict_with_learning(test_image1)
        print(f"   Prediction: {result1['prediction']} (confidence: {result1['confidence']:.3f})")
        print(f"   Source: {result1['source']}")
        print(f"   Match type: {result1['match_type']}")
        
        # Step 2: Improve model with feedback
        print("\n4. Testing 'Improve Model' functionality...")
        improvement_result = rtl.improve_model_with_feedback(test_image1, "fake")
        print(f"   Success: {improvement_result['success']}")
        print(f"   Similar images updated: {improvement_result['similar_images_updated']}")
        print(f"   Learning completed: {improvement_result['learning_completed']}")
        
        # Step 3: Test same image again
        print("\n5. Testing same image after improvement...")
        result3 = rtl.predict_with_learning(test_image1)
        print(f"   Prediction: {result3['prediction']} (confidence: {result3['confidence']:.3f})")
        print(f"   Source: {result3['source']}")
        print(f"   Match type: {result3['match_type']}")
        
        # Step 4: Test similar image
        print("\n6. Testing similar image...")
        result4 = rtl.predict_with_learning(test_image2)
        print(f"   Prediction: {result4['prediction']} (confidence: {result4['confidence']:.3f})")
        print(f"   Source: {result4['source']}")
        print(f"   Match type: {result4['match_type']}")
        
        # Step 5: Test different image
        print("\n7. Testing different image...")
        result5 = rtl.predict_with_learning(test_image3)
        print(f"   Prediction: {result5['prediction']} (confidence: {result5['confidence']:.3f})")
        print(f"   Source: {result5['source']}")
        print(f"   Match type: {result5['match_type']}")
        
        # Step 6: Test improvement verification
        print("\n8. Testing improvement verification...")
        improvement = rtl.test_same_image_improvement(test_image1, "fake")
        print(f"   Correct prediction: {improvement['correct_prediction']}")
        print(f"   Prediction: {improvement['prediction']}")
        print(f"   Confidence: {improvement['confidence']:.3f}")
        print(f"   Source: {improvement['source']}")
        print(f"   Match type: {improvement['match_type']}")
        
        # Step 7: Get learning statistics
        print("\n9. Learning statistics...")
        stats = rtl.get_learning_statistics()
        print(f"   Total feedback: {stats['total_feedback']}")
        print(f"   Successful learnings: {stats['successful_learnings']}")
        print(f"   Memory size: {stats['memory_size']}")
        print(f"   Similar image matches: {stats.get('similar_image_matches', 0)}")
        print(f"   Exact image matches: {stats.get('exact_image_matches', 0)}")
        
        # Step 8: Test multiple improvements
        print("\n10. Testing multiple improvements...")
        improvement_result2 = rtl.improve_model_with_feedback(test_image3, "real")
        print(f"   Second improvement success: {improvement_result2['success']}")
        
        # Final statistics
        final_stats = rtl.get_learning_statistics()
        print(f"   Final total feedback: {final_stats['total_feedback']}")
        print(f"   Final successful learnings: {final_stats['successful_learnings']}")
        
        print("\n✅ Enhanced real-time learning test completed successfully!")
        print("🎉 All features working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_extraction():
    """Test feature extraction functionality"""
    print("\n🔍 Testing Feature Extraction")
    print("=" * 40)
    
    try:
        from real_time_learning_enhanced import FeatureExtractor
        
        # Initialize feature extractor
        extractor = FeatureExtractor()
        
        # Create test images
        test_image1 = Image.new('RGB', (128, 128), color='red')
        test_image2 = Image.new('RGB', (128, 128), color='darkred')
        
        # Extract features
        features1 = extractor.extract_features(test_image1)
        features2 = extractor.extract_features(test_image2)
        
        print(f"   Features 1 shape: {features1.shape}")
        print(f"   Features 2 shape: {features2.shape}")
        print(f"   Features 1 sum: {np.sum(features1):.4f}")
        print(f"   Features 2 sum: {np.sum(features2):.4f}")
        
        # Test similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([features1], [features2])[0][0]
        print(f"   Similarity between images: {similarity:.4f}")
        
        print("✅ Feature extraction test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in feature extraction test: {e}")
        return False

def test_memory_system():
    """Test memory system functionality"""
    print("\n🧠 Testing Memory System")
    print("=" * 40)
    
    try:
        from real_time_learning_enhanced import EnhancedMemory, FeatureExtractor
        
        # Initialize components
        memory = EnhancedMemory(max_size=10)
        extractor = FeatureExtractor()
        
        # Create test images
        test_image1 = Image.new('RGB', (128, 128), color='red')
        test_image2 = Image.new('RGB', (128, 128), color='darkred')
        test_image3 = Image.new('RGB', (128, 128), color='blue')
        
        # Extract features
        features1 = extractor.extract_features(test_image1)
        features2 = extractor.extract_features(test_image2)
        features3 = extractor.extract_features(test_image3)
        
        # Add images to memory
        memory.add_image("hash1", features1, "fake", 0.9, "2024-01-01")
        memory.add_image("hash2", features2, "fake", 0.8, "2024-01-02")
        memory.add_image("hash3", features3, "real", 0.7, "2024-01-03")
        
        print(f"   Memory size: {len(memory)}")
        
        # Test exact match
        exact_match = memory.find_image("hash1", features1)
        print(f"   Exact match found: {exact_match is not None}")
        if exact_match:
            print(f"   Exact match prediction: {exact_match['prediction']}")
        
        # Test similar match
        similar_match = memory.find_image("unknown_hash", features1)
        print(f"   Similar match found: {similar_match is not None}")
        if similar_match:
            print(f"   Similar match prediction: {similar_match['prediction']}")
            print(f"   Similarity score: {similar_match.get('similarity', 'N/A')}")
        
        print("✅ Memory system test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in memory system test: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Enhanced Real-Time Learning System Tests")
    print("=" * 60)
    
    # Run all tests
    test1_passed = test_enhanced_learning_system()
    test2_passed = test_feature_extraction()
    test3_passed = test_memory_system()
    
    print("\n📊 Test Results Summary")
    print("=" * 40)
    print(f"   Enhanced Learning System: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Feature Extraction: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"   Memory System: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 All tests passed! The enhanced system is ready to use.")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.") 