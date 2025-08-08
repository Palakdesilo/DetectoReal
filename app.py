#!/usr/bin/env python3
"""
DetectoReal - AI Image Authenticity Detection
Unified UI Design
"""

import streamlit as st
import os
import json
from PIL import Image
import base64
from io import BytesIO
import datetime
import time
from enhanced_feedback import EnhancedFeedbackCollector, FeedbackAnalyzer
from real_time_learning_enhanced_simple import RLHFImageClassifier

# Initialize RLHF image classifier
if 'real_time_learning_system' not in st.session_state:
    st.session_state.real_time_learning_system = RLHFImageClassifier(
        learning_rate=1e-4,
        memory_size=1000
    )
    
    # Verify model loading after initialization
    rtl = st.session_state.real_time_learning_system
    verification = rtl.verify_model_loading()
    print(f"🔍 Initial model verification: {verification}")
    
    # CRITICAL FIX: Force reload learned model if session state was cleared
    if verification.get('learned_model_exists', False) and not verification.get('session_has_model', False):
        print("🔄 Session state cleared but learned model exists - force reloading...")
        rtl.force_reload_learned_model()
else:
    # Verify model loading for existing instance
    rtl = st.session_state.real_time_learning_system
    verification = rtl.verify_model_loading()
    print(f"🔍 Existing model verification: {verification}")

# Check if model has been improved (for internal use only)
model_improved = os.path.exists('learned_model.pth')

# Page configuration
st.set_page_config(
    page_title="DetectoReal - AI Image Authenticity Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide the sidebar completely
st.markdown("""
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Unified CSS with dark theme
st.markdown("""
<style>
    /* CSS Variables for Neutral Color Palette */
    :root {
        --primary-blue: #4a90e2;
        --primary-teal: #38b2ac;
        --primary-purple: #805ad5;
        --accent-blue: #3182ce;
        --accent-teal: #319795;
        --accent-purple: #6b46c1;
        
        /* Neutral Grays */
        --gray-50: #f9fafb;
        --gray-100: #f3f4f6;
        --gray-200: #e5e7eb;
        --gray-300: #d1d5db;
        --gray-400: #9ca3af;
        --gray-500: #6b7280;
        --gray-600: #4b5563;
        --gray-700: #374151;
        --gray-800: #1f2937;
        --gray-900: #111827;
        
        /* Off-whites and Mid-tones */
        --off-white: #fafafa;
        --warm-white: #fefefe;
        --cool-white: #f8fafc;
        --mid-gray: #64748b;
        --light-gray: #94a3b8;
        
        /* Status Colors */
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --info: #3b82f6;
        
        /* Background Gradients */
        --bg-gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --bg-gradient-secondary: linear-gradient(135deg, #4a90e2 0%, #38b2ac 100%);
        --bg-gradient-tertiary: linear-gradient(135deg, #805ad5 0%, #4a90e2 100%);
        
        /* Text Colors */
        --text-primary: #f8fafc;
        --text-secondary: #e2e8f0;
        --text-muted: #94a3b8;
        --text-inverse: #1f2937;
    }

    /* Base styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-size: 16px;
        font-family: "Source Sans", sans-serif;
        font-weight: 400;
        line-height: 1.6;
        color: var(--text-primary) !important;
        background: var(--bg-gradient-primary) !important;
    }

    /* Force dark theme */
    .stApp {
        background: var(--bg-gradient-primary) !important;
        color: var(--text-primary) !important;
    }

    .main .block-container {
        background: transparent !important;
        color: var(--text-primary) !important;
    }

    .stMarkdown, .stText, .stButton, .stSelectbox, .stFileUploader {
        color: var(--text-primary) !important;
    }

    .main {
        background: transparent;
        min-height: 100vh;
        padding: 2rem;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary);
        font-weight: 700;
        line-height: 1.2;
    }

    h1 {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
        background: var(--bg-gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    h2 {
        font-size: 1.875rem;
        margin-bottom: 1rem;
    }

    h3 {
        font-size: 1.5rem;
        margin-bottom: 0.75rem;
    }

    p {
        color: var(--text-secondary);
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }

    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }

    .status-active {
        background: var(--success);
        box-shadow: 0 0 8px var(--success);
    }

    .status-inactive {
        background: var(--error);
        box-shadow: 0 0 8px var(--error);
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.1); }
    }

    /* Cards */
    .card {
        background: rgba(31, 41, 55, 0.95);
        border: 1px solid var(--gray-600);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }

    .card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
    }

    /* Upload area */
    .upload-area {
        border: 2px dashed var(--primary-blue);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: rgba(31, 41, 55, 0.8);
        transition: all 0.3s ease;
        margin: 1rem 0;
    }

    .upload-area:hover {
        border-color: var(--primary-purple);
        background: rgba(31, 41, 55, 0.95);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: var(--off-white);
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #805ad5, #6b46c1);
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Messages */
    .message {
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
        animation: slideIn 0.4s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
        border-left: 4px solid;
    }
    
    .message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transform: translateX(-100%);
        animation: shimmer 2s infinite;
    }

    .message-success {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(34, 197, 94, 0.05));
        border-color: #22c55e;
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }

    .message-error {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05));
        border-color: #ef4444;
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }

    .message-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.05));
        border-color: #f59e0b;
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Prediction results */
    .prediction-result {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.125rem;
        font-weight: 700;
        margin: 1rem 0;
        animation: fadeInUp 0.5s ease;
    }

    .prediction-fake {
        color: darkred;
    }

    .prediction-real {
        color: lightgreen;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* File info */
    .file-info {
        background: rgba(45, 55, 72, 0.8);
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.875rem;
    }

    /* Features grid */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }

    .feature-item {
        background: rgba(45, 55, 72, 0.95);
        border: 1px solid #4a5568;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .feature-item:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        
        h1 {
            font-size: 2rem;
        }
        
        .upload-area {
            padding: 1.5rem;
        }
        
        .features-grid {
            grid-template-columns: 1fr;
        }
    }

    /* Hide Streamlit elements */
    .stDeployButton {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="
    padding: 1.5rem 2rem 7rem 2rem;
    margin: 0;
    text-align: center;
">
    <div style="color: #ffffff; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.2rem;">🕵️‍♂️ DetectoReal</div>
    <div style="color: #ffffff; font-size: 1.2rem;">An AI-powered tool to detect fake vs real images.</div>
</div>
""", unsafe_allow_html=True)

# Upload section
uploaded_file = st.file_uploader(
    "Choose an image file (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload any image to analyze if it's real or AI-generated"
)

# Background debugging (hidden from user interface)
rtl = st.session_state.real_time_learning_system
verification = rtl.verify_model_loading()
print(f"🔍 Background model verification: {verification}")

# Analysis section
if uploaded_file is not None:
    # Track current uploaded file to detect new uploads
    current_file_key = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.type}"
    
    # Check if this is a new image upload
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    
    if st.session_state.last_uploaded_file != current_file_key:
        # New image uploaded - completely reset all previous processing state
        st.session_state.show_improve_section = False
        st.session_state.training_active = False
        st.session_state.prediction_updated = False
        st.session_state.updated_prediction = None
        st.session_state.training_image = None
        st.session_state.training_correction = None
        st.session_state.last_uploaded_file = current_file_key
        
        # Force a clean state for new image processing
        st.rerun()
    
    with st.spinner("🔍 Analyzing image..."):
        time.sleep(0.5)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📸 Image Analysis")
        
        # Display image
        image = Image.open(uploaded_file)
        resized_image = image.resize((200, 200))
        st.image(resized_image, caption="Uploaded Image")
        
        # Get prediction with real-time learning
        rtl = st.session_state.real_time_learning_system
        result = rtl.predict_with_learning(image)
        
        prediction = result["prediction"]
        
        # Check if prediction was updated after training
        if 'prediction_updated' in st.session_state and st.session_state.prediction_updated:
            # Use the updated prediction from training
            updated_prediction = st.session_state.updated_prediction
            prediction_text = "🔴 FAKE DETECTED" if updated_prediction == "fake" else "🟢 REAL DETECTED"
            prediction_class = "prediction-fake" if updated_prediction == "fake" else "prediction-real"
            
            st.markdown(f"""
            <div class="prediction-result {prediction_class}" style="padding: 2rem; text-align: center; margin: 2rem 0; border: 3px solid {'#ef4444' if updated_prediction == 'fake' else '#10b981'}; border-radius: 12px; background: rgba(31, 41, 55, 0.95);">
                <div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                    {prediction_text}
                </div>
                <div style="font-size: 1.2rem; opacity: 0.9; margin-top: 0.5rem;">
                    ✅ Result Updated After Training
                </div>
                <div style="font-size: 1rem; opacity: 0.7; margin-top: 0.5rem;">
                    Model learned from your feedback
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Reset the update flag
            st.session_state.prediction_updated = False
        else:
            # Show original prediction
            prediction_text = "🔴 FAKE DETECTED" if prediction == "fake" else "🟢 REAL DETECTED"
            prediction_class = "prediction-fake" if prediction == "fake" else "prediction-real"
            
            st.markdown(f"""
            <div class="prediction-result {prediction_class}" style="padding: 2rem; text-align: center; margin: 2rem 0;">
                <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">
                    {prediction_text}
                </div>
                <div style="font-size: 1rem; opacity: 0.8; margin-top: 0.5rem;">
                    AI-powered detection result
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🤔 Was this prediction correct?")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
             if st.button("✅ Correct", key="correct_btn", use_container_width=True):
                 st.markdown("""
                 <div class="message message-success">
                     <div style="display: flex; align-items: center; gap: 0.5rem;">
                         <span style="font-size: 1.5rem;">✅</span>
                         <div>
                             <div style="font-weight: 700; margin-bottom: 0.25rem;">Feedback Received!</div>
                             <div style="font-size: 0.9rem; opacity: 0.9;">Thank you for the confirmation</div>
                         </div>
                     </div>
                 </div>
                 """, unsafe_allow_html=True)
                 
                 rtl = st.session_state.real_time_learning_system
                 feedback_result = rtl.predict_with_learning(
                     image=image,
                     user_feedback="Correct prediction",
                     user_correction=prediction
                 )
        
        with col_b:
            if 'show_improve_section' not in st.session_state:
                st.session_state.show_improve_section = False
            
            if st.button("❌ Incorrect", key="incorrect_btn", use_container_width=True):
                st.session_state.show_improve_section = True
            
            if st.session_state.show_improve_section:
                st.markdown("### 🚀 Improve Model")
                st.markdown("Select the correct classification and click Improve Model:")
                
                correction_type = st.selectbox(
                    "What is the correct classification?",
                    ["Select...", "It's actually AI-Generated", "It's actually Real"],
                    key="unified_correction_select"
                )
                
                if correction_type != "Select...":
                    user_correction = "fake" if "AI-Generated" in correction_type else "real"
                    
                    if st.button("🔧 Improve Model", key="improve_model_btn", use_container_width=True, type="primary"):
                        rtl = st.session_state.real_time_learning_system
                        
                        # Check if already training
                        if rtl.is_training():
                            st.warning("🔄 Model is already being trained. Please wait for current training to complete.")
                        else:
                            with st.spinner("🧠 Starting model training..."):
                                improvement_result = rtl.improve_model_with_feedback(image, user_correction)
                                
                                if improvement_result['success']:
                                    # Store training state for monitoring
                                    st.session_state.training_active = True
                                    st.session_state.training_image = image
                                    st.session_state.training_correction = user_correction
                                    
                                    st.markdown("""
                                    <div class="message message-success">
                                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                                            <span style="font-size: 1.5rem;">✅</span>
                                            <div>
                                                <div style="font-weight: 700; margin-bottom: 0.25rem;">Model Training Started!</div>
                                                <div style="font-size: 0.9rem; opacity: 0.9;">Training is happening in the background. You can continue using the app.</div>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.rerun()

                                else:
                                    st.error(f"❌ Error: {improvement_result.get('error', 'Unknown error occurred')}")
                
                if st.button("🔄 Reset", key="reset_btn", use_container_width=True):
                    st.session_state.show_improve_section = False
                    st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Training progress monitoring
    if 'training_active' in st.session_state and st.session_state.training_active:
        rtl = st.session_state.real_time_learning_system
        
        if rtl.is_training():
            st.markdown("### 🔄 Training Progress")
            progress_placeholder = st.empty()
            
            # Continuous monitoring with auto-refresh
            import time
            
            # Get current progress
            progress = rtl.get_training_progress()
            
            if progress['status'] == 'training':
                if progress['total_epochs'] > 0:
                    epoch_progress = progress['epoch'] / progress['total_epochs']
                    # Ensure progress is within valid range [0.0, 1.0]
                    epoch_progress = max(0.0, min(1.0, epoch_progress))
                    progress_placeholder.progress(epoch_progress, text=f"Epoch {progress['epoch']}/{progress['total_epochs']} - Loss: {progress['loss']:.4f}")
                    
                    # Auto-refresh every 0.5 seconds to show real-time progress
                    time.sleep(0.5)
                    st.rerun()
                else:
                    progress_placeholder.text("🔄 Training in progress...")
                    time.sleep(0.5)
                    st.rerun()
            elif progress['status'] == 'completed':
                progress_placeholder.success("✅ Training completed!")
                
                # Force a small delay to ensure training is fully complete
                time.sleep(1)
                
                # Update prediction immediately after training
                updated_result = rtl.predict_with_learning(st.session_state.training_image)
                updated_prediction = updated_result["prediction"]
                
                # Store the updated prediction in session state for display
                st.session_state.updated_prediction = updated_prediction
                st.session_state.prediction_updated = True
                
                # Show prominent success message
                st.markdown("""
                <div class="message message-success" style="margin: 2rem 0; padding: 2rem; border-radius: 12px; background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(34, 197, 94, 0.1)); border: 2px solid #22c55e;">
                    <div style="display: flex; align-items: center; gap: 1rem; text-align: center;">
                        <span style="font-size: 3rem;">🎉</span>
                        <div>
                            <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem; color: #22c55e;">Training Completed Successfully!</div>
                            <div style="font-size: 1.1rem; opacity: 0.9;">All 20 epochs completed. Model has learned from your feedback.</div>
                            <div style="font-size: 1rem; opacity: 0.8; margin-top: 0.5rem;">The prediction result has been updated below.</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Update the prediction display with prominent styling
                updated_prediction_text = "🔴 FAKE DETECTED" if updated_prediction == "fake" else "🟢 REAL DETECTED"
                updated_prediction_class = "prediction-fake" if updated_prediction == "fake" else "prediction-real"
                
                st.markdown(f"""
                <div class="prediction-result {updated_prediction_class}" style="padding: 2rem; text-align: center; margin: 2rem 0; border: 3px solid {'#ef4444' if updated_prediction == 'fake' else '#10b981'}; border-radius: 12px; background: rgba(31, 41, 55, 0.95); box-shadow: 0 10px 25px rgba(0,0,0,0.3);">
                    <div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                        {updated_prediction_text}
                    </div>
                    <div style="font-size: 1.2rem; opacity: 0.9; margin-top: 0.5rem;">
                        ✅ Result Updated After Training
                    </div>
                    <div style="font-size: 1rem; opacity: 0.7; margin-top: 0.5rem;">
                        Model learned from your feedback
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Reset training state
                st.session_state.training_active = False
                st.session_state.show_improve_section = False
                st.rerun()
                
            elif progress['status'] == 'error':
                progress_placeholder.error("❌ Training failed!")
                st.session_state.training_active = False
        else:
            # Training completed but not detected by is_training()
            st.session_state.training_active = False

else:
    # Landing page
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h3>🚀 Get Started</h3>
        <p style="font-size: 1.125rem;">
            Upload an image above to begin AI-powered authenticity analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## ✨ Advanced Features")
    
    st.markdown("""
    <div class="features-grid">
        <div class="feature-item">
            <div class="feature-icon">🔍</div>
            <h4>Advanced AI Detection</h4>
            <p>State-of-the-art neural network with high precision detection capabilities.</p>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🤖</div>
            <h4>Real-Time Learning</h4>
            <p>Model learns immediately from your feedback with instant improvement and memory.</p>
        </div>
        <div class="feature-item">
            <div class="feature-icon">⚡</div>
            <h4>Real-time Processing</h4>
            <p>Instant predictions in seconds.</p>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🛡️</div>
            <h4>Privacy First</h4>
            <p>Secure processing with complete privacy protection.</p>
        </div>
        <div class="feature-item">
            <div class="feature-icon">📱</div>
            <h4>User-Friendly</h4>
            <p>Simple drag-and-drop interface with no technical knowledge required.</p>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🎯</div>
            <h4>High Accuracy</h4>
            <p>Trained on millions of images with advanced deep learning techniques.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    


# Footer
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 15px 0;
        text-align: center;
        background: linear-gradient(135deg, rgba(31, 41, 55, 0.95) 0%, rgba(17, 24, 39, 0.95) 100%);
        backdrop-filter: blur(12px);
        color: #f8fafc;
        font-size: 14px;
        font-weight: 500;
        border-top: 1px solid rgba(74, 85, 104, 0.2);
        z-index: 999;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .footer strong {
        color: #4a90e2;
        font-weight: 600;
    }
    
    .footer:hover {
        background: linear-gradient(135deg, rgba(31, 41, 55, 0.98) 0%, rgba(17, 24, 39, 0.98) 100%);
    }
    
    @media (max-width: 768px) {
        .footer {
            padding: 12px 0;
            font-size: 13px;
        }
    }
    </style>

    <div class="footer">
        Built with ❤️ using <strong>Streamlit</strong> and <strong>PyTorch</strong>
    </div>
""", unsafe_allow_html=True)


