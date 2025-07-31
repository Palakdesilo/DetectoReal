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
from retrain_model import retrain_model_with_feedback
from predict import load_prediction_model, predict_image_from_pil, get_detailed_analysis
from automatic_learning_system import AutomaticLearningSystem

# Initialize automatic learning system (hidden from UI)
if 'automatic_learning_system' not in st.session_state:
    st.session_state.automatic_learning_system = AutomaticLearningSystem(
        feedback_threshold=5,  # Retrain after 5 feedback items
        auto_retrain=True,
        retrain_interval_hours=1  # Allow retraining every hour
    )

# Page configuration
st.set_page_config(
    page_title="DetectoReal - AI Image Authenticity Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Force dark theme
st.markdown("""
<style>
    /* Override Streamlit's theme detection */
    [data-testid="stAppViewContainer"] {
        background: var(--bg-gradient-primary) !important;
    }
    
    /* Force dark theme on all Streamlit components */
    .stApp {
        background: var(--bg-gradient-primary) !important;
    }
    
    /* Override any light theme elements */
    .stMarkdown, .stText, .stButton, .stSelectbox, .stFileUploader, .stSpinner {
        color: var(--text-primary) !important;
        background: transparent !important;
    }
    
    /* Ensure upload area uses our dark theme */
    .stFileUploader > div {
        color: var(--text-primary) !important;
    }
    
    /* Header styling - topmost layer with maximum specificity */
    div[data-testid="stMarkdown"] h1,
    .stMarkdown h1,
    h1 {
        color: #ffffff !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
        text-align: center !important;
        z-index: 9999 !important;
        position: relative !important;
        opacity: 1 !important;
    }
    
    /* Header paragraph styling - topmost layer */
    div[data-testid="stMarkdown"] p,
    .stMarkdown p,
    p {
        color: #f8fafc !important;
        font-size: 1.125rem !important;
        font-weight: 500 !important;
        text-align: center !important;
        z-index: 9999 !important;
        position: relative !important;
        opacity: 1 !important;
    }
    
    /* Force header container to stay on top */
    .stMarkdown > div:first-child {
        z-index: 9999 !important;
        position: relative !important;
        background: transparent !important;
    }
    
    /* Specific header classes with maximum specificity */
    .header-container {
        z-index: 9999 !important;
        position: relative !important;
        background: transparent !important;
        opacity: 1 !important;
    }
    
    .main-title {
        color: #ffffff !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
        text-align: center !important;
        z-index: 9999 !important;
        position: relative !important;
        opacity: 1 !important;
    }
    
    .main-description {
        color: #f8fafc !important;
        font-size: 1.125rem !important;
        font-weight: 500 !important;
        text-align: center !important;
        z-index: 9999 !important;
        position: relative !important;
        opacity: 1 !important;
    }
    
    /* Header banner styling */
    .header-banner {
        padding: 1.5rem 2rem !important;
        border-radius: 12px !important;
        margin-bottom: 2rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 1rem !important;
        z-index: 9999 !important;
        position: relative !important;
        opacity: 1 !important;
    }
    
    .header-banner h1 {
        color: #ffffff !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #ffffff !important;
        -webkit-text-stroke: 0 !important;
    }
    
    /* Additional white text enforcement */
    .header-banner h1,
    .header-banner .main-title,
    div[data-testid="stMarkdown"] .header-banner h1 {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    .header-banner span {
        font-size: 2.5rem !important;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3)) !important;
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Unified CSS with dark theme
st.markdown("""
<style>
    /* CSS Variables for Neutral Color Palette */
    :root {
        /* Neutral Color Palette */
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
        --scrollbar-width: 10px;
        font-family: "Source Sans", sans-serif;
        font-weight: 400;
        line-height: 1.6;
        text-size-adjust: 100%;
        -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
        -webkit-font-smoothing: auto;
        box-sizing: border-box;
        scrollbar-width: thin;
        scrollbar-color: transparent transparent;
        transition: all 0.3s ease;
        position: absolute;
        color: var(--text-primary) !important;
        inset: 0px;
        color-scheme: dark !important;
        overflow: hidden;
        background: var(--bg-gradient-primary) !important;
    }

    /* Force dark theme and override Streamlit defaults */
    .stApp {
        background: var(--bg-gradient-primary) !important;
        color: var(--text-primary) !important;
    }

    .main .block-container {
        background: transparent !important;
        color: var(--text-primary) !important;
    }

    /* Override Streamlit's default light theme */
    [data-testid="stAppViewContainer"] {
        background: var(--bg-gradient-primary) !important;
    }

    /* Ensure all text uses our color scheme */
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
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
        animation: slideIn 0.3s ease;
    }

    .message-success {
        background: var(--success);
        color: var(--off-white);
    }

    .message-error {
        background: var(--error);
        color: var(--off-white);
    }

    .message-warning {
        background: var(--warning);
        color: var(--off-white);
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

    /* Confidence meter */
    .confidence-meter {
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .confidence-bar {
        height: 12px;
        background: var(--gray-300);
        border-radius: 6px;
        overflow: hidden;
        margin: 0.5rem 0;
    }

    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--error), var(--warning), var(--success));
        border-radius: 6px;
        transition: width 1s ease;
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
<div class="header-banner" style="
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    z-index: 9999;
    position: relative;
    opacity: 1 !important;
">
    <span style="font-size: 2.5rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));">🕵️‍♂️</span>
    <h1 style="
        color: #ffffff !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        opacity: 1 !important;
        -webkit-text-fill-color: #ffffff !important;
        -webkit-text-stroke: 0 !important;
    ">DetectoReal</h1>
</div>
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="color: white; font-size: 1.125rem; margin: 0;">An AI-powered tool to detect fake vs real images.</p>
</div>
""", unsafe_allow_html=True)

# Upload section
st.markdown("""
<div>
    <h3 style="margin-bottom: 1rem;">📁 Upload Your Image</h3>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image file", 
    type=['png', 'jpg', 'jpeg'],
    label_visibility="collapsed"
)

# Analysis section
if uploaded_file is not None:
    with st.spinner("🔍 Analyzing image..."):
        time.sleep(0.5)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📸 Image Analysis")
        
        # Display image
        image = Image.open(uploaded_file)
        resized_image = image.resize((150, 150))
        st.image(resized_image, caption="Uploaded Image")
        
        # Get prediction
        als = st.session_state.automatic_learning_system
        result = als.predict_and_collect_feedback(image)
        
        prediction = result["prediction"]
        confidence = result["confidence"]
        
        # Prediction result and confidence on same row
        confidence_percentage = confidence * 100
        prediction_text = "🔴 FAKE DETECTED" if prediction == "fake" else "🟢 REAL DETECTED"
        prediction_class = "prediction-fake" if prediction == "fake" else "prediction-real"
        
        st.markdown(f"""
        <div class="prediction-result {prediction_class}" style="display: flex; justify-content: space-between; align-items: center; padding: 1.5rem;">
            <div style="font-size: 1.25rem; font-weight: 700;">
                {prediction_text}
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.875rem; margin-bottom: 0.25rem; opacity: 0.8;">Confidence</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{confidence_percentage:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🤔 Was this prediction correct?")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("✅ Correct", key="correct_btn", use_container_width=True):
                als = st.session_state.automatic_learning_system
                feedback_result = als.predict_and_collect_feedback(
                    image=image,
                    user_feedback="Correct prediction",
                    user_correction=prediction
                )
                
                if feedback_result['feedback_saved']:
                    st.markdown("""
                    <div class="message message-success">
                        ✅ Thank you for the confirmation! Feedback saved successfully.
                    </div>
                    """, unsafe_allow_html=True)
        
        with col_b:
            if st.button("❌ Incorrect", key="incorrect_btn", use_container_width=True):
                correction_type = st.selectbox(
                    "What is the correct classification?",
                    ["Select...", "It's actually AI-Generated", "It's actually Real"],
                    key="correction_select"
                )
                
                if correction_type != "Select...":
                    user_correction = "fake" if "AI-Generated" in correction_type else "real"
                    als = st.session_state.automatic_learning_system
                    feedback_result = als.predict_and_collect_feedback(
                        image=image,
                        user_feedback=f"Correction: {correction_type}",
                        user_correction=user_correction
                    )
                    
                    if feedback_result['feedback_saved']:
                        st.markdown("""
                        <div class="message message-success">
                            ✅ Thank you for the correction! Feedback saved successfully.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if feedback_result['should_retrain']:
                            st.markdown("""
                            <div class="message message-warning">
                                🔄 Ready for continuous model improvement!
                            </div>
                            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Improve Model", key="retrain_btn", use_container_width=True):
            with st.spinner("🔄 Continuously improving model..."):
                als = st.session_state.automatic_learning_system
                success = als.trigger_retraining(method="rlhf")
                if success:
                    st.markdown("""
                    <div class="message message-success">
                        🎉 Model continuously improved! The existing model has been enhanced based on your feedback.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="message message-error">
                        ❌ Continuous improvement failed. Please try again or check system status.
                    </div>
                    """, unsafe_allow_html=True)

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
            <h4>Continuous Learning</h4>
            <p>Existing model continuously improves from user feedback with RLHF technology.</p>
        </div>
        <div class="feature-item">
            <div class="feature-icon">⚡</div>
            <h4>Real-time Processing</h4>
            <p>Instant predictions with confidence scores in seconds.</p>
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
<div style="text-align: center; margin-top: 3rem; padding: 1.5rem; border-top: 1px solid #4a5568;">
    <p style="font-weight: 600; margin-bottom: 0.5rem;">🔍 DetectoReal - AI Image Authenticity Detection</p>
    <p style="font-size: 0.875rem; color: #a0aec0;">Built with Streamlit and PyTorch | Advanced ML Technology</p>
</div>
""", unsafe_allow_html=True)
