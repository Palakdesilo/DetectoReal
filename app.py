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
from real_time_learning_enhanced_simple import RLHFImageClassifier

# Initialize RLHF image classifier
if 'real_time_learning_system' not in st.session_state:
    st.session_state.real_time_learning_system = RLHFImageClassifier(
        learning_rate=1e-4,
        memory_size=1000
    )
    print(f"🚀 Initialized RLHF system with model ID: {id(st.session_state.real_time_learning_system.model)}")
else:
    print(f"📂 Using existing RLHF system with model ID: {id(st.session_state.real_time_learning_system.model)}")

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
    .stApp {
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
    
    /* Override Streamlit-generated element container classes */
    .stElementContainer,
    .element-container,
    .st-emotion-cache-v3w3zg,
    .eertqu00 {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
         /* Header styling - topmost layer with maximum specificity */
     .stMarkdown h1,
     h1 {
         color: #ffffff !important;
         font-size: 2.5rem !important;
         font-weight: 700 !important;
         margin-bottom: 1rem !important;
         text-align: center !important;
         z-index: 9999 !important;
         position: relative !important;
         opacity: 1 !important;
         -webkit-text-fill-color: #ffffff !important;
         -webkit-text-stroke: 0 !important;
         text-shadow: none !important;
     }
    
         /* Header paragraph styling - topmost layer */
     .stMarkdown p,
     p {
         color: #ffffff !important;
         font-size: 1.125rem !important;
         font-weight: 500 !important;
         text-align: center !important;
         z-index: 9999 !important;
         position: relative !important;
         opacity: 1 !important;
         -webkit-text-fill-color: #ffffff !important;
         -webkit-text-stroke: 0 !important;
         text-shadow: none !important;
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
         text-align: center !important;
         z-index: 9999 !important;
         position: relative !important;
         opacity: 1 !important;
         -webkit-text-fill-color: #ffffff !important;
         -webkit-text-stroke: 0 !important;
         text-shadow: none !important;
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
        padding: 1.5rem 2rem 6rem 2rem; !important;
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
         opacity: 1 !important;
     }
</style>
""", unsafe_allow_html=True)

# === CUSTOM CSS FOR FILE UPLOADER ===
st.markdown("""
<style>
    .stFileUploader {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
    }

    .stFileUploader:hover {
        border-color: #764ba2;
        background: rgba(118, 75, 162, 0.05);
        transform: translateY(-2px);
    }

    .upload-note {
        color: #555;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 0.5rem;
    }

    .upload-title {
        text-align: center;
        color: #333;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Background color override for entire app */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
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
    .stApp {
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
        
        # Get prediction with real-time learning
        rtl = st.session_state.real_time_learning_system
        print(f"🔍 Making prediction with model ID: {id(rtl.model)}")
        result = rtl.predict_with_learning(image)
        
        prediction = result["prediction"]
        
        # Prediction result
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
        
        # Show prediction source and match type

    
    with col2:
        st.markdown("### 🤔 Was this prediction correct?")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
             if st.button("✅ Correct", key="correct_btn", use_container_width=True):
                 # Show success message immediately
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
                 
                 # Process feedback in background (non-blocking)
                 rtl = st.session_state.real_time_learning_system
                 feedback_result = rtl.predict_with_learning(
                     image=image,
                     user_feedback="Correct prediction",
                     user_correction=prediction
                 )
        
        with col_b:
            # Use session state to track if user clicked incorrect
            if 'show_improve_section' not in st.session_state:
                st.session_state.show_improve_section = False
            
            if st.button("❌ Incorrect", key="incorrect_btn", use_container_width=True):
                st.session_state.show_improve_section = True
            
            # Show improve model section if user clicked incorrect
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
                        print(f"🧠 Improving model with ID: {id(rtl.model)}")
                        
                        with st.spinner("🧠 Teaching the model..."):
                            # Use the enhanced improve_model_with_feedback method
                            improvement_result = rtl.improve_model_with_feedback(image, user_correction)
                            
                            if improvement_result['success']:
                                st.markdown("""
                                <div class="message message-success">
                                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                                        <span style="font-size: 1.5rem;">✅</span>
                                        <div>
                                            <div style="font-weight: 700; margin-bottom: 0.25rem;">Model Improved Successfully!</div>
                                            <div style="font-size: 0.9rem; opacity: 0.9;">Learned everything about this image and similar images</div>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show learning verification
                                if improvement_result.get('learning_verified', False):
                                    st.markdown("""
                                    <div class="message message-success">
                                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                                            <span style="font-size: 1.5rem;">🎉</span>
                                            <div>
                                                <div style="font-weight: 700; margin-bottom: 0.25rem;">Perfect Learning!</div>
                                                <div style="font-size: 0.9rem; opacity: 0.9;">Model now correctly predicts this image</div>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown("""
                                    <div class="message message-warning">
                                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                                            <span style="font-size: 1.5rem;">🔄</span>
                                            <div>
                                                <div style="font-weight: 700; margin-bottom: 0.25rem;">Learning in Progress</div>
                                                <div style="font-size: 0.9rem; opacity: 0.9;">Model is improving with each feedback</div>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Test the same image to verify learning
                                st.markdown("### 🧪 Learning Verification")
                                st.markdown("Testing the same image to verify learning:")
                                
                                test_result = rtl.test_same_image_improvement(image, user_correction)
                                
                                if test_result['correct_prediction']:
                                    st.markdown("""
                                    <div class="message message-success">
                                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                                            <span style="font-size: 1.5rem;">✅</span>
                                            <div>
                                                <div style="font-weight: 700; margin-bottom: 0.25rem;">Learning Verified!</div>
                                                <div style="font-size: 0.9rem; opacity: 0.9;">Same image now predicts correctly: """ + test_result['prediction'] + """</div>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown("""
                                    <div class="message message-warning">
                                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                                            <span style="font-size: 1.5rem;">⚠️</span>
                                            <div>
                                                <div style="font-weight: 700; margin-bottom: 0.25rem;">Learning Needs More Work</div>
                                                <div style="font-size: 0.9rem; opacity: 0.9;">Current prediction: """ + test_result['prediction'] + """ (Expected: """ + user_correction + """)</div>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Show learning details
                                st.markdown(f"""
                                <div style="background: rgba(31, 41, 55, 0.8); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                                    <h4>📊 Learning Details</h4>
                                    <ul style="margin: 0; padding-left: 1.5rem;">
                                        <li><strong>Source:</strong> {test_result['source']}</li>
                                        <li><strong>Match Type:</strong> {test_result['match_type']}</li>
                                        <li><strong>Memory Used:</strong> {'Yes' if test_result['memory_used'] else 'No'}</li>
                                        <li><strong>Augmentations Applied:</strong> {improvement_result.get('augmentations_applied', 0)}</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Reset the section
                                st.session_state.show_improve_section = False
                                st.rerun()
                            else:
                                st.markdown(f"""
                                <div class="message message-error">
                                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                                        <span style="font-size: 1.5rem;">❌</span>
                                        <div>
                                            <div style="font-weight: 700; margin-bottom: 0.25rem;">Learning Error</div>
                                            <div style="font-size: 0.9rem; opacity: 0.9;">{improvement_result.get('error', 'Unknown error')}</div>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Add a reset button
                if st.button("🔄 Reset", key="reset_btn", use_container_width=True):
                    st.session_state.show_improve_section = False
                    st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)

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
    
    # Add learning persistence note
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(74, 144, 226, 0.1), rgba(56, 178, 172, 0.1)); 
                border: 1px solid rgba(74, 144, 226, 0.3); 
                border-radius: 12px; 
                padding: 1.5rem; 
                margin: 2rem 0;">
        <h4 style="color: #4a90e2; margin-bottom: 1rem;">💡 Learning Persistence</h4>
        <p style="color: #e2e8f0; margin-bottom: 0.5rem;">
            <strong>Your learning is automatically saved!</strong> When you provide feedback and correct the model, 
            it learns and remembers this information. The learning persists across:
        </p>
        <ul style="color: #e2e8f0; margin: 0; padding-left: 1.5rem;">
            <li><strong>Page refreshes</strong> - Your learning won't be lost when you refresh the page</li>
            <li><strong>Browser sessions</strong> - Learning is saved to files for long-term persistence</li>
            <li><strong>Multiple uploads</strong> - The model remembers previous corrections</li>
        </ul>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 1rem; margin-bottom: 0;">
            💾 Learning data is stored locally and in session state for maximum reliability.
        </p>
    </div>
    """, unsafe_allow_html=True)



# Footer with matching theme
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


