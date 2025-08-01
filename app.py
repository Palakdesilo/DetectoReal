import streamlit as st
from PIL import Image, UnidentifiedImageError
import sys
import os
import time
import base64
from io import BytesIO
import json
import datetime
import glob

# === PAGE SETTINGS ===
st.set_page_config(
    page_title="DetectoReal - AI Image Authenticity Detector",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === CUSTOM CSS FOR MODERN UI ===
st.markdown("""
<style>
    /* Modern color scheme and fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Upload area styling */
    .upload-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
    }
    
    .upload-title {
        text-align: center;
        color: #333;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* File uploader styling */
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
    
    /* Image display styling */
    .image-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .image-container img {
        border-radius: 10px;
        max-width: 100%;
        height: auto;
    }
    
    /* Results styling */
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    }
    
    .result-fake {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
    }
    
    .result-real {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
    }
    
    .result-uncertain {
        background: linear-gradient(135deg, #ffd43b, #fcc419);
        color: #333;
    }
    
    .result-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .result-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    .confidence-bar {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        height: 8px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease;
    }
    
    .confidence-fill-fake {
        background: linear-gradient(90deg, #ff6b6b, #ee5a52);
    }
    
    .confidence-fill-real {
        background: linear-gradient(90deg, #51cf66, #40c057);
    }
    
    .confidence-fill-uncertain {
        background: linear-gradient(90deg, #ffd43b, #fcc419);
    }
    
    /* Loading animation */
    .loading-container {
        text-align: center;
        padding: 2rem;
    }
    
    .spinner {
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top: 4px solid #667eea;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 0 auto 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    /* Info cards */
    .info-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .info-card h3 {
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .info-card p {
        color: #666;
        line-height: 1.6;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .result-title {
            font-size: 2rem;
        }
    }
    
    /* Hover effects for interactive elements */
    .upload-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    }
    
    /* Smooth transitions for all interactive elements */
    * {
        transition: all 0.3s ease;
    }
    
    /* Feedback styling */
    .feedback-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .feedback-title {
        color: #333;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# === FEEDBACK SYSTEM ===
def save_feedback(image_data, model_prediction, user_correction, confidence):
    """Save user feedback for model training"""
    feedback_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_prediction": model_prediction,
        "user_correction": user_correction,
        "confidence": confidence,
        "image_data": image_data
    }
    
    # Create feedback directory if it doesn't exist
    os.makedirs("feedback_data", exist_ok=True)
    
    # Save feedback to file
    feedback_file = f"feedback_data/feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(feedback_file, 'w') as f:
        json.dump(feedback_data, f, indent=2)
    
    return feedback_file

def encode_image_to_base64(image):
    """Convert PIL image to base64 string"""
    import base64
    from io import BytesIO
    
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

# === MODEL LOADING ===
@st.cache_resource
def load_prediction_model():
    """Load the prediction model with error handling"""
    try:
        from predict import predict_image_from_pil, get_detailed_analysis
        return predict_image_from_pil, get_detailed_analysis
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.error("Please ensure the model.pth file is included in your deployment.")
        return None, None



# Load the prediction functions
predict_function, detailed_analysis_function = load_prediction_model()

if predict_function is None or detailed_analysis_function is None:
    st.error("""
    ## 🚨 Model Loading Error
    
    The application failed to load the AI model. This could be due to:
    
    1. **Missing model file**: Ensure `model.pth` is included in your repository
    2. **File path issues**: The model file should be in the root directory
    3. **Deployment issues**: Check your Streamlit Cloud deployment settings
    
    ### To fix this:
    - Make sure `model.pth` is committed to your repository
    - Verify the file is not in `.gitignore`
    - Check that the file size is reasonable for deployment
    """)
    st.stop()

# Clear the loading message and show the main interface
st.empty()

# === HEADER SECTION ===
st.markdown("""
<div style="text-align: center; padding: 3rem 0;">
    <h1 style="color: white; font-size: 3rem; font-weight: 700; margin-bottom: 1rem;">
        🕵️‍♂️ DetectoReal
    </h1>
    <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-bottom: 2rem;">
        An AI-powered tool to detect fake vs real images.
    </p>
</div>
""", unsafe_allow_html=True)

# === UPLOAD SECTION ===

uploaded_file = st.file_uploader(
    "Choose an image file (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"],
    help="Upload any image to analyze if it's real or AI-generated"
)

# === PROCESSING SECTION ===
if uploaded_file is not None:
    try:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Create two-column layout
        col1, col2 = st.columns([1, 1], gap="small")
        
        # Add custom CSS to reduce gap further
        st.markdown("""
        <style>
        [data-testid="column"] {
            margin: 0 0.5rem !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with col1:
            # Display uploaded image with title above
            st.markdown("""
            <div style="text-align: left; margin-bottom: 0.2rem;">
                <h3 style="color: white;">📸 Uploaded Image</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, caption="", width=250, use_container_width=False)
            
            # Show loading animation
            with st.spinner("🔍 Analyzing image..."):
                time.sleep(1)  # Simulate processing time for better UX
                
                try:
                    # Get detailed analysis
                    analysis = detailed_analysis_function(image)
                    
                    # Create result display
                    if analysis["prediction"].lower() == "fake":
                        icon = "🔴"
                        title = "FAKE DETECTED"
                        result_color = "#ff6b6b"
                        message_color = "#ff4444"
                    elif analysis["prediction"].lower() == "real":
                        icon = "🟢"
                        title = "REAL DETECTED"
                        result_color = "#51cf66"
                        message_color = "#44ff44"
                    else:
                        icon = "⚠️"
                        title = "UNCERTAIN"
                        result_color = "#ffd43b"
                        message_color = "#ffff44"
                    
                    # Display simple result below the image
                    st.write(f"{icon} {title}")
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
        
        with col2:
            # Feedback section on the right side
            st.markdown("""
            <div style="margin-top: 2rem;">
                <h4 style="color: white; margin-bottom: 1rem;">🤔 Was this prediction correct?</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Feedback buttons in a clean layout
            col_feedback1, col_feedback2, col_feedback3 = st.columns(3, gap="small")
            
            with col_feedback1:
                if st.button("✅ Correct", type="primary", use_container_width=True):
                    st.success("Thank you for confirming! This helps improve our model.")
            
            with col_feedback2:
                if st.button("❌ It's AI-Generated", use_container_width=True):
                    # Save feedback for incorrect prediction
                    image_base64 = encode_image_to_base64(image)
                    feedback_file = save_feedback(
                        image_base64, 
                        analysis["prediction"], 
                        "fake", 
                        analysis["confidence"]
                    )
                    st.error("Thank you for the correction! Feedback saved.")
            
            with col_feedback3:
                if st.button("❌ It's Real", use_container_width=True):
                    # Save feedback for incorrect prediction
                    image_base64 = encode_image_to_base64(image)
                    feedback_file = save_feedback(
                        image_base64, 
                        analysis["prediction"], 
                        "real", 
                        analysis["confidence"]
                    )
                    st.error("Thank you for the correction! Feedback saved.")
            
            # Retraining option
            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Retrain Model with Feedback", use_container_width=True):
                with st.spinner("🔄 Retraining model with feedback data..."):
                    try:
                        # Import and run retraining
                        from retrain_model import retrain_model_with_feedback, analyze_feedback
                        
                        # First analyze feedback
                        feedback_data = []
                        feedback_files = glob.glob("feedback_data/feedback_*.json")
                        for file_path in feedback_files:
                            with open(file_path, 'r') as f:
                                feedback_data.append(json.load(f))
                        
                        if feedback_data:
                            st.success(f"📊 Found {len(feedback_data)} feedback samples")
                            
                            # Run retraining
                            success = retrain_model_with_feedback()
                            if success:
                                st.success("🎉 Model retraining completed successfully!")
                                st.info("The model has been updated with your feedback. New predictions will be more accurate.")
                            else:
                                st.error("❌ Model retraining failed. Please check the console for details.")
                        else:
                            st.warning("⚠️ No feedback data found. Please provide some corrections first.")
                            
                    except Exception as e:
                        st.error(f"❌ Error during retraining: {str(e)}")
                    
    except UnidentifiedImageError:
        st.error("❌ Could not identify the image. Please upload a valid image file.")
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

# === DEMO SECTION ===
if uploaded_file is None:
    # Create a more interactive demo section
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin: 0.5rem;">
            <h4 style="color: white; margin-bottom: 0.5rem;">🔍 How it Works</h4>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                Our AI analyzes subtle patterns in images to distinguish between real photos and AI-generated content
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin: 0.5rem;">
            <h4 style="color: white; margin-bottom: 0.5rem;">📊 High Accuracy</h4>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                Trained on thousands of real and AI-generated images for reliable detection
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin: 0.5rem;">
            <h4 style="color: white; margin-bottom: 0.5rem;">⚡ Fast Analysis</h4>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                Get results in seconds with detailed confidence scores and analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add some interactive elements
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin-top: 2rem;">
        <h3 style="color: white; margin-bottom: 1rem;">🎯 Ready to Detect</h3>
        <p style="color: rgba(255,255,255,0.8); margin-bottom: 1rem;">
            Upload an image above to start the analysis. The AI will examine the image and determine if it's real or AI-generated.
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
            <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 10px; font-size: 0.9rem;">
                🔴 AI-Generated
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 10px; font-size: 0.9rem;">
                🟢 Real Photos
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# === FOOTER ===
st.markdown("""
<div style="text-align: center; padding: 2rem; margin-top: 3rem;">
    <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
        Built with ❤️ using Streamlit and PyTorch
    </p>
</div>
""", unsafe_allow_html=True)
