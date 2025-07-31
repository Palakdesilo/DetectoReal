import streamlit as st # type: ignore
from PIL import Image, UnidentifiedImageError # type: ignore
import sys
import os

# === PAGE SETTINGS ===
st.set_page_config(page_title="DetectoReal", layout="centered")

# === CUSTOM CSS ===
st.markdown("""
<style>
.stFileUploader button {
    background: #B0E0E6 !important;
    color: black !important;
    border: 1px solid #a0cfd0 !important;
    border-radius: 5px !important;
}
.stFileUploader button:hover {
    background: #a8dce4 !important;
    border: 1px solid #89c4cc !important;
}
.stFileUploader div[data-testid="stFileUploaderClearButton"] button {
    background: transparent !important;
    color: black !important;
    border: none !important;
}

/* Make uploaded images smaller and more compact */
.stImage {
    max-width: 400px !important;
    margin: 0 auto !important;
}

/* Compact layout for results */
.result-container {
    margin-top: 10px !important;
    padding: 10px !important;
    border-radius: 8px !important;
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

# === MAIN TITLE ===
st.title("🕵️‍♂️ DetectoReal")

# === MODEL LOADING ===
@st.cache_resource
def load_prediction_model():
    """Load the prediction model with error handling"""
    try:
        from predict import predict_image_from_pil
        return predict_image_from_pil
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.error("Please ensure the model.pth file is included in your deployment.")
        return None

# Load the prediction function
predict_function = load_prediction_model()

if predict_function is None:
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

# === UPLOAD SECTION ===
uploaded_file = st.file_uploader("📁 Browse a real or AI-generated image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load image directly from uploaded file without saving to disk
        image = Image.open(uploaded_file).convert("RGB")
        
        # Create a compact layout with columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display image in a smaller, centered format
            st.image(image, caption="🖼 Uploaded Image", width=300)
        
        with col2:
            # Run prediction using the image object directly
            try:
                prediction = predict_function(image)
                
                # Show colored result in a compact format
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                if prediction.lower() == "fake":
                    st.markdown(f"<h2 style='color:red; margin: 0;'>🔴 FAKE</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:red; font-size: 16px; margin: 5px 0;'>AI-Generated Image Detected</p>", unsafe_allow_html=True)
                elif prediction.lower() == "real":
                    st.markdown(f"<h2 style='color:green; margin: 0;'>🟢 REAL</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:green; font-size: 16px; margin: 5px 0;'>Authentic Image Detected</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='color:orange; margin: 0;'>⚠️ UNCERTAIN</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:orange; font-size: 16px; margin: 5px 0;'>Unable to determine</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")

    except UnidentifiedImageError:
        st.error("❌ Could not identify the image. Please upload a valid image.")
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
