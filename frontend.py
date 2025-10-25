import streamlit as st
import requests
import json
import os
import tempfile
from PIL import Image
import io

# Configure Streamlit page
st.set_page_config(
    page_title="H-MaMa Fake News Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .fake-prediction {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .real-prediction {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
    .explanation-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory and return the file path."""
    try:
        # Create temporary directory if it doesn't exist
        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Clean filename to avoid issues
        safe_filename = uploaded_file.name.replace(" ", "_").replace("\\", "_").replace("/", "_")
        
        # Save file with normalized path
        file_path = os.path.join(temp_dir, safe_filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Return normalized path (forward slashes for API compatibility)
        normalized_path = file_path.replace("\\", "/")
        return normalized_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def submit_feedback(text, predicted_label, predicted_confidence, actual_label, feedback_type):
    """Submit feedback to improve the model"""
    try:
        payload = {
            "text": text,
            "predicted_label": predicted_label,
            "predicted_confidence": predicted_confidence,
            "actual_label": actual_label,
            "user_id": "streamlit_user"
        }
        
        response = requests.post(
            "http://localhost:8000/feedback",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return True
        else:
            st.error(f"Error submitting feedback: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error submitting feedback: {e}")
        return False

def call_explain_api(text, image_path=None):
    """Call the /explain endpoint of the FastAPI backend."""
    try:
        # Prepare the request payload
        payload = {"text": text}
        
        # Only add image_path if we have a valid image file
        if image_path:
            # Convert to absolute path and normalize
            abs_path = os.path.abspath(image_path)
            normalized_path = abs_path.replace("\\", "/")
            
            if os.path.exists(normalized_path):
                payload["image_path"] = normalized_path
                print(f"DEBUG: Using image path: {normalized_path}")  # Debug info
            else:
                print(f"DEBUG: Image not found: {normalized_path}")  # Debug info
                # Don't include image_path if file doesn't exist
        # If no image provided, don't include image_path in payload (API will handle text-only)
        
        print(f"DEBUG: API payload: {payload}")  # Debug info
        
        # Make the API call
        response = requests.post(
            "http://localhost:8000/explain",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to the API. Please make sure the FastAPI server is running on http://localhost:8000"
    except requests.exceptions.Timeout:
        return None, "Request timed out. The API is taking too long to respond."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def display_prediction_result(result):
    """Display the prediction results in a user-friendly format."""
    if not result:
        return
    
    # Extract key information
    label = result.get('label', 'UNKNOWN')
    confidence = result.get('confidence', 0)
    human_explanation = result.get('human_explanation', 'No explanation available.')
    probabilities = result.get('probabilities', {})
    contradiction_score = result.get('contradiction_score', 0)
    aigc_flags = result.get('aigc_flags', {})
    
    # Display prediction with color coding and confidence warnings
    confidence_warning = ""
    if confidence < 60:
        confidence_warning = "⚠️ Low confidence - prediction may be unreliable"
    elif confidence < 80:
        confidence_warning = "⚠️ Medium confidence - consider additional verification"
    
    if label == 'FAKE':
        st.markdown(f"""
        <div class="prediction-box fake-prediction">
            <h2>🔴 Prediction: {label}</h2>
            <h3>Confidence: {confidence}%</h3>
            {f'<p style="color: #ff6b6b; font-weight: bold;">{confidence_warning}</p>' if confidence_warning else ''}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box real-prediction">
            <h2>🟢 Prediction: {label}</h2>
            <h3>Confidence: {confidence}%</h3>
            {f'<p style="color: #ff6b6b; font-weight: bold;">{confidence_warning}</p>' if confidence_warning else ''}
        </div>
        """, unsafe_allow_html=True)
    
    # Display human-readable explanation
    st.markdown("### 📝 Explanation")
    st.markdown(f"""
    <div class="explanation-box">
        <p><strong>{human_explanation}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display confidence explanation if available
    confidence_explanation = result.get('confidence_explanation', '')
    if confidence_explanation:
        st.markdown("### 🔍 Confidence Analysis")
        st.markdown(f"""
        <div class="explanation-box">
            <p><strong>{confidence_explanation}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display uncertainty indicators
    uncertainty_indicators = result.get('uncertainty_indicators', [])
    if uncertainty_indicators:
        st.markdown("### ⚠️ Uncertainty Indicators")
        for indicator in uncertainty_indicators:
            st.markdown(f"• {indicator}")
    
    # Display model health if available
    model_health = result.get('model_health', {})
    if model_health:
        st.markdown("### 📊 Model Health")
        health_score = model_health.get('health_score', 0)
        reliability = model_health.get('reliability', 0)
        
        health_color = "green" if health_score > 0.7 else "orange" if health_score > 0.5 else "red"
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid {health_color};">
            <p><strong>Health Score:</strong> {health_score:.1%}</p>
            <p><strong>Reliability:</strong> {reliability:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display additional details in expandable section
    with st.expander("🔍 Technical Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Probabilities:**")
            if probabilities:
                st.write(f"- Fake: {probabilities.get('fake', 0):.1f}%")
                st.write(f"- Real: {probabilities.get('real', 0):.1f}%")
            
            st.write("**Text-Image Alignment:**")
            st.write(f"- Contradiction Score: {contradiction_score:.3f}")
            if contradiction_score > 0.7:
                st.write("  (High contradiction)")
            elif contradiction_score < 0.3:
                st.write("  (Good alignment)")
            else:
                st.write("  (Moderate alignment)")
        
        with col2:
            st.write("**AI-Generated Content Detection:**")
            if aigc_flags:
                text_flag = aigc_flags.get('text', 0)
                img_flag = aigc_flags.get('image', 0)
                st.write(f"- Text: {text_flag:.3f}")
                st.write(f"- Image: {img_flag:.3f}")
                
                if text_flag > 0.5 or img_flag > 0.5:
                    st.warning("⚠️ High AI-generated content detected")
        
        # Add feedback section
        st.markdown("---")
        st.markdown("### 💬 Help Improve the Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ This prediction is correct", key="correct_feedback"):
                submit_feedback(text_input, label, confidence/100, label, "correct")
                st.success("Thank you for your feedback!")
        
        with col2:
            if st.button("❌ This prediction is wrong", key="incorrect_feedback"):
                # Show feedback form
                st.session_state.show_feedback_form = True
        
        if st.session_state.get('show_feedback_form', False):
            st.markdown("**What is the correct label?**")
            correct_label = st.selectbox("Select correct label:", ["REAL", "FAKE"], key="correct_label")
            
            if st.button("Submit Feedback", key="submit_feedback"):
                submit_feedback(text_input, label, confidence/100, correct_label, "incorrect")
                st.success("Thank you for your feedback! The model will learn from this.")
                st.session_state.show_feedback_form = False

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 H-MaMa Fake News Detection</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Enter Text**: Type or paste the news text you want to analyze
        2. **Upload Image** (Optional): Upload an image related to the news
        3. **Click Check**: Analyze the content for fake news detection
        4. **View Results**: See the prediction and explanation
        """)
        
        st.header("ℹ️ About")
        st.markdown("""
        This tool uses the H-MaMa multimodal fake news detection system to analyze text and images for authenticity.
        
        **Features:**
        - Text analysis with token attribution
        - Image-text alignment checking
        - AI-generated content detection
        - Human-readable explanations
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Input Content")
        
        # Text input
        text_input = st.text_area(
            "Enter the news text to analyze:",
            placeholder="Paste your news article or text here...",
            height=150,
            help="Enter the text content you want to analyze for fake news detection."
        )
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image (optional):",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Upload an image related to the news text for multimodal analysis."
        )
        
        # Display uploaded image preview
        image_path = None
        if uploaded_file is not None:
            # Save the uploaded file
            image_path = save_uploaded_file(uploaded_file)
            
            if image_path:
                st.success(f"✅ Image uploaded: {uploaded_file.name}")
                
                # Display image preview
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image Preview", use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")
            else:
                st.error("❌ Failed to save uploaded image")
        
        # Check button
        check_button = st.button(
            "🔍 Check for Fake News",
            type="primary",
            use_container_width=True,
            disabled=not text_input.strip()
        )
    
    with col2:
        st.header("📊 Results")
        
        if check_button:
            if not text_input.strip():
                st.error("Please enter some text to analyze.")
            else:
                # Show loading spinner
                with st.spinner("Analyzing content..."):
                    # Call the API
                    result, error = call_explain_api(text_input, image_path)
                
                if error:
                    st.markdown(f"""
                    <div class="error-box">
                        <strong>❌ Error:</strong> {error}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Display results
                    display_prediction_result(result)
                    
                    # Show success message
                    st.markdown("""
                    <div class="success-box">
                        <strong>✅ Analysis Complete!</strong> The results are displayed above.
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👆 Enter text and click 'Check for Fake News' to get started.")
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <h4>⚠️ Important Disclaimer</h4>
        <p><strong>This is a research prototype.</strong> The model's predictions may not always be accurate and should not be the sole basis for determining news authenticity. Always verify information through multiple reliable sources.</p>
        <p><strong>Model limitations:</strong> The system may have biases based on its training data and may not perform well on all types of content. Use with caution for critical decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>H-MaMa Fake News Detection System | Powered by Multimodal AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
