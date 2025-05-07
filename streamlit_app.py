"""
Advanced OCR System with PyTorch Compatibility Fix
"""
# Import system modules first
import os
import sys

# Set environment variables to prevent PyTorch-Streamlit conflicts
os.environ["STREAMLIT_SERVER_WATCH_FILES"] = "false"
os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"

# Import remaining modules
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torch.cuda.amp import autocast
import pandas as pd
import tempfile
import time
import base64
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoTokenizer
)
import re
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import io
from datetime import datetime

# Disable Streamlit's file watcher for all modules
import streamlit.watcher.path_watcher
original_watch_file = streamlit.watcher.path_watcher.watch_file

def safe_watch_file(path, callback):
    # Skip watching PyTorch modules
    if 'torch' in path or 'transformers' in path:
        return lambda: None  # Return a no-op stop function
    return original_watch_file(path, callback)

# Apply the monkey patch
streamlit.watcher.path_watcher.watch_file = safe_watch_file

# Set page configuration
st.set_page_config(
    page_title="Advanced OCR System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1565C0;
    }
    .metric-label {
        font-size: 1rem;
        color: #424242;
    }
    .highlight {
        background-color: #ffff00;
        padding: 0 4px;
        border-radius: 3px;
    }
    .image-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #616161;
        font-size: 0.8rem;
    }
    .stButton>button {
        background-color: #1976D2;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    .success-text {
        color: #2E7D32;
        font-weight: bold;
    }
    .error-text {
        color: #C62828;
        font-weight: bold;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .warning-box {
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .progress-container {
        margin: 20px 0;
    }
    /* Custom loader */
    .loader {
        border: 16px solid #f3f3f3;
        border-radius: 50%;
        border-top: 16px solid #3498db;
        width: 120px;
        height: 120px;
        animation: spin 2s linear infinite;
        margin: 20px auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for image processing
def add_logo_to_sidebar():
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="color: #1E88E5;">OCR System</h1>
        <p style="color: #424242;">Powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

def preprocess_image(image):
    """Apply advanced preprocessing techniques to the image"""
    # Convert to CV2 format
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to handle varying illumination
    thresh = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply deblurring with unsharp masking
    gaussian = cv2.GaussianBlur(thresh, (0, 0), 3)
    deblurred = cv2.addWeighted(thresh, 1.8, gaussian, -0.8, 0)
    
    # Apply sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(deblurred, -1, kernel)
    
    # Denoise using non-local means
    denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)
    
    # Convert back to PIL and RGB
    pil_image = Image.fromarray(denoised)
    rgb_image = Image.merge('RGB', [pil_image, pil_image, pil_image])
    
    return rgb_image, denoised  # Return both processed PIL image and CV2 image for display

def normalize_text(text):
    """Normalize text for soft matching"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text

def create_preprocessing_visualization(original_img, processed_img):
    """Create visualization of preprocessing steps"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Processed image
    axes[1].imshow(processed_img, cmap='gray')
    axes[1].set_title("Processed Image")
    axes[1].axis('off')
    
    fig.tight_layout()
    return fig

def display_confidence_visualization(similarity):
    """Create a visualization for the confidence score"""
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Create a horizontal bar chart
    bar_colors = ['#f44336', '#ff9800', '#ffeb3b', '#4caf50']
    color_idx = min(int(similarity * 4), 3)  # Map to 0-3 index
    
    ax.barh(['Confidence'], [similarity], color=bar_colors[color_idx])
    ax.barh([''], [1-similarity], left=[similarity], color='#f5f5f5')
    
    # Add a vertical line at common thresholds
    ax.axvline(x=0.85, color='black', linestyle='--', alpha=0.5)
    
    # Add text labels
    for x, label in [(0.2, 'Low'), (0.5, 'Medium'), (0.85, 'High')]:
        ax.text(x, 0, label, ha='center', va='center', fontsize=10, color='black')
    
    # Add the value as text
    ax.text(similarity, 0, f'{similarity:.2f}', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white' if similarity > 0.5 else 'black')
    
    # Configure the axes
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Confidence Score')
    ax.get_yaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    fig.tight_layout()
    return fig

@st.cache_resource
def load_trocr_model():
    """Load and cache TrOCR model to avoid reloading"""
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
        
        # Set TrOCR configuration
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.max_length = 64
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3
        model.config.length_penalty = 2.0
        model.config.num_beams = 4
        
        return processor, model
    except Exception as e:
        st.error(f"Error loading TrOCR model: {e}")
        return None, None

@st.cache_resource
def load_pali_model(hf_token):
    """Load and cache PaLI-Gemma model to avoid reloading"""
    if not hf_token:
        return None, None, None
    
    try:
        model_name = "google/paligemma-3b-pt-224"
        processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            token=hf_token
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        
        return processor, model, tokenizer
    except Exception as e:
        st.error(f"Error loading PaLI-Gemma model: {e}")
        return None, None, None

class OCRSystem:
    def __init__(self, hf_token=None):
        """Initialize the OCR system"""
        # Add a placeholder for progress
        self.progress_placeholder = st.empty()
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token
        
        # Initialize model flags
        self.trocr_initialized = False
        self.pali_initialized = False
    
    def init_trocr(self):
        """Initialize TrOCR model"""
        with self.progress_placeholder.container():
            st.info("Loading TrOCR model... This might take a moment.")
            progress_bar = st.progress(0)
            
            # Simulate loading progress
            for i in range(100):
                # Update progress bar
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            
            try:
                self.trocr_processor, self.trocr_model = load_trocr_model()
                
                if self.trocr_processor is not None and self.trocr_model is not None:
                    self.trocr_model.to(self.device)
                    self.trocr_initialized = True
                    st.success("TrOCR model loaded successfully!")
                else:
                    st.error("Failed to load TrOCR model")
                    
            except Exception as e:
                st.error(f"Error loading TrOCR model: {e}")
    
    def init_pali(self):
        """Initialize PaLI-Gemma model (optional)"""
        if not self.hf_token:
            st.warning("No Hugging Face token provided. Skipping PaLI-Gemma initialization.")
            return
        
        with self.progress_placeholder.container():
            st.info("Loading PaLI-Gemma model... This might take a moment.")
            progress_bar = st.progress(0)
            
            # Simulate loading progress
            for i in range(100):
                # Update progress bar
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            
            try:
                self.pali_processor, self.pali_model, self.pali_tokenizer = load_pali_model(self.hf_token)
                
                if self.pali_processor is not None and self.pali_model is not None and self.pali_tokenizer is not None:
                    self.pali_initialized = True
                    st.success("PaLI-Gemma model loaded successfully!")
                else:
                    st.error("Failed to load PaLI-Gemma model")
                    
            except Exception as e:
                st.error(f"Error loading PaLI-Gemma model: {e}")
    
    def process_image(self, image):
        """Process a single image through the full pipeline"""
        results = {}
        
        try:
            # Preprocess image
            processed_image, cv2_processed = preprocess_image(image)
            results['processed_image'] = processed_image
            results['cv2_processed'] = cv2_processed
            
            # Create visualization of preprocessing
            results['preprocessing_viz'] = create_preprocessing_visualization(np.array(image), cv2_processed)
            
            # Only proceed if TrOCR is initialized
            if not self.trocr_initialized:
                with self.progress_placeholder.container():
                    self.init_trocr()
            
            # Initial TrOCR prediction
            with self.progress_placeholder.container():
                st.info("Performing OCR analysis...")
                progress_bar = st.progress(0)
                
                # Simulate processing time
                for i in range(100):
                    progress_bar.progress(i + 1)
                    time.sleep(0.01)
                
                try:
                    pixel_values = self.trocr_processor(images=processed_image, return_tensors="pt").pixel_values.to(self.device)
                    
                    with torch.no_grad(), autocast(enabled=self.device.type == 'cuda'):
                        generated_ids = self.trocr_model.generate(pixel_values)
                        initial_pred = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    results['initial_prediction'] = initial_pred
                    st.success("OCR analysis complete!")
                except Exception as e:
                    st.error(f"Error during OCR analysis: {e}")
                    results['initial_prediction'] = "Error during analysis"
                    return results
            
            # Refined prediction with PaLI-Gemma (if initialized)
            if not self.pali_initialized and self.hf_token:
                with self.progress_placeholder.container():
                    self.init_pali()
                    
            if self.pali_initialized:
                with self.progress_placeholder.container():
                    st.info("Refining OCR results using PaLI-Gemma...")
                    progress_bar = st.progress(0)
                    
                    # Simulate processing time
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        time.sleep(0.02)
                    
                    try:
                        refined_pred = self.refine_prediction(processed_image, initial_pred)
                        results['refined_prediction'] = refined_pred
                        
                        # Calculate similarity for confidence visualization
                        similarity = SequenceMatcher(None, 
                                                  normalize_text(initial_pred), 
                                                  normalize_text(refined_pred)).ratio()
                        results['confidence_viz'] = display_confidence_visualization(similarity)
                        results['confidence_score'] = similarity
                        
                        st.success("Refinement complete!")
                    except Exception as e:
                        st.error(f"Error during refinement: {e}")
                        results['refined_prediction'] = initial_pred  # Fallback to initial prediction
            else:
                results['refined_prediction'] = initial_pred  # No refinement available
                
        except Exception as e:
            st.error(f"Error in processing pipeline: {e}")
            # Make sure we have some results to return
            if 'initial_prediction' not in results:
                results['initial_prediction'] = "Error during processing"
            if 'refined_prediction' not in results:
                results['refined_prediction'] = results.get('initial_prediction', "Error during processing")
        
        return results
    
    def refine_prediction(self, image, initial_pred):
        """Refine the prediction using PaLI-Gemma with proper token formatting"""
        try:
            # Explicitly add the <image> token to the beginning of the prompt
            prompt = f"<image> OCR correction task: The text in this image appears to be '{initial_pred}'. What is the correct text?"
            
            # Process the inputs with the processor
            inputs = self.pali_processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Set generation parameters
            generation_config = {
                "max_new_tokens": 256,
                "temperature": 0.2,
                "do_sample": True,
                "top_p": 0.92,
                "top_k": 50,
                "repetition_penalty": 1.2,
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 3
            }
            
            with torch.no_grad():
                outputs = self.pali_model.generate(**inputs, **generation_config)
            
            # Decode and extract the refined text
            refined_text = self.pali_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the corrected text using regex pattern matching
            match = re.search(r'correct text is[:\s]*[\"\']*([^\"\']*)[\"\']*', refined_text, re.IGNORECASE)
            if match:
                extracted_text = match.group(1).strip()
                return extracted_text
            
            # If specific pattern not found, try to clean up the output
            refined_text = refined_text.replace(prompt, "").strip()
            
            # Look for the actual text which often comes after a colon or quotes
            match = re.search(r'[:\"]([^:\"]+)[\"]*$', refined_text)
            if match:
                return match.group(1).strip()
                
            return refined_text
        except Exception as e:
            st.error(f"Refinement failed: {e}")
            return initial_pred

def main():
    # Add logo to sidebar
    add_logo_to_sidebar()
    
    # Main header
    st.markdown("<h1 class='main-header'>Advanced OCR System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 30px;'>Extract and enhance text from images with AI</p>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("<div class='sub-header'>Settings</div>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'ocr_system' not in st.session_state:
        hf_token = st.sidebar.text_input("Hugging Face Token (for PaLI-Gemma)", 
                                        type="password", 
                                        value="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",
                                        help="Token for accessing Hugging Face models. Required for PaLI-Gemma refinement.")
        
        use_pali = st.sidebar.checkbox("Use PaLI-Gemma for refinement", 
                                       value=True,
                                       help="Enable to use PaLI-Gemma model for refining OCR results. Requires a Hugging Face token.")
        
        # Create OCR system
        st.session_state.ocr_system = OCRSystem(hf_token if use_pali else None)
        
        # UI options
        st.sidebar.markdown("<div class='sub-header'>Display Options</div>", unsafe_allow_html=True)
        st.session_state.show_preprocessing = st.sidebar.checkbox("Show preprocessing visualization", value=True)
        st.session_state.show_confidence = st.sidebar.checkbox("Show confidence visualization", value=True)
    
    # Add a refresh button to sidebar
    if st.sidebar.button("Reload Models"):
        # Clear session state
        st.session_state.pop('ocr_system', None)
        st.experimental_rerun()
    
    # Input options
    st.markdown("<div class='sub-header'>Image Input</div>", unsafe_allow_html=True)
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Upload Image", "Take Photo", "Batch Processing"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload an image containing text", type=["jpg", "jpeg", "png", "bmp"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='card'><h3>Original Image</h3></div>", unsafe_allow_html=True)
                st.image(image, use_container_width=True, caption="Uploaded Image")
            
            # Process button
            if st.button("Process Image", key="process_single"):
                with st.spinner("Processing..."):
                    # Process the image
                    results = st.session_state.ocr_system.process_image(image)
                    
                    # Display processed image
                    with col2:
                        st.markdown("<div class='card'><h3>Processed Image</h3></div>", unsafe_allow_html=True)
                        st.image(results['cv2_processed'], use_container_width=True, caption="Processed Image")
                    
                    # Display preprocessing visualization if enabled
                    if st.session_state.show_preprocessing:
                        st.markdown("<div class='sub-header'>Preprocessing Steps</div>", unsafe_allow_html=True)
                        st.pyplot(results['preprocessing_viz'])
                    
                    # Display results
                    st.markdown("<div class='sub-header'>OCR Results</div>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<div class='card'><h3>Initial OCR Result</h3></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='info-box'>{results['initial_prediction']}</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<div class='card'><h3>Refined Result</h3></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='info-box'>{results['refined_prediction']}</div>", unsafe_allow_html=True)
                    
                    # Display confidence visualization if enabled and available
                    if st.session_state.show_confidence and 'confidence_viz' in results:
                        st.markdown("<div class='sub-header'>Confidence Assessment</div>", unsafe_allow_html=True)
                        st.pyplot(results['confidence_viz'])
                        
                    # Download button for extracted text
                    text_result = results.get('refined_prediction', results.get('initial_prediction', ''))
                    if text_result:
                        st.download_button(
                            "Download as Text File",
                            text_result,
                            f"ocr_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            "text/plain"
                        )
    
    with tab2:
        st.markdown("<div class='info-box'>Take a photo with your camera for immediate OCR processing.</div>", unsafe_allow_html=True)
        camera_image = st.camera_input("Take a photo")
        
        if camera_image is not None:
            # Process button
            if st.button("Process Photo", key="process_camera"):
                with st.spinner("Processing..."):
                    # Process the camera image
                    image = Image.open(camera_image).convert("RGB")
                    results = st.session_state.ocr_system.process_image(image)
                    
                    # Display results
                    st.markdown("<div class='sub-header'>OCR Results</div>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<div class='card'><h3>Initial OCR Result</h3></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='info-box'>{results['initial_prediction']}</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<div class='card'><h3>Refined Result</h3></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='info-box'>{results['refined_prediction']}</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='warning-box'>Batch processing allows you to process multiple images at once and export results to CSV.</div>", unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} files")
            
            # Display thumbnails of uploaded images
            columns = st.columns(4)
            for i, uploaded_file in enumerate(uploaded_files[:8]):  # Show first 8 images
                with columns[i % 4]:
                    st.image(Image.open(uploaded_file), width=100)
            
            if len(uploaded_files) > 8:
                st.write(f"...and {len(uploaded_files) - 8} more")
            
            # Process button
            if st.button("Process All Images", key="process_batch"):
                with st.spinner("Processing batch..."):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process each image
                    results = []
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Update progress
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing image {i+1}/{len(uploaded_files)}")
                        
                        # Process the image
                        try:
                            image = Image.open(uploaded_file).convert("RGB")
                            image_results = st.session_state.ocr_system.process_image(image)
                            
                            results.append({
                                'filename': uploaded_file.name,
                                'initial_text': image_results['initial_prediction'],
                                'refined_text': image_results['refined_prediction'],
                                'confidence': image_results.get('confidence_score', 0)
                            })
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")
                    
                    # Create DataFrame and show results
                    df = pd.DataFrame(results)
                    st.dataframe(df)
                    
                    # Download button for CSV
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download CSV Results",
                        csv,
                        f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        key="download-csv"
                    )
    
    # Add installation section
    with st.expander("Installation Instructions"):
        st.markdown("""
        ## Installation Instructions
        
        This OCR system requires several dependencies. Follow these steps to install everything correctly:
        
        ### Required Dependencies
        
        ```bash
        # Create a virtual environment (recommended)
        python -m venv ocr_env
        source ocr_env/bin/activate  # On Windows: ocr_env\\Scripts\\activate
        
        # Install core dependencies
        pip install streamlit==1.24.0
        pip install pillow>=9.0.0
        pip install opencv-python-headless==4.7.0.72
        pip install torch torchvision
        pip install transformers
        pip install numpy pandas matplotlib
        ```
        
        ### Troubleshooting Common Issues
        
        1. **PyTorch-Streamlit Compatibility**: If you encounter errors with PyTorch and Streamlit's file watcher, add these environment variables before running:
           ```bash
           export STREAMLIT_SERVER_WATCH_FILES=false
           export TORCH_USE_RTLD_GLOBAL=YES
           ```
        
        2. **Missing 'distutils'**: If you get a "No module named 'distutils'" error, install Python development tools:
           ```bash
           sudo apt-get update
           sudo apt-get install python3-dev python3-distutils
           ```
        
        3. **OpenCV Issues**: If OpenCV installation fails, try the headless version:
           ```bash
           pip install opencv-python-headless==4.7.0.72
           ```
        
        4. **CUDA Issues**: If you have GPU compatibility problems, install the CPU-only version of PyTorch:
           ```bash
           pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
           ```
        """)
    
    # Footer
    st.markdown("<div class='footer'>© 2025 Advanced OCR System</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
