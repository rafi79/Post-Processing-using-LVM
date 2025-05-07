"""
Advanced OCR System with PyTorch Compatibility Fix

This version maintains all original features while fixing the PyTorch-Streamlit
compatibility issues.
"""
import os
import sys

# CRITICAL: Set environment variables to fix compatibility issues
os.environ["STREAMLIT_SERVER_WATCH_FILES"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"  # Helps with PyTorch loading issues

# First, import regular packages
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import time
import re
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import io
import traceback
import base64
from datetime import datetime

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

# Flag to determine if PyTorch imports worked
PYTORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
CV2_AVAILABLE = False

# Try to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    st.warning("OpenCV (cv2) not found. Installing a compatible version...")
    try:
        # Try to install with pip
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.7.0.72"])
        import cv2
        CV2_AVAILABLE = True
        st.success("OpenCV installed successfully!")
    except Exception as e:
        st.error(f"Failed to install OpenCV: {e}")
        from PIL import ImageFilter, ImageEnhance

# Now try to import PyTorch safely
try:
    import torch
    from torch.cuda.amp import autocast
    PYTORCH_AVAILABLE = True
except ImportError:
    st.warning("PyTorch not found. Some advanced OCR features will be disabled.")
except Exception as e:
    st.warning(f"Error importing PyTorch: {e}. Some advanced OCR features will be disabled.")

# Try to import transformers safely
if PYTORCH_AVAILABLE:
    try:
        from transformers import (
            TrOCRProcessor,
            VisionEncoderDecoderModel,
            AutoProcessor,
            AutoModelForVision2Seq,
            AutoTokenizer
        )
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        st.warning("Transformers library not found. Advanced OCR models will be disabled.")
    except Exception as e:
        st.warning(f"Error importing Transformers: {e}. Advanced OCR models will be disabled.")

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
    if not CV2_AVAILABLE:
        # Simple preprocessing with PIL
        if image.mode != 'L':
            image = image.convert('L')
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(2.0)
        # Sharpen
        sharpened = enhanced.filter(ImageFilter.SHARPEN)
        # Return both PIL image and numpy array
        return sharpened, np.array(sharpened)
    
    # OpenCV preprocessing
    try:
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
        # Handle different modes for PIL merge
        if hasattr(pil_image, 'mode') and pil_image.mode != 'RGB':
            # If using PIL's merge, we need RGB channels
            rgb_image = Image.merge('RGB', [pil_image, pil_image, pil_image])
        else:
            # Fallback for placeholder
            rgb_image = Image.new('RGB', pil_image.size)
            rgb_image.paste(pil_image)
    
    except Exception as e:
        st.warning(f"Error in image preprocessing: {e}. Using simplified preprocessing.")
        # Simplified fallback preprocessing
        if isinstance(image, np.ndarray):
            if image.ndim == 3:  # Color image
                # Convert to grayscale
                gray = np.mean(image, axis=2).astype(np.uint8)
            else:
                gray = image
        else:  # PIL Image
            gray = np.array(image.convert('L'))
        
        # Apply simple thresholding
        _, denoised = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        pil_image = Image.fromarray(denoised)
        rgb_image = Image.new('RGB', pil_image.size)
        rgb_image.paste(pil_image)
    
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
    
    # Make sure inputs are numpy arrays
    if not isinstance(original_img, np.ndarray):
        original_img = np.array(original_img)
    if not isinstance(processed_img, np.ndarray):
        processed_img = np.array(processed_img)
    
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

# Create safe model loading functions that use global variables
@st.cache_resource
def load_trocr_model():
    """Load and cache TrOCR model to avoid reloading"""
    if not TRANSFORMERS_AVAILABLE or not PYTORCH_AVAILABLE:
        return None, None
    
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
    if not TRANSFORMERS_AVAILABLE or not PYTORCH_AVAILABLE:
        return None, None, None
    
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

# Function to try tesseract if PyTorch models fail
def process_with_tesseract(image):
    """Fallback to Tesseract OCR if available"""
    try:
        import pytesseract
        text = pytesseract.image_to_string(image)
        return text
    except ImportError:
        return "pytesseract not installed. Install with: pip install pytesseract"
    except Exception as e:
        return f"Error using Tesseract: {e}"

# Function to try EasyOCR if PyTorch models fail
def process_with_easyocr(image):
    """Fallback to EasyOCR if available"""
    try:
        import easyocr
        reader = easyocr.Reader(['en'])
        results = reader.readtext(np.array(image))
        text = " ".join([result[1] for result in results])
        return text
    except ImportError:
        return "easyocr not installed. Install with: pip install easyocr"
    except Exception as e:
        return f"Error using EasyOCR: {e}"

class OCRSystem:
    def __init__(self, hf_token=None):
        """Initialize the OCR system"""
        # Add a placeholder for progress
        self.progress_placeholder = st.empty()
        
        # Initialize device
        if PYTORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
            
        self.hf_token = hf_token
        
        # Initialize model flags
        self.trocr_initialized = False
        self.pali_initialized = False
        
        # Cache to avoid multiple inits in same session
        if 'trocr_initialized' in st.session_state:
            self.trocr_initialized = st.session_state.trocr_initialized
            
        if 'pali_initialized' in st.session_state:
            self.pali_initialized = st.session_state.pali_initialized
    
    def init_trocr(self):
        """Initialize TrOCR model with improved error handling"""
        if not PYTORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            st.warning("PyTorch or Transformers not available. Cannot initialize TrOCR.")
            return
            
        with self.progress_placeholder.container():
            st.info("Loading TrOCR model... This might take a moment.")
            progress_bar = st.progress(0)
            
            # Simulated loading progress
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            
            # Load model with caching
            self.trocr_processor, self.trocr_model = load_trocr_model()
            
            if self.trocr_processor is not None and self.trocr_model is not None:
                # Move model to device
                try:
                    self.trocr_model.to(self.device)
                    self.trocr_initialized = True
                    st.session_state.trocr_initialized = True
                    st.success("TrOCR model loaded successfully!")
                except Exception as e:
                    st.error(f"Error moving TrOCR model to device: {e}")
                    # Fallback to CPU
                    if PYTORCH_AVAILABLE:
                        self.device = torch.device("cpu")
                        try:
                            self.trocr_model.to(self.device)
                            self.trocr_initialized = True
                            st.session_state.trocr_initialized = True
                            st.success("TrOCR model loaded successfully (using CPU)!")
                        except Exception as e2:
                            st.error(f"Could not initialize TrOCR model: {e2}")
            else:
                st.error("Failed to load TrOCR model. OCR functionality will be limited.")
                self.trocr_initialized = False
    
    def init_pali(self):
        """Initialize PaLI-Gemma model with improved error handling"""
        if not PYTORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            st.warning("PyTorch or Transformers not available. Cannot initialize PaLI-Gemma.")
            return
            
        if not self.hf_token:
            st.warning("No Hugging Face token provided. Skipping PaLI-Gemma initialization.")
            return
        
        with self.progress_placeholder.container():
            st.info("Loading PaLI-Gemma model... This might take a moment.")
            progress_bar = st.progress(0)
            
            # Simulated loading progress
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            
            # Load model with caching
            self.pali_processor, self.pali_model, self.pali_tokenizer = load_pali_model(self.hf_token)
            
            if self.pali_processor is not None and self.pali_model is not None and self.pali_tokenizer is not None:
                self.pali_initialized = True
                st.session_state.pali_initialized = True
                st.success("PaLI-Gemma model loaded successfully!")
            else:
                st.error("Failed to load PaLI-Gemma model. Text refinement will be disabled.")
                self.pali_initialized = False
    
    def process_image(self, image):
        """Process a single image through the full pipeline with enhanced error handling"""
        results = {}
        
        # Add a safety net for processing
        try:
            # Preprocess image
            processed_image, cv2_processed = preprocess_image(image)
            results['processed_image'] = processed_image
            results['cv2_processed'] = cv2_processed
            
            # Create visualization of preprocessing
            results['preprocessing_viz'] = create_preprocessing_visualization(np.array(image.convert('L')), cv2_processed)
            
            # Check if we can use advanced OCR models
            if PYTORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
                # Initialize TrOCR if needed
                if not self.trocr_initialized:
                    with self.progress_placeholder.container():
                        self.init_trocr()
                
                # Initial TrOCR prediction
                with self.progress_placeholder.container():
                    st.info("Performing OCR analysis...")
                    progress_bar = st.progress(0)
                    
                    # Simulated processing time
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        time.sleep(0.01)
                    
                    # Check if model initialized successfully
                    if not self.trocr_initialized:
                        st.warning("TrOCR model not available. Trying alternative OCR methods.")
                        
                        # Try Tesseract first
                        tesseract_result = process_with_tesseract(processed_image)
                        if not tesseract_result.startswith("Error") and not tesseract_result.startswith("pytesseract not installed"):
                            results['initial_prediction'] = tesseract_result
                            st.success("OCR analysis completed with Tesseract!")
                        else:
                            # Try EasyOCR next
                            easyocr_result = process_with_easyocr(processed_image)
                            if not easyocr_result.startswith("Error") and not easyocr_result.startswith("easyocr not installed"):
                                results['initial_prediction'] = easyocr_result
                                st.success("OCR analysis completed with EasyOCR!")
                            else:
                                results['initial_prediction'] = "Could not perform OCR. Please install either Tesseract or EasyOCR."
                    else:
                        try:
                            pixel_values = self.trocr_processor(images=processed_image, return_tensors="pt").pixel_values.to(self.device)
                            
                            with torch.no_grad(), autocast(enabled=self.device.type == 'cuda'):
                                generated_ids = self.trocr_model.generate(pixel_values)
                                initial_pred = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                            
                            results['initial_prediction'] = initial_pred
                            st.success("OCR analysis complete!")
                        except Exception as e:
                            error_details = traceback.format_exc()
                            st.error(f"Error during OCR analysis: {e}")
                            st.info("Trying alternative OCR methods...")
                            
                            # Try Tesseract as a fallback
                            tesseract_result = process_with_tesseract(processed_image)
                            if not tesseract_result.startswith("Error") and not tesseract_result.startswith("pytesseract not installed"):
                                results['initial_prediction'] = tesseract_result
                                st.success("OCR analysis completed with Tesseract!")
                            else:
                                # Try EasyOCR as another fallback
                                easyocr_result = process_with_easyocr(processed_image)
                                if not easyocr_result.startswith("Error") and not easyocr_result.startswith("easyocr not installed"):
                                    results['initial_prediction'] = easyocr_result
                                    st.success("OCR analysis completed with EasyOCR!")
                                else:
                                    results['initial_prediction'] = "Failed to perform OCR. Please check the installation."
                                    st.error("All OCR methods failed.")
                                    st.expander("Technical Error Details").code(error_details)
                
                # Refined prediction with PaLI-Gemma (if initialized)
                if not self.pali_initialized and self.hf_token:
                    # Try to initialize PaLI
                    self.init_pali()
                    
                if self.pali_initialized:
                    with self.progress_placeholder.container():
                        st.info("Refining OCR results using PaLI-Gemma...")
                        progress_bar = st.progress(0)
                        
                        # Simulated processing time
                        for i in range(100):
                            progress_bar.progress(i + 1)
                            time.sleep(0.02)
                        
                        try:
                            refined_pred = self.refine_prediction(processed_image, results['initial_prediction'])
                            results['refined_prediction'] = refined_pred
                            
                            # Calculate similarity for confidence visualization
                            similarity = SequenceMatcher(None, 
                                                       normalize_text(results['initial_prediction']), 
                                                       normalize_text(refined_pred)).ratio()
                            results['confidence_viz'] = display_confidence_visualization(similarity)
                            results['confidence_score'] = similarity
                            
                            st.success("Refinement complete!")
                        except Exception as e:
                            error_details = traceback.format_exc()
                            st.error(f"Error during refinement: {e}")
                            st.expander("Technical Error Details").code(error_details)
                            results['refined_prediction'] = results['initial_prediction']  # Fallback to initial prediction
                else:
                    results['refined_prediction'] = results['initial_prediction']  # No refinement available
            else:
                # Use alternative OCR methods if PyTorch is not available
                st.info("Advanced OCR models not available. Using alternative OCR methods.")
                
                # Try Tesseract first
                tesseract_result = process_with_tesseract(processed_image)
                if not tesseract_result.startswith("Error") and not tesseract_result.startswith("pytesseract not installed"):
                    results['initial_prediction'] = tesseract_result
                    st.success("OCR analysis completed with Tesseract!")
                else:
                    # Try EasyOCR next
                    easyocr_result = process_with_easyocr(processed_image)
                    if not easyocr_result.startswith("Error") and not easyocr_result.startswith("easyocr not installed"):
                        results['initial_prediction'] = easyocr_result
                        st.success("OCR analysis completed with EasyOCR!")
                    else:
                        results['initial_prediction'] = "Could not perform OCR. Please install either Tesseract or EasyOCR."
                
                # No refinement available
                results['refined_prediction'] = results.get('initial_prediction', "OCR failed")
                
        except Exception as e:
            error_details = traceback.format_exc()
            st.error(f"Error in OCR processing pipeline: {e}")
            st.expander("Technical Error Details").code(error_details)
            # Ensure we have some results to return
            if 'initial_prediction' not in results:
                results['initial_prediction'] = "Error during processing"
            if 'refined_prediction' not in results:
                results['refined_prediction'] = results.get('initial_prediction', "Error during processing")
        
        return results
    
    def refine_prediction(self, image, initial_pred):
        """Refine the prediction using PaLI-Gemma with proper token formatting and error handling"""
        if not PYTORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            return initial_pred
            
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
                                        help="Token for accessing Hugging Face models. Required for PaLI-Gemma refinement.")
        
        use_pali = st.sidebar.checkbox("Use PaLI-Gemma for refinement", 
                                       value=True if TRANSFORMERS_AVAILABLE and PYTORCH_AVAILABLE else False,
                                       help="Enable to use PaLI-Gemma model for refining OCR results. Requires a Hugging Face token.",
                                       disabled=not (TRANSFORMERS_AVAILABLE and PYTORCH_AVAILABLE))
        
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
    
    # Show dependency status
    with st.sidebar.expander("System Status"):
        st.write("OpenCV: ", "✅ Available" if CV2_AVAILABLE else "❌ Not available")
        st.write("PyTorch: ", "✅ Available" if PYTORCH_AVAILABLE else "❌ Not available")
        st.write("Transformers: ", "✅ Available" if TRANSFORMERS_AVAILABLE else "❌ Not available")
        
        # Check for Tesseract
        try:
            import pytesseract
            tesseract_version = pytesseract.get_tesseract_version()
            st.write(f"Tesseract: ✅ Available (v{tesseract_version})")
        except:
            st.write("Tesseract: ❌ Not available")
            
        # Check for EasyOCR
        try:
            import easyocr
            st.write("EasyOCR: ✅ Available")
        except:
            st.write("EasyOCR: ❌ Not available")
    
    # Input options
    st.markdown("<div class='sub-header'>Image Input</div>", unsafe_allow_html=True)
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Upload Image", "Take Photo", "Batch Processing"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload an image containing text", type=["jpg", "jpeg", "png", "bmp"])
        
        if uploaded_file is not None:
            try:
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
                        if st.session_state.show_preprocessing and 'preprocessing_viz' in results:
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
                            
                        # Download button for the text
                        text_result = results.get('refined_prediction', results.get('initial_prediction', ''))
                        if text_result:
                            st.download_button(
                                "Download as Text File",
                                text_result,
                                f"ocr_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                "text/plain"
                            )
            except Exception as e:
                st.error(f"Error processing the uploaded image: {e}")
                st.info("Make sure the uploaded file is a valid image format.")
    
    with tab2:
        st.markdown("<div class='info-box'>Take a photo with your camera for immediate OCR processing.</div>", unsafe_allow_html=True)
        camera_image = st.camera_input("Take a photo")
        
        if camera_image is not None:
            try:
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
                            
                        # Display confidence visualization if enabled and available
                        if st.session_state.show_confidence and 'confidence_viz' in results:
                            st.markdown("<div class='sub-header'>Confidence Assessment</div>", unsafe_allow_html=True)
                            st.pyplot(results['confidence_viz'])
                            
                        # Download button for the text
                        text_result = results.get('refined_prediction', results.get('initial_prediction', ''))
                        if text_result:
                            st.download_button(
                                "Download as Text File",
                                text_result,
                                f"ocr_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                "text/plain"
                            )
            except Exception as e:
                st.error(f"Error processing the camera image: {e}")
    
    with tab3:
        st.markdown("<div class='warning-box'>Batch processing allows you to process multiple images at once and export results to CSV.</div>", unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} files")
            
            # Display thumbnails of uploaded images
            columns = st.columns(4)
            for i, uploaded_file in enumerate(uploaded_files[:8]):  # Show first 8 images
                with columns[i % 4]:
                    try:
                        st.image(Image.open(uploaded_file), width=100)
                    except Exception as e:
                        st.warning(f"Could not display thumbnail for {uploaded_file.name}")
            
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
                    failed = 0
                    
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
                                'confidence': image_results.get('confidence_score', 0),
                                'status': 'success'
                            })
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")
                            failed += 1
                            results.append({
                                'filename': uploaded_file.name,
                                'initial_text': 'Error during processing',
                                'refined_text': 'Error during processing',
                                'confidence': 0,
                                'status': 'failed'
                            })
                    
                    # Create DataFrame and show results
                    if results:
                        df = pd.DataFrame(results)
                        
                        # Display summary
                        if failed > 0:
                            st.warning(f"{failed} out of {len(uploaded_files)} images failed to process.")
                        
                        # Display the DataFrame
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
                    else:
                        st.error("No results were obtained from batch processing.")
    
    # Add info section
    with st.expander("About this OCR System"):
        st.markdown("""
        ## Enhanced OCR System
        
        This system combines multiple AI models to extract text from images with high accuracy:
        
        1. **TrOCR**: Microsoft's Transformer-based OCR model, fine-tuned for text recognition
        2. **PaLI-Gemma**: Google's multimodal model that can understand both images and text for refinement
        3. **Fallbacks**: Tesseract OCR and EasyOCR if the advanced models are not available
        
        ### Preprocessing Pipeline
        
        Images go through a sophisticated preprocessing pipeline:
        - Adaptive thresholding for handling varying illumination
        - Deblurring with unsharp masking
        - Sharpening with kernel operations
        - Denoising with non-local means algorithm
        
        ### Best Practices for Good Results
        
        - Use well-lit, clear images with good contrast
        - Avoid excessive skew or distortion
        - For handwritten text, ensure it's written clearly
        - Crop images to focus on the text area when possible
        """)
    
    # Add installation section
    with st.expander("Installation Instructions"):
        st.markdown("""
        ## Installation Instructions
        
        This OCR system requires several dependencies. Follow these steps to install everything correctly:
        
        ### Option 1: Using pip
        
        ```bash
        # Create and activate a virtual environment (recommended)
        python -m venv ocr_env
        source ocr_env/bin/activate  # On Windows: ocr_env\\Scripts\\activate
        
        # Update pip and install core dependencies
        pip install --upgrade pip
        pip install streamlit==1.24.0
        pip install pillow>=9.0.0
        pip install opencv-python-headless==4.7.0.72
        pip install torch torchvision
        pip install transformers
        pip install numpy pandas matplotlib
        
        # Optional fallback OCR engines
        pip install pytesseract  # Also requires Tesseract installed on your system
        pip install easyocr
        ```
        
        ### Option 2: Using the runner script
        
        This application provides a special runner script that works around PyTorch-Streamlit compatibility issues:
        
        ```python
        # Create a file named run_ocr.py with the following code:
        import os
        import sys

        # Set environment variables to fix compatibility issues
        os.environ["STREAMLIT_SERVER_WATCH_FILES"] = "false"
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"

        # Import streamlit after setting environment variables
        import streamlit.web.bootstrap as bootstrap
        
        # Run the app with file watcher disabled
        bootstrap.run("advanced_ocr_app.py", "", [], flag_options={
            'server.fileWatcherType': 'none',
            'browser.serverAddress': 'localhost',
            'server.headless': True,
            'server.runOnSave': False
        })
        ```
        
        Then run the application with:
        ```bash
        python run_ocr.py
        ```
        
        ### Common Issues and Solutions
        
        1. **PyTorch-Streamlit Compatibility**: If you encounter errors with PyTorch and Streamlit's file watcher, use the runner script above.
        
        2. **Missing 'distutils'**: If you see a "No module named 'distutils'" error, install Python development tools:
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
