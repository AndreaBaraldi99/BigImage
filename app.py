"""
Ship Detection Dashboard
========================

Streamlit web application for maritime surveillance using deep learning.
This dashboard provides an intuitive interface for ship detection and segmentation.

Features:
- Real-time image upload and processing
- Interactive prediction visualization
- Confidence threshold adjustment
- Ship detection statistics
- Model performance metrics

Author: Ship Detection Research Team
Date: June 2025
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64
from typing import Tuple, Optional
from scipy import ndimage

# Import PyTorch with proper error handling
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    # Disable torch class warnings and set environment variable
    import os
    os.environ['TORCH_CLASSES_UNWRAP'] = '1'
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", message=".*torch.classes.*")
except ImportError as e:
    st.error(f"PyTorch import error: {e}")
    st.stop()

# Import albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as e:
    st.error(f"Albumentations import error: {e}")
    st.stop()

# Import plotly with fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Some advanced visualizations will be disabled.")

# Set page configuration
st.set_page_config(
    page_title="Ship Detection Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress PyTorch warnings and set environment variables
import os
import warnings
os.environ['TORCH_CLASSES_UNWRAP'] = '1'
os.environ['TORCH_CPP_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*path.*")

# Set matplotlib backend for better compatibility
import matplotlib
matplotlib.use('Agg')

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e3b8a;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# U-Net Model Definition (same as in notebook)
class UNet(nn.Module):
    """Standard U-Net implementation for ship segmentation."""
    
    def __init__(self, input_channels=3, upsample_mode='deconv'):
        super(UNet, self).__init__()
        self.upsample_mode = upsample_mode

        # Encoder path
        self.c1 = self.conv_block(input_channels, 16, dropout=0.1)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = self.conv_block(16, 32, dropout=0.1)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = self.conv_block(32, 64, dropout=0.2)
        self.p3 = nn.MaxPool2d(2)
        self.c4 = self.conv_block(64, 128, dropout=0.2)
        self.p4 = nn.MaxPool2d(2)

        # Bottleneck
        self.c5 = self.conv_block(128, 256, dropout=0.3)

        # Decoder path
        self.u6 = self.upsample(256, 128)
        self.c6 = self.conv_block(256, 128, dropout=0.2)
        self.u7 = self.upsample(128, 64)
        self.c7 = self.conv_block(128, 64, dropout=0.2)
        self.u8 = self.upsample(64, 32)
        self.c8 = self.conv_block(64, 32, dropout=0.1)
        self.u9 = self.upsample(32, 16)
        self.c9 = self.conv_block(32, 16, dropout=0.1)

        # Output layer
        self.final = nn.Conv2d(16, 1, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using He normal initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def conv_block(self, in_channels, out_channels, dropout=0.0):
        """Double convolution block with ReLU and dropout."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )

    def upsample(self, in_channels, out_channels):
        """Upsampling using transposed convolution or bilinear interpolation."""
        if self.upsample_mode == 'deconv':
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )

    def forward(self, x):
        """Forward pass through U-Net with skip connections."""
        # Encoder
        c1 = self.c1(x)
        p1 = self.p1(c1)
        c2 = self.c2(p1)
        p2 = self.p2(c2)
        c3 = self.c3(p2)
        p3 = self.p3(c3)
        c4 = self.c4(p3)
        p4 = self.p4(c4)

        # Bottleneck
        c5 = self.c5(p4)

        # Decoder with skip connections
        u6 = self.u6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.c6(u6)
        
        u7 = self.u7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.c7(u7)
        
        u8 = self.u8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.c8(u8)
        
        u9 = self.u9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.c9(u9)

        return self.final(c9)

class ShipDetectionInference:
    """Production inference pipeline for ship detection."""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._setup_transforms()
        
    def _setup_device(self, device: str) -> torch.device:
        if device == 'auto':
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        model = UNet(input_channels=3, upsample_mode='deconv')
        
        try:
            # Suppress additional warnings during model loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
        except FileNotFoundError:
            st.error(f"Model file not found: {model_path}")
            st.stop()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
        
        model.eval()
        model.to(self.device)
        return model
    
    def _setup_transforms(self) -> A.Compose:
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_np = np.array(image)
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict(self, image: Image.Image, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, float]:
        image_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits)
        
        prob_mask = probabilities.squeeze().cpu().numpy()
        binary_mask = (prob_mask > threshold).astype(np.uint8)
        max_confidence = float(prob_mask.max())
        
        return binary_mask, prob_mask, max_confidence
    
    def get_ship_statistics(self, binary_mask: np.ndarray) -> dict:
        total_pixels = binary_mask.size
        ship_pixels = binary_mask.sum()
        ship_percentage = (ship_pixels / total_pixels) * 100
        
        return {
            'ships_detected': bool(ship_pixels > 0),
            'ship_pixels': int(ship_pixels),
            'total_pixels': int(total_pixels),
            'coverage_percentage': float(ship_percentage)
        }

@st.cache_resource(show_spinner=False, max_entries=1)
def load_model():
    """Load model with caching for better performance."""
    model_path = "best_model_amp.pth"
    try:
        # Suppress warnings during caching
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ShipDetectionInference(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def create_overlay_visualization(original_image: Image.Image, prob_mask: np.ndarray, 
                                binary_mask: np.ndarray, threshold: float) -> Image.Image:
    """Create overlay visualization of predictions on original image."""
    # Resize masks to match original image size
    original_size = original_image.size
    prob_resized = Image.fromarray((prob_mask * 255).astype(np.uint8)).resize(original_size)
    binary_resized = Image.fromarray((binary_mask * 255).astype(np.uint8)).resize(original_size)
    
    # Convert to RGBA for overlay
    original_rgba = original_image.convert('RGBA')
    
    # Create colored overlay for ships
    overlay = Image.new('RGBA', original_size, (0, 0, 0, 0))
    overlay_array = np.array(overlay)
    
    # Add red overlay where ships are detected
    binary_array = np.array(binary_resized)
    ship_pixels = binary_array > 128
    overlay_array[ship_pixels] = [255, 0, 0, 100]  # Red with transparency
    
    overlay = Image.fromarray(overlay_array)
    result = Image.alpha_composite(original_rgba, overlay)
    
    return result.convert('RGB')

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🚢 Maritime Ship Detection Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Advanced AI-Powered Maritime Surveillance System
    Upload satellite or aerial maritime imagery to detect and analyze ship presence using 
    state-of-the-art deep learning technology.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model information
    with st.sidebar.expander("📊 Model Information", expanded=True):
        st.info("""
        **Architecture**: U-Net with Skip Connections  
        **Training Data**: Airbus Ship Detection Challenge  
        **Input Size**: 256×256 pixels  
        **Output**: Binary segmentation mask  
        """)
    
    # Load model
    with st.spinner("Loading AI model..."):
        inference_pipeline = load_model()
    
    if inference_pipeline is not None:
        st.sidebar.success("✅ Model loaded successfully")
        device_info = f"🖥️ Device: {inference_pipeline.device}"
        st.sidebar.info(device_info)
    else:
        st.sidebar.error("❌ Error loading model")
        st.error("Cannot proceed without a valid model. Please ensure 'best_model.pth' exists.")
        st.stop()
    
    # Prediction parameters
    st.sidebar.subheader("🎛️ Prediction Parameters")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Threshold for converting probability to binary prediction"
    )
    
    show_probability = st.sidebar.checkbox("Show Probability Map", value=True)
    show_overlay = st.sidebar.checkbox("Show Detection Overlay", value=True)
    
    # File upload
    st.subheader("📤 Upload Maritime Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload satellite or aerial imagery of maritime areas"
    )
    
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        st.subheader("🖼️ Uploaded Image")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Original Image", use_container_width=True)
        
        # Image information
        st.subheader("📋 Image Information")
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        
        with info_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📐 Dimensions</h4>
                <p>{image.size[0]} × {image.size[1]} pixels</p>
            </div>
            """, unsafe_allow_html=True)
        
        with info_col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎨 Mode</h4>
                <p>{image.mode}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with info_col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Format</h4>
                <p>{image.format or 'Unknown'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with info_col4:
            file_size = len(uploaded_file.getvalue()) / 1024  # KB
            st.markdown(f"""
            <div class="metric-card">
                <h4>💾 Size</h4>
                <p>{file_size:.1f} KB</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Prediction button
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("🤖 Analyzing image for ship detection..."):
                # Perform prediction
                binary_mask, prob_mask, max_confidence = inference_pipeline.predict(
                    image, threshold=confidence_threshold
                )
                
                # Get statistics
                stats = inference_pipeline.get_ship_statistics(binary_mask)
                
                # Display results
                st.subheader("🎯 Detection Results")
                
                # Metrics
                metric_col1, metric_col3, metric_col4 = st.columns(3)
                
                with metric_col1:
                    status_class = "success-metric" if stats['ships_detected'] else "metric-card"
                    status_text = "✅ DETECTED" if stats['ships_detected'] else "❌ NO SHIPS"
                    st.markdown(f"""
                    <div class="metric-card {status_class}">
                        <h4>🚢 Ship Status</h4>
                        <p><strong>{status_text}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col3:
                    confidence_class = "success-metric" if max_confidence > 0.8 else "warning-metric" if max_confidence > 0.5 else "metric-card"
                    st.markdown(f"""
                    <div class="metric-card {confidence_class}">
                        <h4>📊 Max Confidence</h4>
                        <p><strong>{max_confidence:.3f}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📐 Coverage</h4>
                        <p><strong>{stats['coverage_percentage']:.2f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualizations
                st.subheader("📊 Prediction Visualizations")
                
                vis_cols = []
                num_visualizations = 1 + int(show_probability) + int(show_overlay)
                
                if num_visualizations == 1:
                    vis_cols = [st.columns(1)[0]]
                elif num_visualizations == 2:
                    vis_cols = st.columns(2)
                else:
                    vis_cols = st.columns(3)
                
                col_idx = 0
                
                # Binary prediction
                with vis_cols[col_idx]:
                    st.subheader("🎯 Binary Prediction")
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(binary_mask, cmap='gray')
                    ax.set_title(f'Ship Detection (Threshold: {confidence_threshold})')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
                col_idx += 1
                
                # Probability map
                if show_probability and col_idx < len(vis_cols):
                    with vis_cols[col_idx]:
                        st.subheader("🌡️ Probability Map")
                        fig, ax = plt.subplots(figsize=(8, 8))
                        im = ax.imshow(prob_mask, cmap='viridis', vmin=0, vmax=1)
                        ax.set_title('Ship Probability Map')
                        ax.axis('off')
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        st.pyplot(fig)
                        plt.close()
                    col_idx += 1
                
                # Overlay visualization
                if show_overlay and col_idx < len(vis_cols):
                    with vis_cols[col_idx]:
                        st.subheader("🔍 Detection Overlay")
                        overlay_image = create_overlay_visualization(
                            image, prob_mask, binary_mask, confidence_threshold
                        )
                        st.image(overlay_image, caption="Ships highlighted in red", use_container_width=True)
                
                # Detailed statistics
                if stats['ships_detected']:
                    st.subheader("📈 Detailed Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"""
                        **Detection Summary:**
                        - Ship pixels: {stats['ship_pixels']:,}
                        - Total image pixels: {stats['total_pixels']:,}
                        - Ship coverage: {stats['coverage_percentage']:.2f}%
                        - Maximum confidence: {max_confidence:.3f}
                        """)
                    
                    with col2:
                        # Confidence distribution histogram
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.hist(prob_mask.flatten(), bins=50, alpha=0.7, color='blue')
                        ax.axvline(confidence_threshold, color='red', linestyle='--', 
                                  label=f'Threshold: {confidence_threshold}')
                        ax.set_xlabel('Confidence Score')
                        ax.set_ylabel('Pixel Count')
                        ax.set_title('Confidence Score Distribution')
                        ax.legend()
                        st.pyplot(fig)
                        plt.close()
                
                # Download results
                st.subheader("💾 Download Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Binary mask download
                    binary_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
                    buf = io.BytesIO()
                    binary_img.save(buf, format='PNG')
                    st.download_button(
                        label="📥 Download Binary Mask",
                        data=buf.getvalue(),
                        file_name="ship_detection_binary.png",
                        mime="image/png"
                    )
                
                with col2:
                    # Probability map download
                    prob_img = Image.fromarray((prob_mask * 255).astype(np.uint8))
                    buf = io.BytesIO()
                    prob_img.save(buf, format='PNG')
                    st.download_button(
                        label="📥 Download Probability Map",
                        data=buf.getvalue(),
                        file_name="ship_detection_probability.png",
                        mime="image/png"
                    )
                
                with col3:
                    # Statistics CSV
                    stats_df = pd.DataFrame([stats])
                    csv = stats_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Statistics",
                        data=csv,
                        file_name="ship_detection_stats.csv",
                        mime="text/csv"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚢 Maritime Ship Detection Dashboard | Powered by Deep Learning | Research Implementation</p>
        <p>Built with Streamlit • PyTorch • Computer Vision</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
