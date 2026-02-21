
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
import rasterio
import numpy as np
import os
import sys
import glob
from pathlib import Path

# Add parent directory and GEE_dynamic to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
gee_dir = os.path.join(parent_dir, 'GEE_dynamic')
if gee_dir not in sys.path:
    sys.path.append(gee_dir)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_deeplab_model(num_classes=4, in_channels=5):
    """
    Returns a DeepLabV3 model with ResNet50 backbone modified for N input channels.
    Must match the architecture used in training.
    """
    model = deeplabv3_resnet50(weights=None, num_classes=num_classes)
    
    # Modify first conv layer to accept 5 channels (R, G, B, NIR, SCL)
    original_conv1 = model.backbone.conv1
    new_conv1 = nn.Conv2d(
        in_channels, 
        original_conv1.out_channels, 
        kernel_size=original_conv1.kernel_size, 
        stride=original_conv1.stride, 
        padding=original_conv1.padding, 
        bias=original_conv1.bias
    )
    nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
    model.backbone.conv1 = new_conv1
    
    return model

def load_model(model_path, num_classes=4, in_channels=5):
    """
    Load a trained DeepLabV3 model from disk.
    
    Args:
        model_path: Path to .pth file
        num_classes: Number of output classes
        in_channels: Number of input channels
    
    Returns:
        model: Loaded PyTorch model in eval mode
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = get_deeplab_model(num_classes, in_channels)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print(f"✅ Model loaded from {model_path}")
    return model

def load_image(image_path):
    """
    Load a Sentinel-2 GeoTIFF image and prepare for inference.
    
    Args:
        image_path: Path to .tiff/.tif file
    
    Returns:
        image_tensor: Torch tensor (1, C, H, W)
        profile: Rasterio profile for saving output
    """
    with rasterio.open(image_path) as src:
        image = src.read()  # (C, H, W)
        profile = src.profile
        
        # Validate band count
        if image.shape[0] < 5:
            raise ValueError(f"Expected at least 5 bands, got {image.shape[0]} in {os.path.basename(image_path)}")
        
        # Convert to float32
        image = image.astype(np.float32)
        
        # Add batch dimension and convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, C, H, W)
        
    return image_tensor, profile

def predict_single_image(model, image_path, output_path=None):
    """
    Run inference on a single image and optionally save the result.
    
    Args:
        model: Trained PyTorch model
        image_path: Path to input image
        output_path: Optional path to save predicted mask
    
    Returns:
        prediction: Numpy array (H, W) with class labels
    """
    # Load image
    image_tensor, profile = load_image(image_path)
    image_tensor = image_tensor.to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)['out']  # (1, num_classes, H, W)
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # (H, W)
    
    # Save if output path provided
    if output_path:
        # Update profile for single-band output
        output_profile = profile.copy()
        output_profile.update({
            'count': 1,
            'dtype': rasterio.uint8,
            'compress': 'lzw'
        })
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(prediction.astype(rasterio.uint8), 1)
        
        print(f"✅ Saved prediction to {output_path}")
    
    return prediction

def batch_inference(model_path, input_dir, output_dir, file_pattern="*.tiff"):
    """
    Run inference on all images in a directory.
    
    Args:
        model_path: Path to trained model
        input_dir: Directory containing input images
        output_dir: Directory to save predictions
        file_pattern: Glob pattern for input files
    
    Returns:
        results: Dictionary mapping input paths to output paths
    """
    # Load model once
    model = load_model(model_path)
    
    # Find all matching files
    image_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not image_files:
        print(f"⚠️ No images found matching {file_pattern} in {input_dir}")
        return {}
    
    print(f"🚀 Running inference on {len(image_files)} images...")
    
    results = {}
    for i, image_path in enumerate(image_files, 1):
        try:
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_predicted.tif")
            
            # Run inference
            print(f"[{i}/{len(image_files)}] Processing {base_name}...")
            predict_single_image(model, image_path, output_path)
            
            results[image_path] = output_path
            
        except Exception as e:
            print(f"❌ Failed to process {os.path.basename(image_path)}: {e}")
            continue
    
    print(f"\n✅ Inference complete. Processed {len(results)}/{len(image_files)} images.")
    return results

if __name__ == "__main__":
    # Example usage
    MODEL_PATH = r'd:\Work\EUDR-compliance\models\farm_deeplab.pth'
    INPUT_DIR = r'd:\Work\EUDR-compliance\data\raw_satellite\2024_current'
    OUTPUT_DIR = r'd:\Work\EUDR-compliance\data\predictions_2024'
    
    # Run batch inference
    batch_inference(MODEL_PATH, INPUT_DIR, OUTPUT_DIR)
