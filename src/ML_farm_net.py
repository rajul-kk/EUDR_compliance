
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import os
import sys
import numpy as np

# Add parent directory and GEE_dynamic to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
gee_dir = os.path.join(parent_dir, 'GEE_dynamic')
if gee_dir not in sys.path:
    sys.path.append(gee_dir)

try:
    from preprocessing.dataset_loader import FarmSegmentationDataset
except ImportError:
    # Fallback if running from root
    sys.path.append(os.path.abspath('GEE_dynamic'))
    from preprocessing.dataset_loader import FarmSegmentationDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_deeplab_model(num_classes=4, in_channels=6):
    """
    Returns a DeepLabV3 model with ResNet50 backbone modified for N input channels.
    """
    # Load model with no pretrained weights (since we are changing input channels significantly)
    # Alternatively, load pretrained and modify first layer, initializing new weights average
    model = deeplabv3_resnet50(weights=None, num_classes=num_classes)
    
    # Modify the first convolutional layer of the backbone
    # Original: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    original_conv1 = model.backbone.conv1
    
    # Create new layer with in_channels
    new_conv1 = nn.Conv2d(
        in_channels, 
        original_conv1.out_channels, 
        kernel_size=original_conv1.kernel_size, 
        stride=original_conv1.stride, 
        padding=original_conv1.padding, 
        bias=original_conv1.bias
    )
    
    # Initialize weights (optional: copy RGB weights if pretrained)
    # For now, simple initialization
    nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
    
    # Replace
    model.backbone.conv1 = new_conv1
    
    return model

def train_model(raw_dir, mask_dir, output_model_path, epochs=10, batch_size=4, learning_rate=1e-4):
    """
    Trains the DeepLabV3 model.
    """
    print(f"Initializing Dataset from {raw_dir}...")
    dataset = FarmSegmentationDataset(raw_dir, mask_dir, cache_aligned_masks=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) # workers=0 for windows compat
    
    print(f"Dataset size: {len(dataset)}")
    
    model = get_deeplab_model().to(DEVICE)
    model.train()
    
    criterion = nn.CrossEntropyLoss(ignore_index=255) # ignore cloud masked pixels
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting training on {DEVICE} for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE) # (B, H, W) LongTensor
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(images)['out'] # Dictionary output for DeepLab {'out': tensor, 'aux': tensor}
            
            # Calculate Loss
            loss = criterion(outputs, masks)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch+1} Complete. Average Loss: {epoch_loss / len(dataloader):.4f}")
        
    # Save Model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    # Example Usage
    RAW_DIR = r'd:\Work\EUDR-compliance\data\raw_satellite\2020_baseline'
    MASK_DIR = r'd:\Work\EUDR-compliance\data\hybrid_masks'
    MODEL_PATH = r'd:\Work\EUDR-compliance\models\farm_deeplab.pth'
    
    # Ensure dirs exist before running (just a check)
    if os.path.exists(RAW_DIR) and os.path.exists(MASK_DIR):
        train_model(RAW_DIR, MASK_DIR, MODEL_PATH, epochs=2, batch_size=2)
    else:
        print("Data directories not found. Check paths.")
