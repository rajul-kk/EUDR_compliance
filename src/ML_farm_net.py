
import logging
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import argparse
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

def train_model(raw_dir, mask_dir, output_model_path, epochs=10, batch_size=4, learning_rate=1e-4,
                exclude_crops=None, exclude_regions=None):
    """
    Trains the DeepLabV3 model.
    """
    logger.info("Initializing dataset from %s", raw_dir)
    dataset = FarmSegmentationDataset(
        raw_dir, mask_dir, 
        cache_aligned_masks=True,
        exclude_crops=exclude_crops,
        exclude_regions=exclude_regions
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) # workers=0 for windows compat
    
    logger.info("Dataset size: %d", len(dataset))
    
    model = get_deeplab_model().to(DEVICE)
    model.train()
    
    criterion = nn.CrossEntropyLoss(ignore_index=255) # ignore cloud masked pixels
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    logger.info("Starting training on %s for %d epochs", DEVICE, epochs)
    
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
                logger.info("Epoch [%d/%d] Step [%d/%d] loss=%.4f", epoch+1, epochs, i+1, len(dataloader), loss.item())
                
        logger.info("Epoch %d complete avg_loss=%.4f", epoch+1, epoch_loss / len(dataloader))
        
    # Save Model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(model.state_dict(), output_model_path)
    logger.info("Model saved to %s", output_model_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLabV3 baseline model.")

    default_raw_dir = os.path.join(project_root, 'data', 'raw_satellite', '2020_baseline')
    default_mask_dir = os.path.join(project_root, 'data', 'hybrid_masks')
    default_model_path = os.path.join(project_root, 'models', 'farm_deeplab.pth')

    parser.add_argument('--raw-dir', default=default_raw_dir, help='Directory with baseline (2020) images')
    parser.add_argument('--mask-dir', default=default_mask_dir, help='Directory with hybrid masks')
    parser.add_argument('--output-model-path', default=default_model_path, help='Output model .pth path')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Adam learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.raw_dir) or not os.path.exists(args.mask_dir):
        raise FileNotFoundError(
            f"Training inputs not found. raw_dir={args.raw_dir}, mask_dir={args.mask_dir}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_model(
        raw_dir=args.raw_dir,
        mask_dir=args.mask_dir,
        output_model_path=args.output_model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        exclude_crops=None,
        exclude_regions=None,
    )
