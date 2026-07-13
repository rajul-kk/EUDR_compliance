
import logging

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)
import argparse
import os
import sys

import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.segmentation import deeplabv3_resnet50

# Add parent directory and GEE_dynamic to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
gee_dir = os.path.join(parent_dir, 'GEE_dynamic')
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if gee_dir not in sys.path:
    sys.path.append(gee_dir)

from train_utils import compute_miou, seed_everything, split_dataset

try:
    from preprocessing.dataset_loader import FarmSegmentationDataset
except ImportError:
    _gee_dir = os.path.join(parent_dir, "GEE_dynamic")
    if _gee_dir not in sys.path:
        sys.path.insert(0, _gee_dir)
    from preprocessing.dataset_loader import FarmSegmentationDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_deeplab_model(num_classes=4, in_channels=7):
    model = deeplabv3_resnet50(weights=None, num_classes=num_classes)
    pretrained = resnet50(weights=ResNet50_Weights.DEFAULT)
    new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
    with torch.no_grad():
        new_conv1.weight[:, :3] = pretrained.conv1.weight
    model.backbone.conv1 = new_conv1
    return model

def train_model(raw_dir, mask_dir, output_model_path, epochs=10, batch_size=4, learning_rate=1e-4,
                exclude_crops=None, exclude_regions=None, val_ratio=0.15, patience=5, seed=42):
    """Trains the DeepLabV3 model with val split, best-model and per-epoch checkpoints."""
    seed_everything(seed)

    logger.info("Initializing dataset from %s", raw_dir)
    _common = dict(cache_aligned_masks=True, exclude_crops=exclude_crops, exclude_regions=exclude_regions)
    train_dataset = FarmSegmentationDataset(raw_dir, mask_dir, training=True,  **_common)
    val_dataset   = FarmSegmentationDataset(raw_dir, mask_dir, training=False, **_common)
    train_subset, _ = split_dataset(train_dataset, val_ratio=val_ratio, seed=seed)
    _, val_subset   = split_dataset(val_dataset,   val_ratio=val_ratio, seed=seed)
    logger.info("Dataset: %d train, %d val", len(train_subset), len(val_subset))

    _cuda = torch.cuda.is_available()
    if _cuda:
        torch.backends.cudnn.benchmark = True
    _workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=_workers, pin_memory=_cuda,
                              persistent_workers=_workers > 0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=_workers, pin_memory=_cuda,
                            persistent_workers=_workers > 0)

    model = get_deeplab_model().to(DEVICE)
    if torch.cuda.device_count() > 1:
        logger.info("Using DataParallel across %d GPUs", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=_cuda)

    os.makedirs(os.path.dirname(output_model_path) or ".", exist_ok=True)

    best_miou = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    logger.info("Starting training on %s for %d epochs", DEVICE, epochs)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for i, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=_cuda):
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if i % 5 == 0:
                logger.info("Epoch [%d/%d] Step [%d/%d] loss=%.4f",
                            epoch + 1, epochs, i + 1, len(train_loader), loss.item())

        avg_train_loss = train_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_miou_sum = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                with torch.autocast("cuda", enabled=_cuda):
                    outputs = model(images)['out']
                    loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_miou_sum += compute_miou(outputs, masks, num_classes=4)

        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_val_miou = val_miou_sum / max(1, len(val_loader))

        logger.info("Epoch [%d/%d] train_loss=%.4f val_loss=%.4f val_mIoU=%.4f",
                    epoch + 1, epochs, avg_train_loss, avg_val_loss, avg_val_miou)

        _state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()

        if avg_val_miou > best_miou:
            best_miou = avg_val_miou
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(_state, output_model_path)
            logger.info("New best (mIoU=%.4f) saved to %s", best_miou, output_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    return {"best_epoch": float(best_epoch), "best_val_miou": float(best_miou)}


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLabV3 baseline model.")

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_raw_dir = os.path.join(_root, 'data', 'raw_satellite', '2020_baseline')
    default_mask_dir = os.path.join(_root, 'data', 'hybrid_masks')
    default_model_path = os.path.join(_root, 'models', 'farm_deeplab.pth')

    parser.add_argument('--raw-dir', default=default_raw_dir, help='Directory with baseline (2020) images')
    parser.add_argument('--mask-dir', default=default_mask_dir, help='Directory with hybrid masks')
    parser.add_argument('--output-model-path', default=default_model_path, help='Output model .pth path')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Adam learning rate')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Fraction of data held out for validation')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.raw_dir) or not os.path.exists(args.mask_dir):
        raise FileNotFoundError(
            f"Training inputs not found. raw_dir={args.raw_dir}, mask_dir={args.mask_dir}"
        )

    metrics = train_model(
        raw_dir=args.raw_dir,
        mask_dir=args.mask_dir,
        output_model_path=args.output_model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        exclude_crops=None,
        exclude_regions=None,
        val_ratio=args.val_ratio,
        patience=args.patience,
        seed=args.seed,
    )
    logger.info("Best epoch: %d | Best val mIoU: %.4f", int(metrics['best_epoch']), metrics['best_val_miou'])
