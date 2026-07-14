"""M1a: Single-image DeepLabV3 baseline trained on Hansen GFC labels.

Predicts land-cover class per pixel from a single Sentinel-2 composite.
Labels: 0 = non-forest, 1 = forest-2020, 2 = post-EUDR loss.

This is a single-image model — it answers "what is this pixel?"
To detect deforestation it must be run twice (2020 + 2024) and the two
output masks compared post-hoc. Contrast with M3 (Siamese), which takes
both images in one forward pass and answers "did this pixel change?"
"""

import argparse
import logging
import os
import random
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.segmentation import deeplabv3_resnet50

logger = logging.getLogger(__name__)

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from preprocessing.change_dataset import load_image, load_mask
from train_utils import compute_miou, seed_everything, split_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3  # Hansen: 0=non-forest, 1=forest-2020, 2=post-EUDR-loss

_FARM_RE = re.compile(r"(.+?)_(2020|2024)(?:_|\.tiff?)")


class HansenDataset(Dataset):
    """Single-image dataset pairing 2020 Sentinel-2 composites with Hansen GFC labels.

    Each sample is (image, label) where:
      image — (7, H, W) float32 Sentinel-2 composite (R/G/B/NIR/SCL/NDVI/NDWI)
      label — (H, W) int64 with values 0=non-forest, 1=forest-2020, 2=post-EUDR-loss
    """

    def __init__(self, t1_dir: str, mask_dir: str, training: bool = False) -> None:
        self._training = training
        self.samples: list = []

        for f in sorted(os.listdir(t1_dir)):
            if not f.endswith((".tif", ".tiff")):
                continue
            m = _FARM_RE.match(f)
            if not m:
                continue
            label_path = os.path.join(mask_dir, f"{m.group(1)}_hansen_label.tif")
            if os.path.exists(label_path):
                self.samples.append((os.path.join(t1_dir, f), label_path))

        logger.info("HansenDataset: %d samples (training=%s)", len(self.samples), training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image as PILImage

        img_path, label_path = self.samples[idx]
        img, _ = load_image(img_path)    # (7, H, W) float32
        label = load_mask(label_path)    # (H, W) int64, values 0/1/2

        h, w = img.shape[1], img.shape[2]
        if label.shape != (h, w):
            label = np.array(
                PILImage.fromarray(label.astype(np.uint8)).resize((w, h), PILImage.NEAREST),
                dtype=np.int64,
            )

        if self._training:
            if random.random() < 0.5:
                img = img[:, :, ::-1].copy(); label = label[:, ::-1].copy()
            if random.random() < 0.5:
                img = img[:, ::-1, :].copy(); label = label[::-1, :].copy()
            k = random.randint(0, 3)
            if k > 0:
                img = np.rot90(img, k=k, axes=(1, 2)).copy()
                label = np.rot90(label, k=k, axes=(0, 1)).copy()
            factor = random.uniform(0.9, 1.1)
            img[[0, 1, 2, 3, 5, 6]] = np.clip(img[[0, 1, 2, 3, 5, 6]] * factor, 0, None)

        return torch.from_numpy(img), torch.from_numpy(label)


def get_deeplab_model(num_classes: int = NUM_CLASSES, in_channels: int = 7) -> nn.Module:
    model = deeplabv3_resnet50(weights=None, num_classes=num_classes)
    pretrained = resnet50(weights=ResNet50_Weights.DEFAULT)
    new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
    with torch.no_grad():
        new_conv1.weight[:, :3] = pretrained.conv1.weight
    model.backbone.conv1 = new_conv1
    return model


def train_model(raw_dir, mask_dir, output_model_path, epochs=15, batch_size=8,
                learning_rate=1e-4, val_ratio=0.15, patience=5, seed=42):
    seed_everything(seed)

    logger.info("Initializing HansenDataset from %s", raw_dir)
    train_dataset = HansenDataset(raw_dir, mask_dir, training=True)
    val_dataset   = HansenDataset(raw_dir, mask_dir, training=False)
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
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False,
                              num_workers=_workers, pin_memory=_cuda,
                              persistent_workers=_workers > 0)

    model = get_deeplab_model().to(DEVICE)
    if torch.cuda.device_count() > 1:
        logger.info("Using DataParallel across %d GPUs", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=_cuda)

    os.makedirs(os.path.dirname(output_model_path) or ".", exist_ok=True)

    best_miou, best_epoch, epochs_without_improvement = -1.0, -1, 0

    logger.info("Starting training on %s for %d epochs", DEVICE, epochs)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=_cuda):
                loss = criterion(model(images)["out"], masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            if i % 5 == 0:
                logger.info("Epoch [%d/%d] Step [%d/%d] loss=%.4f",
                            epoch + 1, epochs, i + 1, len(train_loader), loss.item())

        avg_train_loss = train_loss / max(1, len(train_loader))

        model.eval()
        val_loss, val_miou_sum = 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                with torch.autocast("cuda", enabled=_cuda):
                    logits = model(images)["out"]
                    val_loss += criterion(logits, masks).item()
                val_miou_sum += compute_miou(logits, masks, num_classes=NUM_CLASSES)

        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_val_miou = val_miou_sum / max(1, len(val_loader))

        scheduler.step(avg_val_miou)
        logger.info("Epoch [%d/%d] train_loss=%.4f val_loss=%.4f val_mIoU=%.4f lr=%.2e",
                    epoch + 1, epochs, avg_train_loss, avg_val_loss, avg_val_miou,
                    optimizer.param_groups[0]["lr"])

        _state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        if avg_val_miou > best_miou:
            best_miou, best_epoch, epochs_without_improvement = avg_val_miou, epoch + 1, 0
            torch.save(_state, output_model_path)
            logger.info("New best (mIoU=%.4f) saved to %s", best_miou, output_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    return {"best_epoch": float(best_epoch), "best_val_miou": float(best_miou)}


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLabV3 baseline (M1a) on Hansen GFC labels.")
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--raw-dir",  default=os.path.join(_root, "data", "raw_satellite", "2020_baseline"))
    parser.add_argument("--mask-dir", default=os.path.join(_root, "data", "hansen_labels"))
    parser.add_argument("--output-model-path", default=os.path.join(_root, "models", "farm_deeplab.pth"))
    parser.add_argument("--epochs",        type=int,   default=15)
    parser.add_argument("--batch-size",    type=int,   default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-ratio",     type=float, default=0.15)
    parser.add_argument("--patience",      type=int,   default=5)
    parser.add_argument("--seed",          type=int,   default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.raw_dir) or not os.path.exists(args.mask_dir):
        raise FileNotFoundError(
            f"Training inputs not found. raw_dir={args.raw_dir}, mask_dir={args.mask_dir}"
        )
    metrics = train_model(
        raw_dir=args.raw_dir, mask_dir=args.mask_dir,
        output_model_path=args.output_model_path,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, val_ratio=args.val_ratio,
        patience=args.patience, seed=args.seed,
    )
    logger.info("Best epoch: %d | Best val mIoU: %.4f",
                int(metrics["best_epoch"]), metrics["best_val_miou"])
