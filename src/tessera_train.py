import argparse
import logging
import os
import random
import sys
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split

# Ensure src/ is on sys.path so sibling modules resolve when running from project root
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from tessera_backbone import TesseraSegmentationModel

current_dir = _src_dir
parent_dir = os.path.dirname(current_dir)
gee_dir = os.path.join(parent_dir, "GEE_dynamic")
if gee_dir not in sys.path:
    sys.path.append(gee_dir)

try:
    from preprocessing.dataset_loader import FarmSegmentationDataset
except ImportError:
    sys.path.append(os.path.abspath("GEE_dynamic"))
    from preprocessing.dataset_loader import FarmSegmentationDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, val_ratio: float, seed: int) -> Tuple[Subset, Subset]:
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split too large for dataset size.")

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def compute_miou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = 255) -> float:
    preds = torch.argmax(logits, dim=1)
    valid = targets != ignore_index

    ious = []
    for class_idx in range(num_classes):
        pred_mask = (preds == class_idx) & valid
        true_mask = (targets == class_idx) & valid

        intersection = torch.logical_and(pred_mask, true_mask).sum().float()
        union = torch.logical_or(pred_mask, true_mask).sum().float()

        if union > 0:
            ious.append((intersection / union).item())

    if not ious:
        return 0.0
    return float(np.mean(ious))


def train_tessera_head(
    raw_dir: str,
    mask_dir: str,
    output_model_path: str,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    val_ratio: float = 0.15,
    patience: int = 3,
    seed: int = 42,
    num_workers: int = 0,
) -> Dict[str, float]:
    seed_everything(seed)

    dataset = FarmSegmentationDataset(raw_dir, mask_dir, cache_aligned_masks=True)
    train_subset, val_subset = split_dataset(dataset, val_ratio=val_ratio, seed=seed)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = TesseraSegmentationModel(in_channels=6, num_classes=4, freeze_encoder=True).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    best_miou = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)["out"]
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_miou_running = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = model(images)["out"]
                loss = criterion(logits, masks)
                val_loss += loss.item()
                val_miou_running += compute_miou(logits, masks, num_classes=4)

        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_val_miou = val_miou_running / max(1, len(val_loader))

        logger.info(
            "Epoch [%d/%d] train_loss=%.4f val_loss=%.4f val_mIoU=%.4f",
            epoch + 1, epochs, avg_train_loss, avg_val_loss, avg_val_miou,
        )

        if avg_val_miou > best_miou:
            best_miou = avg_val_miou
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            model.save_checkpoint(
                output_model_path,
                extra={
                    "best_epoch": best_epoch,
                    "best_val_miou": best_miou,
                    "val_ratio": val_ratio,
                    "seed": seed,
                },
            )
            logger.info("Saved best checkpoint to %s", output_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

    return {
        "best_epoch": float(best_epoch),
        "best_val_miou": float(best_miou),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TESSERA head-only segmentation model.")
    parser.add_argument("--raw-dir", required=True, help="Path to raw Sentinel image directory.")
    parser.add_argument("--mask-dir", required=True, help="Path to hybrid mask directory.")
    parser.add_argument("--output-model-path", required=True, help="Path to save model checkpoint.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_tessera_head(
        raw_dir=args.raw_dir,
        mask_dir=args.mask_dir,
        output_model_path=args.output_model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_ratio=args.val_ratio,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    logger.info("Best epoch: %d | Best val mIoU: %.4f", int(metrics['best_epoch']), metrics['best_val_miou'])


if __name__ == "__main__":
    main()
