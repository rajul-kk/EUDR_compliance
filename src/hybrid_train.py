"""M6: Training script for DeepLabV3 + tessera-embed context hybrid."""

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from hybrid_model import get_hybrid_model
from train_utils import compute_miou, load_embedding, seed_everything, split_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HybridDataset(Dataset):
    """Single-image dataset for M6 hybrid training.

    Wraps FarmSegmentationDataset (single 2020 baseline image → 4-class mask)
    and attaches a per-farm GeoTESSERA embedding as a 128-dim context vector.
    Intentionally uses single-image segmentation, not change-detection pairs —
    M6 is a richer M1a, not a replacement for M3 (Siamese).
    """

    def __init__(self, raw_dir: str, mask_dir: str, embeddings_dir: str, training: bool = False) -> None:
        import re
        import rasterio

        _gee_dir = os.path.join(os.path.dirname(_src_dir), "GEE_dynamic")
        if _gee_dir not in sys.path:
            sys.path.append(_gee_dir)
        from preprocessing.dataset_loader import FarmSegmentationDataset

        self._base = FarmSegmentationDataset(raw_dir, mask_dir, cache_aligned_masks=True, training=training)
        self.embeddings_dir = embeddings_dir
        self._re = re.compile(r"((relation|way)_(\d+))")

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        image, mask = self._base[idx]
        img_path = self._base.image_paths[idx]

        # Locate embedding for this farm (use mean over tiles if multiple match)
        m = self._re.search(os.path.basename(img_path))
        farm_key = m.group(0) if m else ""

        embed_vec = self._find_embed(farm_key)
        return image, mask, embed_vec

    def _find_embed(self, farm_key: str) -> torch.Tensor:
        if farm_key and os.path.isdir(self.embeddings_dir):
            candidates = [f for f in os.listdir(self.embeddings_dir)
                          if farm_key in f and f.endswith(".npy")]
            if candidates:
                arrays = [load_embedding(os.path.join(self.embeddings_dir, c)) for c in candidates]
                # Average over spatial dims to get a single 128-dim context vector
                vec = np.mean([a.mean(axis=(1, 2)) for a in arrays], axis=0)
                return torch.from_numpy(vec.astype(np.float32))

        # Fallback: zero embedding (model learns without context for this sample)
        return torch.zeros(128, dtype=torch.float32)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    dataset = HybridDataset(
        raw_dir=args.raw_dir,
        mask_dir=args.mask_dir,
        embeddings_dir=args.embeddings_dir,
        training=True,
    )
    train_set, val_set = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)
    logger.info("Dataset: %d train, %d val", len(train_set), len(val_set))

    _cuda = torch.cuda.is_available()
    if _cuda:
        torch.backends.cudnn.benchmark = True
    _workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=_workers, pin_memory=_cuda,
                              persistent_workers=_workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=_workers, pin_memory=_cuda,
                            persistent_workers=_workers > 0)

    model = get_hybrid_model().to(DEVICE)
    if torch.cuda.device_count() > 1:
        logger.info("Using DataParallel across %d GPUs", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=_cuda)

    best_miou = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    os.makedirs(os.path.dirname(args.output_model_path) or ".", exist_ok=True)

    logger.info("Starting hybrid training on %s for %d epochs", DEVICE, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for images, masks, embeds in train_loader:
            images, masks, embeds = images.to(DEVICE), masks.to(DEVICE), embeds.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=_cuda):
                if isinstance(model, torch.nn.DataParallel):
                    logits = model.module.forward(images, embeds)["out"]
                else:
                    logits = model(images, embeds)["out"]
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        model.eval()
        val_loss, val_miou = 0.0, 0.0
        with torch.no_grad():
            for images, masks, embeds in val_loader:
                images, masks, embeds = images.to(DEVICE), masks.to(DEVICE), embeds.to(DEVICE)
                with torch.autocast("cuda", enabled=_cuda):
                    if isinstance(model, torch.nn.DataParallel):
                        logits = model.module.forward(images, embeds)["out"]
                    else:
                        logits = model(images, embeds)["out"]
                    loss = criterion(logits, masks)
                val_loss += loss.item()
                val_miou += compute_miou(logits, masks, num_classes=4)

        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_val_miou = val_miou / max(1, len(val_loader))

        logger.info("Epoch [%d/%d] train_loss=%.4f val_loss=%.4f val_mIoU=%.4f",
                    epoch + 1, args.epochs, avg_train_loss, avg_val_loss, avg_val_miou)

        _state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()

        if avg_val_miou > best_miou:
            best_miou = avg_val_miou
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save({"state_dict": _state, "best_epoch": best_epoch, "best_val_miou": best_miou},
                       args.output_model_path)
            logger.info("New best (mIoU=%.4f) saved to %s", best_miou, args.output_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("Best epoch: %d | Best val mIoU: %.4f", best_epoch, best_miou)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train M6 hybrid DeepLabV3 + embed context model.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--embeddings-dir", required=True, help="Directory containing farm-level .npy embeddings")
    parser.add_argument("--output-model-path", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
