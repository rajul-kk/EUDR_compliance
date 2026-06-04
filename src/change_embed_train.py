"""M5: Training script for tessera-embed change detection head."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from change_embed_model import get_change_embed_model
from train_utils import compute_miou, seed_everything, split_dataset


IGNORE_INDEX = 255
FOREST_CLASS = 1


class ChangeEmbedDataset(Dataset):
    """Pairs of (embed_t1, embed_t2) → element-wise difference → binary change mask.

    Expects embedding .npy files named consistently for both years, with matching
    mask .tif files in mask_dir.
    """

    def __init__(self, embeddings_dir_t1: str, embeddings_dir_t2: str, mask_dir: str,
                 year_t1: str = "2020", year_t2: str = "2024") -> None:
        self.pairs: list = []

        from train_utils import extract_farm_key, load_embedding

        self._load_embedding = load_embedding
        self._extract_key = extract_farm_key

        t1_files = {Path(f).stem: os.path.join(embeddings_dir_t1, f)
                    for f in os.listdir(embeddings_dir_t1) if f.endswith(".npy")}
        t2_files = {Path(f).stem: os.path.join(embeddings_dir_t2, f)
                    for f in os.listdir(embeddings_dir_t2) if f.endswith(".npy")}

        for stem, t1_path in t1_files.items():
            if stem not in t2_files:
                continue
            farm_key = self._extract_key(stem)
            if farm_key is None:
                continue

            mask_t1 = os.path.join(mask_dir, f"{farm_key}_{year_t1}_hybrid.tif")
            mask_t2 = os.path.join(mask_dir, f"{farm_key}_{year_t2}_hybrid.tif")
            if not os.path.exists(mask_t1) or not os.path.exists(mask_t2):
                continue

            self.pairs.append((t1_path, t2_files[stem], mask_t1, mask_t2))

        logger.info("ChangeEmbedDataset: %d pairs", len(self.pairs))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        t1_path, t2_path, mask_t1_path, mask_t2_path = self.pairs[idx]

        emb_t1 = self._load_embedding(t1_path)
        emb_t2 = self._load_embedding(t2_path)
        diff = emb_t1 - emb_t2  # (128, H, W)

        with rasterio.open(mask_t1_path) as s:
            mask_t1 = s.read(1).astype(np.int64)
        with rasterio.open(mask_t2_path) as s:
            mask_t2 = s.read(1).astype(np.int64)

        h = min(diff.shape[1], mask_t1.shape[0], mask_t2.shape[0])
        w = min(diff.shape[2], mask_t1.shape[1], mask_t2.shape[1])
        diff = diff[:, :h, :w]
        mask_t1, mask_t2 = mask_t1[:h, :w], mask_t2[:h, :w]

        change = np.zeros((h, w), dtype=np.int64)
        change[(mask_t1 == FOREST_CLASS) & (mask_t2 != FOREST_CLASS) & (mask_t2 != IGNORE_INDEX)] = 1
        change[(mask_t1 == IGNORE_INDEX) | (mask_t2 == IGNORE_INDEX)] = IGNORE_INDEX

        return torch.from_numpy(diff), torch.from_numpy(change)


def _change_f1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    valid = targets != IGNORE_INDEX
    tp = ((preds == 1) & (targets == 1) & valid).sum().float()
    fp = ((preds == 1) & (targets == 0) & valid).sum().float()
    fn = ((preds == 0) & (targets == 1) & valid).sum().float()
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    return (2 * p * r / (p + r + 1e-8)).item()


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    DEVICE = torch.device(args.device)

    dataset = ChangeEmbedDataset(
        embeddings_dir_t1=args.embeddings_dir_t1,
        embeddings_dir_t2=args.embeddings_dir_t2,
        mask_dir=args.mask_dir,
        year_t1=args.year_t1,
        year_t2=args.year_t2,
    )
    train_set, val_set = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = get_change_embed_model(dropout_p=0.2).to(DEVICE)
    weight = torch.tensor([0.3, 0.7]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_INDEX)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    os.makedirs(os.path.dirname(args.output_model_path) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for diff, change in train_loader:
            diff, change = diff.to(DEVICE), change.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(diff)["out"], change)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, val_f1 = 0.0, 0.0
        with torch.no_grad():
            for diff, change in val_loader:
                diff, change = diff.to(DEVICE), change.to(DEVICE)
                logits = model(diff)["out"]
                val_loss += criterion(logits, change).item()
                val_f1 += _change_f1(logits, change)

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_val_f1 = val_f1 / max(1, len(val_loader))

        logger.info("Epoch [%d/%d] train_loss=%.4f val_loss=%.4f val_F1=%.4f",
                    epoch + 1, args.epochs, avg_train_loss, avg_val_loss, avg_val_f1)

        if avg_val_f1 > best_f1:
            best_f1 = avg_val_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save({"state_dict": model.state_dict(), "best_epoch": best_epoch,
                        "best_val_f1": best_f1}, args.output_model_path)
            logger.info("New best (F1=%.4f) saved to %s", best_f1, args.output_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("Best epoch: %d | Best val F1: %.4f", best_epoch, best_f1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train M5 tessera-embed change detection head.")
    parser.add_argument("--embeddings-dir-t1", required=True, help="Embeddings directory for t1 (baseline year)")
    parser.add_argument("--embeddings-dir-t2", required=True, help="Embeddings directory for t2 (current year)")
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-model-path", required=True)
    parser.add_argument("--year-t1", default="2020")
    parser.add_argument("--year-t2", default="2024")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
