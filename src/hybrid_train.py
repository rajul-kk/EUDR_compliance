"""M6: Training script for DeepLabV3 + tessera-embed context hybrid."""

import argparse
import logging
import os
import re
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
from preprocessing.change_dataset import load_image, load_mask
from train_utils import compute_miou, load_embedding, seed_everything, split_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_FARM_RE = re.compile(r"(.+?)_(2020|2024)(?:_|\.tiff?)")
_KEY_RE = re.compile(r"((relation|way)_\d+)")


class HybridDataset(Dataset):
    """Single-image dataset for M6 hybrid training using Hansen GFC labels.

    Loads the 2020 baseline Sentinel-2 composite, the corresponding Hansen GFC
    label (``{farm_key}_hansen_label.tif``, values 0=non-forest/1=forest-2020/2=post-loss),
    and a GeoTESSERA regional embedding as a 128-dim context vector.

    Intentionally single-image (not paired) — M6 is a richer M1a: it predicts
    Hansen land-cover class from the 2020 image + landscape context, not change
    detection. For change detection use M3 (change_siamese_train.py).
    """

    def __init__(self, t1_dir: str, mask_dir: str, embeddings_dir: str, training: bool = False) -> None:
        self._training = training
        self.embeddings_dir = embeddings_dir
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

        logger.info("HybridDataset: %d samples (Hansen labels)", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import random
        from PIL import Image as PILImage

        img_path, label_path = self.samples[idx]
        img, _ = load_image(img_path)     # (7, H, W) float32
        label = load_mask(label_path)     # raw Hansen values 0/1/2

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

        km = _KEY_RE.search(os.path.basename(img_path))
        embed_vec = self._find_embed(km.group(0) if km else "")
        return torch.from_numpy(img), torch.from_numpy(label), embed_vec

    def _find_embed(self, farm_key: str) -> torch.Tensor:
        if farm_key and os.path.isdir(self.embeddings_dir):
            candidates = [f for f in os.listdir(self.embeddings_dir)
                          if farm_key in f and f.endswith(".npy")]
            if candidates:
                arrays = [load_embedding(os.path.join(self.embeddings_dir, c)) for c in candidates]
                vec = np.mean([a.mean(axis=(1, 2)) for a in arrays], axis=0)
                return torch.from_numpy(vec.astype(np.float32))
        return torch.zeros(128, dtype=torch.float32)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    dataset = HybridDataset(
        t1_dir=args.t1_dir,
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=_cuda)

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
                    logits = model(images, embeds)["out"]
                    loss = criterion(logits, masks)
                val_loss += loss.item()
                val_miou += compute_miou(logits, masks, num_classes=3)

        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_val_miou = val_miou / max(1, len(val_loader))

        scheduler.step(avg_val_miou)
        logger.info("Epoch [%d/%d] train_loss=%.4f val_loss=%.4f val_mIoU=%.4f lr=%.2e",
                    epoch + 1, args.epochs, avg_train_loss, avg_val_loss, avg_val_miou,
                    optimizer.param_groups[0]["lr"])

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
    parser.add_argument("--t1-dir", required=True, help="2020 baseline Sentinel-2 image directory")
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
