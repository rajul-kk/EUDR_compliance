"""M3: Training script for Siamese-DeepLabV3 change detection model."""

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from change_siamese_model import get_siamese_model
from preprocessing.change_dataset import ChangeDetectionDataset
from train_utils import seed_everything, split_dataset

IGNORE_INDEX = 255
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    dataset = ChangeDetectionDataset(
        t1_dir=args.t1_dir,
        t2_dir=args.t2_dir,
        mask_dir=args.mask_dir,
        label_backend=args.label_backend,
        histogram_match=True,
    )
    train_set, val_set = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)

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

    model = get_siamese_model().to(DEVICE)
    if torch.cuda.device_count() > 1:
        logger.info("Using DataParallel across %d GPUs", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    # Collect encoder params for warmup freeze (unwrap DataParallel if needed)
    _base = model.module if isinstance(model, torch.nn.DataParallel) else model
    _encoder_params = (
        list(_base.stem.parameters()) + list(_base.layer1.parameters()) +
        list(_base.layer2.parameters()) + list(_base.layer3.parameters()) +
        list(_base.layer4.parameters())
    )
    if args.warmup_epochs > 0:
        for p in _encoder_params:
            p.requires_grad_(False)
        logger.info("Encoder frozen for %d warmup epochs — only FPN+head training", args.warmup_epochs)

    # Upweight the rare forest-loss class
    weight = torch.tensor([0.3, 0.7]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_INDEX)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=_cuda)

    best_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    os.makedirs(os.path.dirname(args.output_model_path) or ".", exist_ok=True)

    logger.info("Starting Siamese training on %s for %d epochs", DEVICE, args.epochs)

    for epoch in range(args.epochs):
        if epoch == args.warmup_epochs and args.warmup_epochs > 0:
            for p in _encoder_params:
                p.requires_grad_(True)
            logger.info("Encoder unfrozen at epoch %d — end-to-end fine-tuning", epoch + 1)
        model.train()
        train_loss = 0.0

        for t1, t2, change in train_loader:
            t1, t2, change = t1.to(DEVICE), t2.to(DEVICE), change.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=_cuda):
                logits = model(t1, t2)["out"]
                loss = criterion(logits, change)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        model.eval()
        val_loss, val_f1 = 0.0, 0.0
        with torch.no_grad():
            for t1, t2, change in val_loader:
                t1, t2, change = t1.to(DEVICE), t2.to(DEVICE), change.to(DEVICE)
                with torch.autocast("cuda", enabled=_cuda):
                    logits = model(t1, t2)["out"]
                    loss = criterion(logits, change)
                val_loss += loss.item()
                val_f1 += _change_f1(logits, change)

        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_val_f1 = val_f1 / max(1, len(val_loader))

        scheduler.step(avg_val_f1)
        logger.info("Epoch [%d/%d] train_loss=%.4f val_loss=%.4f val_F1=%.4f lr=%.2e",
                    epoch + 1, args.epochs, avg_train_loss, avg_val_loss, avg_val_f1,
                    optimizer.param_groups[0]["lr"])

        _state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()

        if avg_val_f1 > best_f1:
            best_f1 = avg_val_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save({"state_dict": _state, "best_epoch": best_epoch, "best_val_f1": best_f1},
                       args.output_model_path)
            logger.info("New best (F1=%.4f) saved to %s", best_f1, args.output_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("Best epoch: %d | Best val F1: %.4f", best_epoch, best_f1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train M3 Siamese-DeepLabV3 change detection model.")
    parser.add_argument("--t1-dir", required=True, help="Baseline (t1) satellite image directory")
    parser.add_argument("--t2-dir", required=True, help="Current (t2) satellite image directory")
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--label-backend", default="hansen", choices=["hansen", "hybrid"])
    parser.add_argument("--output-model-path", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Freeze encoder for this many epochs before end-to-end fine-tuning")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
