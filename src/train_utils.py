"""Shared training utilities used by tessera_train and tessera_embed_train."""

import logging
import random
import re
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[Subset, Subset]:
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split too large for dataset size.")
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


@torch.no_grad()
def compute_miou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    preds = torch.argmax(logits, dim=1)
    valid = targets != ignore_index
    ious = []
    for c in range(num_classes):
        pred_mask = (preds == c) & valid
        true_mask = (targets == c) & valid
        inter = torch.logical_and(pred_mask, true_mask).sum().float()
        union = torch.logical_or(pred_mask, true_mask).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    return float(np.mean(ious)) if ious else 0.0


def extract_farm_key(name: str) -> Optional[str]:
    match = re.match(r"^(relation|way)_\d+", name)
    return match.group(0) if match else None


def mc_dropout_predict(
    model: "torch.nn.Module",
    x: "torch.Tensor",
    n_passes: int = 20,
    extra_inputs: Optional[tuple] = None,
) -> tuple:
    """Monte Carlo Dropout inference: returns (mean_probs, uncertainty_entropy).

    Activates dropout by calling model.train(), then runs n_passes forward passes
    under torch.no_grad(). Returns pixel-wise mean softmax probabilities and
    Shannon entropy as an uncertainty map.

    Args:
        model:        Any model with Dropout2d layers. Must already be on the right device.
        x:            Input tensor (B, C, H, W).
        n_passes:     Number of stochastic forward passes (default 20; use 10 for production).
        extra_inputs: Optional tuple of additional positional args (e.g. embed tensor for M6).

    Returns:
        mean_probs:  (B, num_classes, H, W) float32 — averaged softmax over passes.
        uncertainty: (B, H, W) float32 — pixel-wise Shannon entropy (nats).
    """
    import torch
    import torch.nn.functional as F

    model.train()  # activates Dropout2d
    probs_list = []

    with torch.no_grad():
        for _ in range(n_passes):
            if extra_inputs:
                logits = model(x, *extra_inputs)["out"]
            else:
                logits = model(x)["out"]
            probs_list.append(F.softmax(logits, dim=1))

    mean_probs = torch.stack(probs_list, dim=0).mean(dim=0)  # (B, C, H, W)
    # Shannon entropy: -sum(p * log(p + eps))
    uncertainty = -(mean_probs * (mean_probs + 1e-8).log()).sum(dim=1)  # (B, H, W)
    model.eval()
    return mean_probs, uncertainty


def load_embedding(path: str, scales_path: Optional[str] = None) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D embedding array, got shape {arr.shape} for {path}")

    if scales_path is not None:
        scales = np.load(scales_path)
        if scales.ndim == 2:
            scales = np.expand_dims(scales, axis=-1)
        if arr.shape[:2] != scales.shape[:2]:
            raise ValueError(
                f"Embedding/scales spatial shape mismatch: {arr.shape[:2]} vs {scales.shape[:2]} "
                f"for {path} and {scales_path}"
            )
        arr = arr.astype(np.float32) * scales.astype(np.float32)

    if arr.shape[-1] == 128:
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.shape[0] == 128:
        pass
    else:
        raise ValueError(f"Cannot infer embedding channel axis for shape {arr.shape} ({path})")

    return arr.astype(np.float32)
