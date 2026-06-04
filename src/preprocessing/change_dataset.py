"""Paired change-detection dataset.

Returns (image_t1, image_t2, change_mask) where change_mask marks pixels that
were forest in t1 and are no longer forest in t2 (EUDR relevant forest loss).
"""

import logging
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from preprocessing.histogram_match import match_histogram
from train_utils import seed_everything, split_dataset  # noqa: F401 — re-exported for callers

logger = logging.getLogger(__name__)

CLOUD_SCL_CLASSES = {0, 1, 3, 8, 9, 10}
FOREST_CLASS = 1
IGNORE_INDEX = 255


def _load_image(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (image float32 (C,H,W), scl uint8 (H,W))."""
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)

    if img.shape[0] < 5:
        raise ValueError(f"Expected ≥5 bands, got {img.shape[0]} in {path}")

    scl = img[4].astype(np.uint8)
    red, nir = img[0], img[3]
    ndvi = (nir - red) / (nir + red + 1e-8)
    img = np.concatenate([img, np.expand_dims(ndvi, 0)], axis=0)  # (6, H, W)
    return img, scl


def _load_mask(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.int64)


def _cloud_ignore(mask: np.ndarray, scl: np.ndarray) -> np.ndarray:
    cloud = np.isin(scl, list(CLOUD_SCL_CLASSES))
    out = mask.copy()
    out[cloud] = IGNORE_INDEX
    return out


class ChangeDetectionDataset(Dataset):
    """Paired (t1, t2) Sentinel-2 dataset with binary forest-loss change masks.

    Args:
        t1_dir:    Directory of baseline (e.g. 2020) composites.
        t2_dir:    Directory of current (e.g. 2024) composites.
        mask_dir:  Directory of hybrid segmentation masks.
        histogram_match: If True, normalise t2 histogram to match t1 before returning.
    """

    def __init__(
        self,
        t1_dir: str,
        t2_dir: str,
        mask_dir: str,
        histogram_match: bool = True,
    ) -> None:
        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.mask_dir = mask_dir
        self.histogram_match = histogram_match

        self.pairs: List[Tuple[str, str, str, str]] = []  # (t1_img, t2_img, mask_t1, mask_t2)
        self._build_pairs()
        logger.info("ChangeDetectionDataset: %d paired samples", len(self.pairs))

    def _build_pairs(self) -> None:
        t1_files = {f for f in os.listdir(self.t1_dir) if f.endswith((".tif", ".tiff"))}
        t2_files = {f for f in os.listdir(self.t2_dir) if f.endswith((".tif", ".tiff"))}

        for f in sorted(t1_files):
            m = re.match(r"((relation|way)_(\d+))_(\d{4})_.*\.tiff?", f)
            if not m:
                continue
            farm_key, _, _, year_t1 = m.group(1), m.group(2), m.group(3), m.group(4)

            # Find a matching t2 file for the same farm
            t2_match = next(
                (g for g in t2_files if g.startswith(farm_key + "_") and g != f), None
            )
            if t2_match is None:
                continue

            mt2 = re.match(r"(relation|way)_\d+_(\d{4})_.*\.tiff?", t2_match)
            year_t2 = mt2.group(2) if mt2 else "2024"

            mask_t1 = os.path.join(self.mask_dir, f"{farm_key}_{year_t1}_hybrid.tif")
            mask_t2 = os.path.join(self.mask_dir, f"{farm_key}_{year_t2}_hybrid.tif")

            if not os.path.exists(mask_t1) or not os.path.exists(mask_t2):
                continue

            self.pairs.append((
                os.path.join(self.t1_dir, f),
                os.path.join(self.t2_dir, t2_match),
                mask_t1,
                mask_t2,
            ))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        t1_path, t2_path, mask_t1_path, mask_t2_path = self.pairs[idx]

        img_t1, scl_t1 = _load_image(t1_path)
        img_t2, scl_t2 = _load_image(t2_path)

        if self.histogram_match:
            img_t2 = match_histogram(img_t2, img_t1)

        # Crop to common spatial extent
        h = min(img_t1.shape[1], img_t2.shape[1])
        w = min(img_t1.shape[2], img_t2.shape[2])
        img_t1, img_t2 = img_t1[:, :h, :w], img_t2[:, :h, :w]
        scl_t1, scl_t2 = scl_t1[:h, :w], scl_t2[:h, :w]

        mask_t1 = _cloud_ignore(_load_mask(mask_t1_path)[:h, :w], scl_t1)
        mask_t2 = _cloud_ignore(_load_mask(mask_t2_path)[:h, :w], scl_t2)

        # Binary change mask: 1 = forest loss, 0 = no change; 255 where either year is cloudy
        change = np.zeros((h, w), dtype=np.int64)
        change[(mask_t1 == FOREST_CLASS) & (mask_t2 != FOREST_CLASS) & (mask_t2 != IGNORE_INDEX)] = 1
        change[(mask_t1 == IGNORE_INDEX) | (mask_t2 == IGNORE_INDEX)] = IGNORE_INDEX

        return (
            torch.from_numpy(img_t1),
            torch.from_numpy(img_t2),
            torch.from_numpy(change),
        )
