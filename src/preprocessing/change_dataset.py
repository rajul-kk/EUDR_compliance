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

try:
    from osgeo import gdal as _gdal
    _gdal.PushErrorHandler("CPLQuietErrorHandler")
except ImportError:
    pass

_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from preprocessing.histogram_match import match_histogram
from train_utils import seed_everything, split_dataset  # noqa: F401 — re-exported for callers

logger = logging.getLogger(__name__)

CLOUD_SCL_CLASSES = {0, 1, 3, 8, 9, 10}
FOREST_CLASS = 1
IGNORE_INDEX = 255

# Hansen GFC label values (generate_labels.py)
HANSEN_FOREST_2020 = 1    # was forest as of 31 Dec 2020
HANSEN_POST_EUDR_LOSS = 2  # loss detected 2021-2023


_OPTICAL_BANDS = [0, 1, 2, 3]  # R, G, B, NIR — clip these; SCL is categorical, NDVI/NDWI are derived


def _percentile_clip(img: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Clip each optical band to its own [low, high] percentile.

    Removes outliers from cloud edges and sensor saturation before histogram
    matching so the distribution match is not pulled by extreme values.
    Applied only to optical bands (R/G/B/NIR); SCL/NDVI/NDWI are unaffected.
    """
    out = img.copy()
    for c in _OPTICAL_BANDS:
        lo, hi = np.percentile(img[c], [low, high])
        out[c] = np.clip(img[c], lo, hi)
    return out


def load_image(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (image float32 (C,H,W), scl uint8 (H,W))."""
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)

    if img.shape[0] < 5:
        raise ValueError(f"Expected ≥5 bands, got {img.shape[0]} in {path}")

    scl = img[4].astype(np.uint8)
    img = _percentile_clip(img)           # remove outliers before computing indices
    red, nir, green = img[0], img[3], img[1]
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndwi = (green - nir) / (green + nir + 1e-8)
    img = np.concatenate([img, np.expand_dims(ndvi, 0), np.expand_dims(ndwi, 0)], axis=0)  # (7, H, W)
    return img, scl


_load_image = load_image  # private alias kept for internal use


def load_mask(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.int64)


_load_mask = load_mask  # private alias kept for internal use


def _cloud_ignore(mask: np.ndarray, scl: np.ndarray) -> np.ndarray:
    cloud = np.isin(scl, list(CLOUD_SCL_CLASSES))
    out = mask.copy()
    out[cloud] = IGNORE_INDEX
    return out


class ChangeDetectionDataset(Dataset):
    """Paired (t1, t2) Sentinel-2 dataset with binary forest-loss change masks.

    Supports two label backends:

    Hansen mode (default, recommended):
        mask_dir contains single-file labels named ``{farm_id}_hansen_label.tif``
        with pixel values: 1 = forest in 2020, 2 = post-EUDR loss 2021-2023.
        The change mask is derived directly: value 2 → change=1, value 1 → change=0.

    Hybrid mode (legacy):
        mask_dir contains per-year files named ``{farm_id}_{year}_hybrid.tif``
        with pixel values: 1 = forest, 2 = crops/plantation, 3 = shrub, 0 = other.
        Change is derived by comparing the two per-year masks.

    Args:
        t1_dir:    Directory of baseline (e.g. 2020) composites.
        t2_dir:    Directory of current (e.g. 2024) composites.
        mask_dir:  Directory of label masks.
        label_backend: ``"hansen"`` (default) or ``"hybrid"``.
        histogram_match: If True, normalise t2 histogram to match t1 before returning.
    """

    def __init__(
        self,
        t1_dir: str,
        t2_dir: str,
        mask_dir: str,
        label_backend: str = "hansen",
        histogram_match: bool = True,
    ) -> None:
        if label_backend not in ("hansen", "hybrid"):
            raise ValueError(f"label_backend must be 'hansen' or 'hybrid', got {label_backend!r}")
        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.mask_dir = mask_dir
        self.label_backend = label_backend
        self.histogram_match = histogram_match

        self.pairs: List[Tuple[str, str, str, Optional[str]]] = []  # (t1, t2, label, label_t2_or_None)
        self._build_pairs()
        logger.info("ChangeDetectionDataset[%s]: %d paired samples", label_backend, len(self.pairs))

    @staticmethod
    def _is_zero_image(path: str) -> bool:
        """Return True if all pixels in every band are zero (corrupted tile)."""
        try:
            with rasterio.open(path) as src:
                data = src.read()
            return bool(np.all(data == 0))
        except Exception:
            return True

    def _build_pairs(self) -> None:
        t1_files = {f for f in os.listdir(self.t1_dir) if f.endswith((".tif", ".tiff"))}
        t2_files = {f for f in os.listdir(self.t2_dir) if f.endswith((".tif", ".tiff"))}
        skipped_zero = 0

        for f in sorted(t1_files):
            # Match both filename formats:
            #   old: {farm_key}_{year}_{date}.tiff
            #   new: {farm_key}_{year}.tiff
            m = re.match(r"(.+?)_(2020|2024)(?:_|\.tiff?)", f)
            if not m:
                continue
            farm_key = m.group(1)

            t2_match = next(
                (g for g in t2_files if re.match(rf"^{re.escape(farm_key)}_(2020|2024)", g) and g != f),
                None,
            )
            if t2_match is None:
                continue

            t1_path = os.path.join(self.t1_dir, f)
            t2_path = os.path.join(self.t2_dir, t2_match)
            if self._is_zero_image(t1_path) or self._is_zero_image(t2_path):
                skipped_zero += 1
                continue

            if self.label_backend == "hansen":
                label_path = os.path.join(self.mask_dir, f"{farm_key}_hansen_label.tif")
                if not os.path.exists(label_path):
                    continue
                self.pairs.append((t1_path, t2_path, label_path, None))
            else:
                # Legacy hybrid: need per-year masks
                mt1 = re.match(r".+?_(2020|2024)", f)
                mt2 = re.match(r".+?_(2020|2024)", t2_match)
                year_t1 = mt1.group(1) if mt1 else "2020"
                year_t2 = mt2.group(1) if mt2 else "2024"
                mask_t1 = os.path.join(self.mask_dir, f"{farm_key}_{year_t1}_hybrid.tif")
                mask_t2 = os.path.join(self.mask_dir, f"{farm_key}_{year_t2}_hybrid.tif")
                if not os.path.exists(mask_t1) or not os.path.exists(mask_t2):
                    continue
                self.pairs.append((t1_path, t2_path, mask_t1, mask_t2))

        if skipped_zero:
            logger.warning("Skipped %d pairs with all-zero images (corrupted tiles)", skipped_zero)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        t1_path, t2_path, label_path, label_t2_path = self.pairs[idx]

        img_t1, scl_t1 = _load_image(t1_path)
        img_t2, scl_t2 = _load_image(t2_path)

        if self.histogram_match:
            img_t2 = match_histogram(img_t2, img_t1)

        h = min(img_t1.shape[1], img_t2.shape[1])
        w = min(img_t1.shape[2], img_t2.shape[2])
        img_t1, img_t2 = img_t1[:, :h, :w], img_t2[:, :h, :w]
        scl_t1, scl_t2 = scl_t1[:h, :w], scl_t2[:h, :w]

        if self.label_backend == "hansen":
            # Hansen label encodes both years in one file.
            # Resize label to match satellite tile if needed (Hansen is 30m, Sentinel is 10m).
            raw = _load_mask(label_path)
            if raw.shape != (h, w):
                from PIL import Image as PILImage
                raw = np.array(
                    PILImage.fromarray(raw.astype(np.uint8)).resize((w, h), PILImage.NEAREST),
                    dtype=np.int64,
                )

            # value 2 = post-EUDR loss → change=1; value 1 = forest in 2020 → change=0
            change = np.zeros((h, w), dtype=np.int64)
            change[raw == HANSEN_POST_EUDR_LOSS] = 1

            # Cloud masking: ignore pixels cloudy in either image
            cloud = np.isin(scl_t1, list(CLOUD_SCL_CLASSES)) | np.isin(scl_t2, list(CLOUD_SCL_CLASSES))
            change[cloud] = IGNORE_INDEX

        else:
            # Hybrid legacy: derive change from two per-year masks
            mask_t1 = _cloud_ignore(_load_mask(label_path)[:h, :w], scl_t1)
            mask_t2 = _cloud_ignore(_load_mask(label_t2_path)[:h, :w], scl_t2)
            change = np.zeros((h, w), dtype=np.int64)
            change[(mask_t1 == FOREST_CLASS) & (mask_t2 != FOREST_CLASS) & (mask_t2 != IGNORE_INDEX)] = 1
            change[(mask_t1 == IGNORE_INDEX) | (mask_t2 == IGNORE_INDEX)] = IGNORE_INDEX

        return (
            torch.from_numpy(img_t1),
            torch.from_numpy(img_t2),
            torch.from_numpy(change),
        )
