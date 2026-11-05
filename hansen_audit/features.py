"""Image pair -> 30 m cell features on the Hansen label grid.

Per farm: a (C, G, G) feature stack with G = 37 cells (~0.01 deg tile / 0.00027 deg cell), labels sampled
(nearest neighbour) at cell centres. Channels (15): for t1 and t2: R, G, B, NIR, NDVI of SCL-valid pixel means,
then t2 - t1 for the same five. Validity per cell = min over years of the share of SCL-valid pixels.
"""
from __future__ import annotations

import numpy as np

from .align import label_on_image_grid

GRID = 37
CLOUD_SCL = (0, 1, 3, 8, 9, 10)
CHANNELS = [f"{y}_{b}" for y in ("t1", "t2") for b in ("R", "G", "B", "NIR", "NDVI")] + [f"d_{b}" for b in ("R", "G", "B", "NIR", "NDVI")]


def _cell_means(img: np.ndarray, grid: int = GRID):
    """img (5,H,W) float32 with SCL in band 5. Returns (feat (5,G,G) = R,G,B,NIR,NDVI, valid_frac (G,G))."""
    import torch
    import torch.nn.functional as F

    scl = img[4]
    valid = ~np.isin(scl.astype(np.int16), CLOUD_SCL)
    x = torch.from_numpy(img[:4] * valid[None]).unsqueeze(0)
    w = torch.from_numpy(valid.astype(np.float32))[None, None]
    num = F.adaptive_avg_pool2d(x, grid)[0].numpy()
    den = F.adaptive_avg_pool2d(w, grid)[0, 0].numpy()
    mean = np.where(den > 0, num / np.maximum(den, 1e-6), np.nan)
    ndvi = (mean[3] - mean[0]) / (mean[3] + mean[0] + 1e-8)
    return np.concatenate([mean, ndvi[None]], 0), den


def farm_features(baseline_path, current_path, label_path, grid: int = GRID):
    import rasterio

    with rasterio.open(baseline_path) as s:
        a = s.read().astype(np.float32)
    with rasterio.open(current_path) as s:
        b = s.read().astype(np.float32)
    fa, va = _cell_means(a, grid)
    fb, vb = _cell_means(b, grid)
    lab, cov, _ = label_on_image_grid(label_path, baseline_path, out_shape=(grid, grid))
    X = np.concatenate([fa, fb, fb - fa], 0)
    valid = np.minimum(va, vb)
    return X.astype(np.float32), lab.astype(np.uint8), valid.astype(np.float32), cov
