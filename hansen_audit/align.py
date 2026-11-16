"""Spatial alignment: window the coarse (30 m) Hansen label onto the image footprint.

Label rasters (~5 km box) are larger than the image tiles (~1.1 km box, 512 px).
We never resample the label by interpolation: nearest-neighbour onto the image grid
(block-replicated 30 m cells), and also expose the exact native-pixel window.
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import from_bounds


def label_on_image_grid(label_path, image_path, out_shape=None):
    """Return (label_aligned uint8 [H,W], coverage_fraction, native_window_array).

    coverage_fraction: fraction of the image footprint inside the label raster's bounds (cells outside are filled
    with 255). 255 can also sit inside the footprint: the rebuilt labels (PROTOCOL_E10) mark Hansen no-data with it, so
    readers must keep the classes they want (`1 <= y <= 2`, as model.make_patches does), never test `y >= 1` or `!= 0`.
    """
    with rasterio.open(image_path) as im:
        b, H, W = im.bounds, im.height, im.width
    shape = out_shape or (H, W)
    with rasterio.open(label_path) as lb:
        win = from_bounds(*b, transform=lb.transform)
        arr = lb.read(1, window=win, boundless=True, fill_value=255,
                      out_shape=shape, resampling=rasterio.enums.Resampling.nearest)
        native = lb.read(1, window=win.round_offsets().round_lengths(), boundless=True, fill_value=255)
        lbb = lb.bounds
    ix = max(0.0, min(b.right, lbb.right) - max(b.left, lbb.left))
    iy = max(0.0, min(b.top, lbb.top) - max(b.bottom, lbb.bottom))
    cov = float(ix * iy / ((b.right - b.left) * (b.top - b.bottom)))
    return arr.astype(np.uint8), cov, native
