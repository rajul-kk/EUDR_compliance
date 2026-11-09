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

    coverage_fraction: fraction of the image footprint covered by the label raster
    (cells outside the label raster are filled with 255 = no data / out of extent).
    """
    with rasterio.open(image_path) as im:
        b, H, W = im.bounds, im.height, im.width
    shape = out_shape or (H, W)
    with rasterio.open(label_path) as lb:
        win = from_bounds(*b, transform=lb.transform)
        arr = lb.read(1, window=win, boundless=True, fill_value=255,
                      out_shape=shape, resampling=rasterio.enums.Resampling.nearest)
        native = lb.read(1, window=win.round_offsets().round_lengths(), boundless=True, fill_value=255)
    cov = float((arr != 255).mean())
    return arr.astype(np.uint8), cov, native


def pixel_center_offset_m(csv_lat, csv_lon, image_path):
    """Distance in metres (small-angle) of the CSV farm point from the image centre."""
    with rasterio.open(image_path) as im:
        b = im.bounds
    clon, clat = (b.left + b.right) / 2, (b.bottom + b.top) / 2
    dy = (csv_lat - clat) * 111_320.0
    dx = (csv_lon - clon) * 111_320.0 * np.cos(np.radians(csv_lat))
    return float(np.hypot(dx, dy))
