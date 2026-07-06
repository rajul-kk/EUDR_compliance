"""Rasterio-based mask alignment utilities.

Reprojects a GEE mask (slave) to exactly match the geospatial footprint of a
Sentinel-2 composite (master): same CRS, transform, and pixel dimensions.
Used by FarmSegmentationDataset before feeding masks to the model.
"""

import glob
import os
import re

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


def align_mask_to_image(master_path: str, slave_path: str, output_path: str) -> bool:
    """Reproject slave (mask) to match master (Sentinel-2 image) exactly.

    Args:
        master_path: Path to the Sentinel-2 composite (master geometry).
        slave_path:  Path to the GEE mask (slave to be reprojected).
        output_path: Destination for the aligned single-band uint8 GeoTIFF.

    Returns:
        True on success, False if an exception occurred.
    """
    try:
        with rasterio.open(master_path) as master:
            dst_transform = master.transform
            dst_width = master.width
            dst_height = master.height
            dst_crs = master.crs
            dst_profile = master.profile.copy()

        dst_profile.update(
            driver="GTiff",
            height=dst_height,
            width=dst_width,
            transform=dst_transform,
            crs=dst_crs,
            count=1,
            dtype=rasterio.uint8,
            nodata=0,
        )

        destination = np.zeros((dst_height, dst_width), dtype=np.uint8)

        with rasterio.open(slave_path) as slave:
            reproject(
                source=rasterio.band(slave, 1),
                destination=destination,
                src_transform=slave.transform,
                src_crs=slave.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,  # categorical data — no interpolation
            )

        with rasterio.open(output_path, "w", **dst_profile) as dst:
            dst.write(destination, 1)

        return True

    except Exception as exc:
        print(f"Alignment failed for {os.path.basename(slave_path)} → {os.path.basename(master_path)}: {exc}")
        return False


def check_alignment(master_path: str, slave_path: str) -> bool:
    """Return True if slave is already aligned to master (same CRS, transform, dimensions)."""
    try:
        with rasterio.open(master_path) as m, rasterio.open(slave_path) as s:
            return (
                m.width == s.width
                and m.height == s.height
                and m.crs == s.crs
                and m.transform == s.transform
            )
    except Exception as exc:
        print(f"Error checking alignment: {exc}")
        return False


def batch_align_masks(raw_dir: str, mask_dir: str, output_dir: str) -> None:
    """Align all hybrid masks in mask_dir to their corresponding images in raw_dir.

    File naming convention:
        raw:  ``{type}_{id}_{year}_{date}.tiff``
        mask: ``osm_{type}_{id}_{year}_hybrid.tif``
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_files = glob.glob(os.path.join(raw_dir, "*.tiff")) + glob.glob(os.path.join(raw_dir, "*.tif"))

    for raw_path in raw_files:
        m = re.match(r"(relation|way)_(\d+)_(\d{4})_.*\.tiff?", os.path.basename(raw_path))
        if not m:
            continue
        obj_type, obj_id, year = m.groups()
        mask_path = os.path.join(mask_dir, f"osm_{obj_type}_{obj_id}_{year}_hybrid.tif")
        if not os.path.exists(mask_path):
            continue
        out_name = f"{os.path.splitext(os.path.basename(raw_path))[0]}_mask_aligned.tif"
        align_mask_to_image(raw_path, mask_path, os.path.join(output_dir, out_name))
