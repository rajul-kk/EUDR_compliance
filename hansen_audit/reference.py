"""Reference cells: Zenodo 23874371 land-use polygons vs Hansen re-derived with the pipeline recipe.

Hansen labels are re-derived from the public GFC-2025-v1.13 tiles with the SAME recipe as
GEE_dynamic/src/generate_labels.py (treecover2000 >= 10; lossyear 21..24), on Hansen's native
grid (0.00025 deg, ~27.8 m). The production labels were exported by GEE at scale 30 m in EPSG:4326
(0.000269 deg grid); the two differ by resampling only (documented in docs/reference_labels.md).
"""
from __future__ import annotations

import math

import numpy as np

HANSEN_BASE = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2025-v1.13/Hansen_GFC-2025-v1.13_{layer}_{tile}.tif"
RES = 0.00025
TREECOVER_THRESHOLD = 10

FOREST_IDS = {9, 12}
PLANT_IDS = {1, 2, 4, 5, 6, 7, 8}
# groups: 0 = not annotated, 1 = FOREST, 2 = PLANT, 3 = OTHER
G_NONE, G_FOREST, G_PLANT, G_OTHER = 0, 1, 2, 3


def class_group(class_id: int) -> int:
    if class_id in FOREST_IDS:
        return G_FOREST
    if class_id in PLANT_IDS:
        return G_PLANT
    return G_OTHER


def hansen_label(treecover: np.ndarray, lossyear: np.ndarray) -> np.ndarray:
    """Identical logic to generate_labels.py. 0 non-forest/pre-2021 loss, 1 forest no loss, 2 loss 2021-24."""
    was_forest = treecover >= TREECOVER_THRESHOLD
    post = (lossyear >= 21) & (lossyear <= 24)
    no_pre = (lossyear == 0) | (lossyear >= 21)
    lab = np.zeros(treecover.shape, np.uint8)
    lab[was_forest & no_pre] = 1
    lab[was_forest & post] = 2
    return lab


def tile_name(lon: float, lat: float) -> str:
    """Hansen tiles are named by their upper-left corner, 10 degree squares."""
    tlon = math.floor(lon / 10) * 10
    tlat = math.floor(lat / 10) * 10 + 10
    return f"{abs(tlat):02d}{'N' if tlat >= 0 else 'S'}_{abs(tlon):03d}{'E' if tlon >= 0 else 'W'}"


def read_window(layer: str, west: float, south: float, east: float, north: float):
    """Read a Hansen layer window; returns (array, (west, north)) snapped to the native grid."""
    import rasterio
    from rasterio.windows import Window

    tile = tile_name((west + east) / 2, (north + south) / 2)
    url = "/vsicurl/" + HANSEN_BASE.format(layer=layer, tile=tile)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                      GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2"):
        with rasterio.open(url) as s:
            l, t = s.bounds.left, s.bounds.top
            c0, c1 = int(math.floor((west - l) / RES)), int(math.ceil((east - l) / RES))
            r0, r1 = int(math.floor((t - north) / RES)), int(math.ceil((t - south) / RES))
            arr = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
            return arr, (l + c0 * RES, t - r0 * RES)


def rasterise_reference(gdf, west: float, north: float, shape, sub: int = 5):
    """Rasterise polygons (EPSG:4326, columns class_ID) to the Hansen grid.

    Supersample `sub` x, take majority class_ID per cell. Returns (class_id [H,W] int16 (-1 none),
    coverage [H,W] fraction of sub-cells annotated). Later polygons overwrite earlier on overlap.
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    H, W = shape
    tr = from_origin(west, north, RES / sub, RES / sub)
    shapes = [(g, int(c) + 1) for g, c in zip(gdf.geometry, gdf["class_ID"]) if g is not None and not g.is_empty]
    fine = rasterize(shapes, out_shape=(H * sub, W * sub), transform=tr, fill=0, dtype="int16")
    blocks = fine.reshape(H, sub, W, sub).transpose(0, 2, 1, 3).reshape(H, W, sub * sub)
    cov = (blocks > 0).mean(-1)
    ncls = int(blocks.max()) + 1
    counts = np.stack([(blocks == k).sum(-1) for k in range(1, ncls)], -1) if ncls > 1 else np.zeros((H, W, 1), int)
    maj = counts.argmax(-1).astype(np.int16)  # class_ID (since k=1 -> id 0)
    maj[cov == 0] = -1
    return maj, cov
