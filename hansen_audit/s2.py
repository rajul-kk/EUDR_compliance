"""Sentinel-2 L2A composites on the Hansen grid of a reference grid (PROTOCOL_S2 data rules).

Source: Microsoft Planetary Computer STAC, anonymous SAS token (no account). Each scene is warped onto a fine
EPSG:4326 grid that subdivides the Hansen 0.00025 deg grid 3x3 (~9.3 m), the per-pixel median of clear observations
over the best K scenes is taken, then the fine grid is aggregated to Hansen cells.
"""
from __future__ import annotations

import json
import time
import urllib.request

import numpy as np

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
COLLECTION = "sentinel-2-l2a"
BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
BAD_SCL = (0, 1, 3, 8, 9, 10)  # nodata, saturated, cloud shadow, cloud medium/high, cirrus
SUB = 3  # fine pixels per Hansen cell side
HANSEN_RES = 0.00025
MAX_TILE_CLOUD = 80
K_SCENES = 4  # PROTOCOL_S2 deviation 2 (was 6)
MAX_CANDIDATES = 12  # PROTOCOL_S2 deviation 1: only the 12 lowest tile-cloud items are ranked by SCL
MIN_OBS = 2
MIN_VALID_FRAC = 0.5

# output channel order of a composite on the Hansen grid
CHANNELS = BANDS + ("NDVI", "NDMI", "NDVI_STD", "VALID_FRAC", "NOBS")


def harmonise(dn: np.ndarray, baseline: str | None) -> np.ndarray:
    """DN -> reflectance. Processing baseline >= 04.00 adds a +1000 offset (BOA_ADD_OFFSET = -1000)."""
    x = dn.astype(np.float32)
    if baseline is not None and float(baseline) >= 4.0:
        x = x - 1000.0
    return np.clip(x / 10000.0, 0.0, 1.5)


def clear_mask(scl: np.ndarray) -> np.ndarray:
    return ~np.isin(scl, BAD_SCL)


def grid_bounds(origin, shape):
    """(west, south, east, north) of a Hansen-grid window with origin (west, north) and shape (H, W)."""
    w, n = float(origin[0]), float(origin[1])
    H, W = shape
    return w, n - H * HANSEN_RES, w + W * HANSEN_RES, n


def fine_transform(origin):
    from rasterio.transform import from_origin

    r = HANSEN_RES / SUB
    return from_origin(float(origin[0]), float(origin[1]), r, r)


def aggregate(fine: np.ndarray, sub: int = SUB, fn=np.nanmean) -> np.ndarray:
    """(..., H*sub, W*sub) -> (..., H, W) applying fn over each sub x sub block."""
    *lead, Hs, Ws = fine.shape
    H, W = Hs // sub, Ws // sub
    x = fine[..., : H * sub, : W * sub].reshape(*lead, H, sub, W, sub)
    x = np.moveaxis(x, -3, -2).reshape(*lead, H, W, sub * sub)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return fn(x, axis=-1)


def nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return (a - b) / (a + b)


def composite_to_cells(stack: np.ndarray, clear: np.ndarray) -> np.ndarray:
    """stack (S, B, Hs, Ws) reflectance, clear (S, Hs, Ws) bool -> (len(CHANNELS), H, W) on the Hansen grid.

    Per fine pixel: median over clear scenes; pixel valid if >= MIN_OBS clear scenes. Per cell: mean of valid fine pixels,
    NDVI std over the cell's valid fine pixels, valid fraction and mean observation count.
    """
    s = np.where(clear[:, None], stack, np.nan)
    nobs = clear.sum(0).astype(np.float32)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(s, axis=0)  # (B, Hs, Ws)
    ok = nobs >= MIN_OBS
    med[:, ~ok] = np.nan
    b = dict(zip(BANDS, med))
    ndvi = nd(b["B08"], b["B04"])
    ndmi = nd(b["B08"], b["B11"])
    cells = [aggregate(m) for m in med]
    cells.append(aggregate(ndvi))
    cells.append(aggregate(ndmi))
    cells.append(aggregate(ndvi, fn=np.nanstd))
    vf = aggregate(ok.astype(np.float32), fn=np.mean)
    cells.append(vf)
    cells.append(aggregate(nobs, fn=np.mean))
    out = np.stack(cells).astype(np.float32)
    out[: len(CHANNELS) - 2, vf < MIN_VALID_FRAC] = np.nan
    return out


# ---------------------------------------------------------------- network parts


class Token:
    def __init__(self):
        self._tok, self._exp = None, 0.0

    def get(self) -> str:
        if self._tok is None or time.time() > self._exp - 300:
            d = json.load(urllib.request.urlopen(TOKEN_URL, timeout=60))
            self._tok = d["token"]
            self._exp = time.time() + 45 * 60  # tokens last ~1 h; refresh early
        return self._tok


def search(bounds, start: str, end: str):
    from pystac_client import Client

    c = Client.open(STAC_URL)
    s = c.search(collections=[COLLECTION], bbox=list(bounds), datetime=f"{start}/{end}",
                 query={"eo:cloud_cover": {"lt": MAX_TILE_CLOUD}})
    return sorted(s.items(), key=lambda i: i.properties["eo:cloud_cover"])


def warp_asset(href: str, origin, fine_shape, resampling="nearest", retries=3) -> np.ndarray:
    """Read only the needed window of a COG and warp it onto the fine grid. Out-of-footprint -> 0."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    dst_tr = fine_transform(origin)
    Hs, Ws = fine_shape
    last = None
    for k in range(retries):
        try:
            with rasterio.open(href) as src, WarpedVRT(
                src, crs="EPSG:4326", transform=dst_tr, width=Ws, height=Hs,
                resampling=getattr(Resampling, resampling), nodata=0,
            ) as vrt:
                return vrt.read(1)
        except Exception as e:  # transient HTTP errors
            last = e
            time.sleep(2 * (k + 1))
    raise last


def signed(item, asset: str, token: Token) -> str:
    return item.assets[asset].href + "?" + token.get()


def build_composite(origin, shape, start: str, end: str, token: Token, k: int = K_SCENES, log=print):
    """Full pipeline for one grid and one period. Returns (cells array, metadata dict)."""
    bounds = grid_bounds(origin, shape)
    fine_shape = (shape[0] * SUB, shape[1] * SUB)
    items = search(bounds, start, end)
    ranked = []
    for it in items[:MAX_CANDIDATES]:
        scl = warp_asset(signed(it, "SCL", token), origin, fine_shape)
        cf = float(clear_mask(scl).mean())
        ranked.append((cf, it, scl))
    ranked.sort(key=lambda t: -t[0])
    chosen = [t for t in ranked if t[0] > 0.05][:k]
    log(f"  {start[:4]}: {len(items)} items, clear fractions {[round(t[0], 2) for t in ranked[:10]]}, using {len(chosen)}")
    if not chosen:
        return None, {"n_items": len(items), "scenes": []}
    S = len(chosen)
    stack = np.empty((S, len(BANDS)) + fine_shape, np.float32)
    clear = np.empty((S,) + fine_shape, bool)
    meta = []
    for si, (cf, it, scl) in enumerate(chosen):
        pb = it.properties.get("s2:processing_baseline")
        for bi, b in enumerate(BANDS):
            dn = warp_asset(signed(it, b, token), origin, fine_shape)
            stack[si, bi] = harmonise(dn, pb)
        clear[si] = clear_mask(scl) & (stack[si, 3] > 0)
        meta.append({"id": it.id, "datetime": it.properties["datetime"], "baseline": pb, "clear_frac": cf})
    cells = composite_to_cells(stack, clear)
    return cells, {"n_items": len(items), "scenes": meta}
