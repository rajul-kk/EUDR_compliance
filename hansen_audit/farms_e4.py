"""PROTOCOL_S2 E4 (deviation 5): shared download windows for farm footprints and farm-level summaries."""
from __future__ import annotations

import numpy as np

from .io import read_json
from .reference import LABEL_NODATA
from .s2 import HANSEN_RES, hansen_transform

MIN_LOSS_CELLS = 6


def read_boxes(windows_json) -> dict:
    """{farm: (west, south, east, north)} from an E4/E8 windows.json."""
    W = read_json(windows_json)
    return dict(zip(W["farms"], np.array(W["boxes"])))


def iter_windows(windows_json, preds_npz):
    """(name, window, origin, shape, S2C predictions or None) for each window of an E4/E8 windows.json; the name is
    `win_<id>` and the predictions are None when farm_preds.npz has no entry for it."""
    preds = np.load(preds_npz)
    for w in read_json(windows_json)["windows"]:
        name = f"win_{w['id']}"
        yield name, w, np.array(w["origin"]), tuple(w["shape"]), preds[name] if name in preds.files else None


def label_on_window(label_path, origin, shape):
    """A farm's label raster resampled (nearest) onto a window on Hansen's grid; 255 outside the raster."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    out = np.full(shape, LABEL_NODATA, np.uint8)
    with rasterio.open(label_path) as src:
        reproject(rasterio.band(src, 1), out, dst_transform=hansen_transform(origin), dst_crs="EPSG:4326",
                  resampling=Resampling.nearest, dst_nodata=LABEL_NODATA)
    return out


def group_windows(boxes, span: float = 0.05):
    """boxes: (N, 4) west, south, east, north. Greedy in input order: each window takes every remaining box that keeps the union
    box <= span per side (checked against the seed box's union with that box alone, then re-checked on the full union)."""
    boxes = np.asarray(boxes, float)
    left = list(range(len(boxes)))
    out = []
    while left:
        w, s, e, n = boxes[left[0]]
        members = []
        for j in left:
            W, S, E, N = min(w, boxes[j, 0]), min(s, boxes[j, 1]), max(e, boxes[j, 2]), max(n, boxes[j, 3])
            if E - W <= span + 1e-12 and N - S <= span + 1e-12:
                members.append(j)
                w, s, e, n = W, S, E, N
        out.append(members)
        left = [j for j in left if j not in members]
    return out


def window_grid(boxes):
    """Union box of member farm boxes -> (origin (west, north), shape (H, W)) on a HANSEN_RES grid."""
    boxes = np.asarray(boxes, float)
    w, s, e, n = boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()
    H, W = int(np.ceil((n - s) / HANSEN_RES - 1e-9)), int(np.ceil((e - w) / HANSEN_RES - 1e-9))
    return np.array([w, n]), (H, W)


def farm_cells(origin, shape, box) -> np.ndarray:
    """Bool (H, W): window cells whose centres fall inside the farm box."""
    H, W = shape
    lon = origin[0] + (np.arange(W) + 0.5) * HANSEN_RES
    lat = origin[1] - (np.arange(H) + 0.5) * HANSEN_RES
    inx = (lon >= box[0]) & (lon < box[2])
    iny = (lat > box[1]) & (lat <= box[3])
    return iny[:, None] & inx[None, :]


def farm_summary(loss_pred_counts, min_cells: int = MIN_LOSS_CELLS, n_boot: int = 2000, seed: int = 0):
    """loss_pred_counts: (F, 3) counts of each farm's valid loss cells predicted FOREST / PLANT / OTHER in 2020."""
    c = np.asarray(loss_pred_counts)
    n_loss = c.sum(1)
    flagged = n_loss >= min_cells
    plant_driven = flagged & (c[:, 0] < min_cells)
    out = {"n_farms": int(len(c)), "n_flagged": int(flagged.sum()), "n_plantation_driven": int(plant_driven.sum())}
    if flagged.sum() == 0:
        out["share_plantation_driven"] = None
        return out
    pd_f = plant_driven[flagged].astype(float)
    rng = np.random.default_rng(seed)
    bs = [pd_f[rng.integers(0, len(pd_f), len(pd_f))].mean() for _ in range(n_boot)]
    out["share_plantation_driven"] = {"est": float(pd_f.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
    tot = c[flagged].sum(0)
    out["flagged_farms_loss_cells_by_2020_class"] = {"FOREST": int(tot[0]), "PLANT": int(tot[1]), "OTHER": int(tot[2])}
    return out
