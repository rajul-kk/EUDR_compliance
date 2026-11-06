"""PROTOCOL_S2 E4 (deviation 5): shared download windows for farm footprints and farm-level summaries."""
from __future__ import annotations

import numpy as np

from .s2 import HANSEN_RES

MIN_LOSS_CELLS = 6


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
