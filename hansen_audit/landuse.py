"""Land-use classifier on Sentinel-2 cell features and the confusion-corrected decomposition of Hansen loss (PROTOCOL_S2 E1-E3)."""
from __future__ import annotations

import numpy as np

from . import s2

CLASSES = ("FOREST", "PLANT", "OTHER")  # 0, 1, 2 (reference.class_group order is remapped by group_index)
BASE_FEATURES = ("B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "NDMI", "NDVI_STD")
NEIGH_FEATURES = ("NDVI", "NDMI", "B08")
FEATURES = BASE_FEATURES + tuple(f"{c}_n3" for c in NEIGH_FEATURES)

# PROTOCOL_S2B: block-scale context
S2B_INDICES = ("NBR", "NBR2")
S2B_CONTEXT = ("NDVI", "NDMI", "NBR", "B08", "B11", "NDVI_STD")
S2B_MEAN_R = (1, 3, 7)
S2B_STD_R = (3, 7)
S2B_FEATURES = (BASE_FEATURES + S2B_INDICES
                + tuple(f"{c}_m{r}" for c in S2B_CONTEXT for r in S2B_MEAN_R)
                + tuple(f"{c}_s{r}" for c in S2B_CONTEXT for r in S2B_STD_R))
# PROTOCOL_S2C: S2B + Sentinel-1 annual statistics
S1_BASE = ("VV_med", "VH_med", "RATIO_med", "VV_tstd", "VH_tstd")
S1_CONTEXT = ("VH_med", "RATIO_med")
S1_R = (3, 7)
S2C_FEATURES = (S2B_FEATURES + S1_BASE + tuple(f"{c}_m{r}" for c in S1_CONTEXT for r in S1_R)
                + tuple(f"{c}_s{r}" for c in S1_CONTEXT for r in S1_R))
FEATURE_SETS = {"s2": FEATURES, "s2b": S2B_FEATURES, "s2c": S2C_FEATURES}

LGBM_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                   class_weight="balanced", random_state=0, n_jobs=2, verbose=-1)
MAX_TRAIN_PER_GRID = 20_000
N_FOLDS = 6


def nan_box_mean(x: np.ndarray, r: int = 1) -> np.ndarray:
    """Mean over a (2r+1)^2 window ignoring NaN; NaN where the window has no finite value."""
    H, W = x.shape
    v = np.isfinite(x)
    xp = np.pad(np.where(v, x, 0.0), r)
    vp = np.pad(v.astype(np.float32), r)
    s = np.zeros((H, W), np.float64)
    n = np.zeros((H, W), np.float64)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            s += xp[dy:dy + H, dx:dx + W]
            n += vp[dy:dy + H, dx:dx + W]
    with np.errstate(all="ignore"):
        out = s / n
    out[n == 0] = np.nan
    return out.astype(np.float32)


def nan_window_stats(x: np.ndarray, r: int):
    """NaN-aware mean and standard deviation over a (2r+1)^2 window; NaN where the window has no finite value."""
    from scipy.ndimage import uniform_filter

    v = np.isfinite(x)
    x0 = np.where(v, x, 0.0).astype(np.float64)
    size, kw = 2 * r + 1, dict(mode="constant", cval=0.0)
    n = uniform_filter(v.astype(np.float64), size, **kw)
    with np.errstate(all="ignore"):
        mean = uniform_filter(x0, size, **kw) / n
        var = uniform_filter(x0 * x0, size, **kw) / n - mean * mean
    empty = n < 0.5 / size ** 2
    mean[empty] = np.nan
    std = np.sqrt(np.clip(var, 0, None))
    std[empty] = np.nan
    return mean.astype(np.float32), std.astype(np.float32)


def cell_features(cells: np.ndarray, channels=s2.CHANNELS, feature_set: str = "s2") -> np.ndarray:
    """Composite (C, H, W) -> (len(FEATURE_SETS[feature_set]), H, W). Cells invalid in the composite stay NaN in every feature."""
    ch = {c: i for i, c in enumerate(channels)}
    if feature_set == "s2b":
        return _s2b_features(cells, ch)
    base = [cells[ch[c]] for c in BASE_FEATURES]
    neigh = [nan_box_mean(cells[ch[c]]) for c in NEIGH_FEATURES]
    out = np.stack(base + neigh).astype(np.float32)
    invalid = ~np.isfinite(cells[ch["NDVI"]])
    out[:, invalid] = np.nan
    return out



def _s2b_features(cells: np.ndarray, ch: dict) -> np.ndarray:
    layers = {c: cells[ch[c]].astype(np.float32) for c in BASE_FEATURES}
    layers["NBR"] = s2.nd(layers["B08"], layers["B12"])
    layers["NBR2"] = s2.nd(layers["B11"], layers["B12"])
    out = [layers[c] for c in BASE_FEATURES + S2B_INDICES]
    means, stds = [], []
    for c in S2B_CONTEXT:
        stats = {r: nan_window_stats(layers[c], r) for r in sorted(set(S2B_MEAN_R) | set(S2B_STD_R))}
        means += [stats[r][0] for r in S2B_MEAN_R]
        stds += [stats[r][1] for r in S2B_STD_R]
    out = np.stack(out + means + stds).astype(np.float32)
    out[:, ~np.isfinite(cells[ch["NDVI"]])] = np.nan
    return out



def s2c_features(cells: np.ndarray, channels, s1_cells, s1_channels) -> np.ndarray:
    """S2B features plus Sentinel-1 features. s1_cells None (grid-period not downloaded) -> S1 features all NaN.
    The valid population is S2's: cells invalid in the S2 composite stay NaN everywhere."""
    f2b = cell_features(cells, channels, "s2b")
    H, W = f2b.shape[1:]
    if s1_cells is None:
        f1 = np.full((len(S2C_FEATURES) - len(S2B_FEATURES), H, W), np.nan, np.float32)
    else:
        ch = {c: i for i, c in enumerate(s1_channels)}
        base = [s1_cells[ch[c]].astype(np.float32) for c in S1_BASE]
        stats = {(c, r): nan_window_stats(s1_cells[ch[c]].astype(np.float32), r) for c in S1_CONTEXT for r in S1_R}
        f1 = np.stack(base + [stats[(c, r)][0] for c in S1_CONTEXT for r in S1_R]
                      + [stats[(c, r)][1] for c in S1_CONTEXT for r in S1_R]).astype(np.float32)
        f1[:, ~np.isfinite(f2b[0])] = np.nan
    return np.concatenate([f2b, f1])


def group_index(class_ids: np.ndarray) -> np.ndarray:
    """Reference class_ID -> 0 FOREST, 1 PLANT, 2 OTHER (PROTOCOL.md groups)."""
    from .reference import FOREST_IDS, PLANT_IDS

    g = np.full(class_ids.shape, 2, np.int8)
    g[np.isin(class_ids, list(PLANT_IDS))] = 1
    g[np.isin(class_ids, list(FOREST_IDS))] = 0
    return g


def assign_folds(grid_ids, islands, k: int = N_FOLDS, seed: int = 0) -> dict:
    """Grouped folds balanced by island: shuffle grids within each island, deal them round-robin, continuing across islands."""
    rng = np.random.default_rng(seed)
    folds, pos = {}, 0
    for isl in sorted(set(islands)):
        g = sorted(gid for gid, i in zip(grid_ids, islands) if i == isl)
        rng.shuffle(g)
        for gid in g:
            folds[gid] = pos % k
            pos += 1
    return folds


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> np.ndarray:
    """C[i, j] = number of cells with true class i predicted as j."""
    return np.bincount(y_true.astype(np.int64) * k + y_pred.astype(np.int64), minlength=k * k).reshape(k, k)


def recall_precision(counts: np.ndarray):
    with np.errstate(all="ignore"):
        rec = np.diag(counts) / counts.sum(1)
        prec = np.diag(counts) / counts.sum(0)
    return rec, prec


def balanced_accuracy(counts: np.ndarray) -> float:
    rec, _ = recall_precision(counts)
    return float(np.nanmean(rec))


def corrected_shares(conf_counts: np.ndarray, pred_counts: np.ndarray):
    """Invert misclassification: q = C^T pi, with C[i, j] = P(pred j | true i). Returns (pi, clipped)."""
    C = conf_counts / conf_counts.sum(1, keepdims=True)
    q = pred_counts / pred_counts.sum()
    pi = np.linalg.solve(C.T, q)
    clipped = bool((pi < 0).any())
    pi = np.clip(pi, 0, None)
    return pi / pi.sum(), clipped


def cluster_bootstrap(conf_by_grid: np.ndarray, pred_by_grid: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """conf_by_grid (G, 3, 3), pred_by_grid (G, 3). Resample grids; returns dict of point estimates and percentile 95 % intervals."""
    rng = np.random.default_rng(seed)
    G = len(pred_by_grid)

    def est(idx):
        conf = conf_by_grid[idx].sum(0)
        pred = pred_by_grid[idx].sum(0)
        raw = pred / pred.sum()
        cor, clipped = corrected_shares(conf, pred)
        return raw, cor, clipped

    raw0, cor0, clip0 = est(np.arange(G))
    raws, cors, clips = [], [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, G, G)
        try:
            r, c, cl = est(idx)
        except np.linalg.LinAlgError:
            continue
        raws.append(r)
        cors.append(c)
        clips += cl
    raws, cors = np.array(raws), np.array(cors)

    def ci(a):
        return np.nanpercentile(a, [2.5, 97.5], axis=0)

    out = {"n_grids": int(G), "n_cells": int(pred_by_grid.sum()), "n_boot_ok": int(len(raws)),
           "clipped_point": clip0, "clipped_boot_frac": clips / max(len(raws), 1)}
    for name, p, a in (("raw", raw0, raws), ("corrected", cor0, cors)):
        lo, hi = ci(a)
        out[name] = {c: {"est": float(p[i]), "lo": float(lo[i]), "hi": float(hi[i])} for i, c in enumerate(CLASSES)}
    return out


def metric_bootstrap(conf_by_grid: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Recall, precision and balanced accuracy with grid-cluster bootstrap intervals."""
    rng = np.random.default_rng(seed)
    G = len(conf_by_grid)

    def stats(c):
        rec, prec = recall_precision(c)
        return np.concatenate([rec, prec, [balanced_accuracy(c)]])

    p = stats(conf_by_grid.sum(0))
    bs = np.array([stats(conf_by_grid[rng.integers(0, G, G)].sum(0)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    names = [f"recall_{c}" for c in CLASSES] + [f"precision_{c}" for c in CLASSES] + ["balanced_accuracy"]
    res = {n: {"est": float(p[i]), "lo": float(lo[i]), "hi": float(hi[i])} for i, n in enumerate(names)}
    res["confusion"] = conf_by_grid.sum(0).tolist()
    res["n_grids"] = int(G)
    return res
