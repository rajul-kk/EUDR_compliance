"""E3: a stratified reference sample of cells on the farms, and design-based estimates from its labels.

Strata are Hansen class (0 non-forest, 1 stable tree cover, 2 loss 2021-24) crossed with the S2C prediction of 2020 land use
(0 FOREST, 1 PLANT, 2 OTHER). Estimators follow the stratified-sampling formulas used for map accuracy and area estimation
(Olofsson et al. 2014, Remote Sensing of Environment 148: 42-57; Cochran 1977). Labels come from a person; nothing here
produces them.
"""
from __future__ import annotations

import csv
import math

import numpy as np

CLASSES = ("FOREST", "PLANT", "OTHER")
UNSURE = "UNSURE"
Z95 = 1.959963984540054


def stratum_id(hansen, pred) -> np.ndarray:
    """Hansen class h in {0, 1, 2} and predicted 2020 class p in {0, 1, 2} -> 3 * h + p; -1 where either is invalid."""
    h, p = np.asarray(hansen).astype(int), np.asarray(pred).astype(int)
    ok = (h >= 0) & (h <= 2) & (p >= 0) & (p <= 2)
    return np.where(ok, 3 * h + p, -1)


def n_for_halfwidth(p: float, halfwidth: float, deff: float = 1.0) -> int:
    """Sample size for a 95 % normal interval of the given half-width on a proportion near p (simple random sampling
    times a design effect)."""
    return int(math.ceil(deff * Z95 ** 2 * p * (1 - p) / halfwidth ** 2))


def allocate(N_h: dict, n_total: int, n_min: int = 0) -> dict:
    """Proportional allocation with a floor of n_min per stratum (capped at N_h), largest remainders for the rest."""
    keys = [k for k in sorted(N_h) if N_h[k] > 0]
    n = {k: min(n_min, N_h[k]) for k in keys}
    left = n_total - sum(n.values())
    if left < 0:
        raise ValueError(f"n_total={n_total} is below the floors ({sum(n.values())})")
    room = {k: N_h[k] - n[k] for k in keys}
    tot = sum(room.values())
    if tot == 0 or left == 0:
        return n
    left = min(left, tot)
    share = {k: left * room[k] / tot for k in keys}
    for k in keys:
        n[k] += int(math.floor(share[k]))
    rest = left - sum(int(math.floor(share[k])) for k in keys)
    for k in sorted(keys, key=lambda k: (-(share[k] - math.floor(share[k])), k)):
        if rest == 0:
            break
        if n[k] < N_h[k]:
            n[k] += 1
            rest -= 1
    return n


def draw(strata: np.ndarray, n_h: dict, seed: int) -> np.ndarray:
    """Flat indices of a stratified simple random sample without replacement, grouped by stratum in ascending order."""
    s = np.asarray(strata).ravel()
    rng = np.random.default_rng(seed)
    out = []
    for k in sorted(n_h):
        idx = np.flatnonzero(s == k)
        if n_h[k] > len(idx):
            raise ValueError(f"stratum {k}: n={n_h[k]} > N={len(idx)}")
        out.append(rng.choice(idx, n_h[k], replace=False))
    return np.concatenate(out) if out else np.empty(0, int)


def second_pass_subset(n: int, frac: float, seed: int) -> np.ndarray:
    """Sorted indices of the cells the blind second labeller sees: all of them at frac 1, otherwise a simple random
    subset of round(frac * n) cells (at least one when n > 0)."""
    if not 0 < frac <= 1:
        raise ValueError(f"second-pass fraction must be in (0, 1], got {frac}")
    k = n if frac == 1 else (max(1, round(frac * n)) if n else 0)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def write_sheets(path_sheet: str, path_key: str, lon, lat, key_rows: list, seed: int, prefix: str = "S"):
    """Labelling sheet (shuffled order, coordinates only, blank label) and a separate key linking each sample id to its
    stratum and source. Run twice with different seeds and prefixes for a blind second pass."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lon))
    with open(path_sheet, "w", newline="") as fs, open(path_key, "w", newline="") as fk:
        ws, wk = csv.writer(fs), None
        ws.writerow(["sample_id", "lon", "lat", "label", "confidence", "imagery_date", "note"])
        for rank, i in enumerate(order):
            sid = f"{prefix}{rank:04d}"
            ws.writerow([sid, f"{lon[i]:.6f}", f"{lat[i]:.6f}", "", "", "", ""])
            row = {"sample_id": sid, **key_rows[i]}
            if wk is None:
                wk = csv.DictWriter(fk, fieldnames=list(row))
                wk.writeheader()
            wk.writerow(row)


def stratified_shares(N_h: dict, sample_strata, labels, domain=None, classes=CLASSES) -> dict:
    """Share of each reference class within a domain (a set of strata; default all), with standard error and 95 % interval.

    Labels outside `classes` (UNSURE, blank) are dropped and counted; a stratum keeps its population weight from the
    labelled cells left in it. Variance: sum_h W_h^2 p_hk (1 - p_hk) / (n_h - 1), W_h = N_h / N_domain."""
    s, y = np.asarray(sample_strata), np.asarray(labels, dtype=object)
    dom = sorted(N_h) if domain is None else sorted(domain)
    ok = np.isin(y, list(classes))
    N_dom = sum(N_h[h] for h in dom)
    out = {"n_labelled": int((ok & np.isin(s, dom)).sum()), "n_dropped": int((~ok & np.isin(s, dom)).sum()), "classes": {}}
    empty = [h for h in dom if N_h[h] > 0 and not (ok & (s == h)).any()]
    if empty:
        out["strata_without_labels"] = empty
    for c in classes:
        est, var = 0.0, 0.0
        for h in dom:
            m = ok & (s == h)
            n = int(m.sum())
            if n == 0 or N_h[h] == 0:
                continue
            W = N_h[h] / N_dom
            p = float((y[m] == c).mean())
            est += W * p
            if n > 1:
                var += W ** 2 * p * (1 - p) / (n - 1)
        se = math.sqrt(var)
        out["classes"][c] = {"est": est, "se": se, "lo": max(0.0, est - Z95 * se), "hi": min(1.0, est + Z95 * se)}
    return out


def map_accuracy(N_h: dict, stratum_map_class: dict, sample_strata, labels, classes=CLASSES) -> dict:
    """Estimated area-proportion confusion matrix p[i, j] (map class i, reference class j), with user's and producer's
    accuracy, when every stratum maps to one map class (for example the S2C prediction)."""
    s, y = np.asarray(sample_strata), np.asarray(labels, dtype=object)
    ok = np.isin(y, list(classes))
    N = sum(N_h.values())
    k = len(classes)
    P = np.zeros((k, k))
    for h, Nh in N_h.items():
        m = ok & (s == h)
        if m.sum() == 0 or Nh == 0:
            continue
        i = stratum_map_class[h]
        for j, c in enumerate(classes):
            P[i, j] += Nh / N * float((y[m] == c).mean())
    with np.errstate(invalid="ignore", divide="ignore"):
        ua = np.diag(P) / P.sum(1)
        pa = np.diag(P) / P.sum(0)
    return {"p": P.tolist(), "users": dict(zip(classes, ua.tolist())), "producers": dict(zip(classes, pa.tolist())),
            "overall": float(np.trace(P))}


def plantation_driven_rate(loss_pred_counts, sample_pred, sample_is_forest, min_cells: int = 6, n_boot: int = 2000,
                           n_draw: int = 200, seed: int = 0) -> dict:
    """Share of flagged farms whose flag is plantation-driven (fewer than min_cells loss cells FOREST on 31 Dec 2020), with
    the per-cell truth replaced by the measured P(FOREST | predicted class) on the farms' loss cells.

    loss_pred_counts: (F, 3) loss cells per farm by S2C prediction. sample_pred, sample_is_forest: the labelled loss-cell
    sample (predicted class, reference FOREST yes/no). Each bootstrap resamples the labelled cells within each predicted
    class, re-estimates the three rates, and draws every farm's FOREST count as a sum of binomials; the estimate is the
    mean over draws of the plantation-driven share among flagged farms."""
    c = np.asarray(loss_pred_counts, int)
    flagged = c.sum(1) >= min_cells
    sp, sf = np.asarray(sample_pred, int), np.asarray(sample_is_forest, bool)
    groups = [sf[sp == k] for k in range(3)]
    cf = c[flagged]
    unlabelled = [CLASSES[k] for k in range(3) if cf[:, k].sum() > 0 and len(groups[k]) == 0]
    if len(cf) == 0 or unlabelled:
        return {"n_flagged": int(len(cf)), "est": None, "unlabelled_pred_classes": unlabelled}
    rng = np.random.default_rng(seed)

    def share(q, draws):
        f = sum(rng.binomial(cf[:, k][None, :].repeat(draws, 0), q[k]) for k in range(3))
        return (f < min_cells).mean(1)

    q0 = np.array([g.mean() if len(g) else 0.0 for g in groups])
    est = float(share(q0, n_draw).mean())
    bs = np.empty(n_boot)
    for b in range(n_boot):
        q = np.array([g[rng.integers(0, len(g), len(g))].mean() if len(g) else 0.0 for g in groups])
        bs[b] = share(q, 1)[0]
    return {"n_flagged": int(flagged.sum()), "p_forest_given_pred": dict(zip(CLASSES, q0.tolist())), "est": est,
            "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def agreement(a, b, classes=CLASSES) -> dict:
    """Percent agreement and Cohen's kappa between two labelling passes, on items both passes labelled within `classes`."""
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    m = np.isin(a, list(classes)) & np.isin(b, list(classes))
    a, b = a[m], b[m]
    n = len(a)
    if n == 0:
        return {"n": 0, "agreement": None, "kappa": None}
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in classes)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": n, "agreement": po, "kappa": kappa}


def census_estimate(pred, labels, owner, min_cells: int | None = None, classes=CLASSES) -> dict:
    """Estimate 5: a census, so plain proportions with no sampling error. Shares of the reference classes among the
    labelled cells, overall and by S2C predicted class (UNSURE and blank labels counted and left out), and the farms with
    at least min_cells FOREST loss cells, beside the farms flagged on all their loss cells (as E5).

    pred: S2C class per cell (0 FOREST, 1 PLANT, 2 OTHER); owner: the farm whose outline holds the cell."""
    from hansen_audit.decisions import DEFAULT_MIN_CELLS

    k = DEFAULT_MIN_CELLS if min_cells is None else min_cells
    p, y, o = np.asarray(pred, int), np.asarray(labels, dtype=object), np.asarray(owner, dtype=object)
    ok = np.isin(y, list(classes))

    def shares(m):
        n = int((m & ok).sum())
        return n, int((m & ~ok).sum()), ({c: float((y[m & ok] == c).sum() / n) for c in classes} if n else None)

    n, _, overall = shares(np.ones(len(y), bool))
    by = {}
    for i, c in enumerate(classes):
        nl, nd, sh = shares(p == i)
        by[c] = {"n_labelled": nl, "n_dropped": nd, "shares": sh}
    farms = sorted(set(o.tolist()))
    forest = {f: int(((o == f) & (y == "FOREST")).sum()) for f in farms}
    return {"n": int(len(y)), "n_labelled": n, "n_dropped": int((~ok).sum()), "overall": overall,
            "by_predicted_class": by, "n_farms": len(farms), "min_cells": k,
            "forest_cells_per_farm": forest,
            "farms_with_min_forest_cells": [f for f in farms if forest[f] >= k],
            "farms_flagged_on_all_loss_cells": [f for f in farms if int((o == f).sum()) >= k]}
