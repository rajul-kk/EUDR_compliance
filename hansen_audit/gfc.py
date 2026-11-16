"""PROTOCOL_E9: Hansen labels against the JRC GFC2020 forest map. Pure functions on per-cell arrays and counts.

Hansen classes: 0 not forest in 2000 or lost by 2020, 1 forest through 2024, 2 lost 2021-2024, 255 nodata.
GFC forest fraction: integer percent of the cell covered by GFC2020 forest pixels (0-100).
"""
import os

import numpy as np

from .decisions import farm_flag, rate_ci
from .reference import LABEL_NODATA

NODATA = LABEL_NODATA
FOREST_PCT = 50


def gfc_forest(frac_pct: np.ndarray, threshold: int = FOREST_PCT) -> np.ndarray:
    """Bool: cell is GFC forest when its forest fraction is at least `threshold` percent."""
    return np.asarray(frac_pct) >= threshold


def farm_counts(labels: np.ndarray, forest: np.ndarray) -> dict:
    """Per-farm cell counts by Hansen class, total and GFC forest; nodata cells excluded."""
    labels, forest = np.asarray(labels).ravel(), np.asarray(forest, bool).ravel()
    ok = labels != NODATA
    return {f"class{c}": int((ok & (labels == c)).sum()) for c in (0, 1, 2)} | \
           {f"class{c}_gfc": int((ok & (labels == c) & forest).sum()) for c in (0, 1, 2)}


def gated_flags(loss_cells: np.ndarray, loss_gfc_cells: np.ndarray):
    """(Hansen flag, GFC-gated flag) per farm, both with the product's flag rule (`decisions.farm_flag`)."""
    return farm_flag(loss_cells), farm_flag(loss_gfc_cells)


def ratio_ci(num: np.ndarray, den: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Pooled ratio with a farm-bootstrap 95 % interval (`decisions.rate_ci`) as {est, lo, hi}; None when empty."""
    if float(np.sum(den)) == 0:
        return None
    est, lo, hi = rate_ci(num, den, n_boot, seed)
    return {"est": float(est), "lo": lo, "hi": hi}


def agreement(both: int, hansen_only: int, gfc_only: int, neither: int) -> dict:
    """Overall agreement and Cohen's kappa from the four confusion counts of two boolean forest maps."""
    n = both + hansen_only + gfc_only + neither
    out = {"n": n, "both": both, "hansen_only": hansen_only, "gfc_only": gfc_only, "neither": neither}
    if n == 0:
        return out | {"agreement": None, "kappa": None}
    po = (both + neither) / n
    pe = ((both + hansen_only) * (both + gfc_only) + (gfc_only + neither) * (hansen_only + neither)) / n ** 2
    return out | {"agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else None}


def agreement_from_counts(col: dict) -> dict:
    """Hansen forest in 2020 (classes 1 and 2) against GFC forest, from summed farm_counts columns."""
    h, hg = int(col["class1"].sum() + col["class2"].sum()), int(col["class1_gfc"].sum() + col["class2_gfc"].sum())
    n0, g0 = int(col["class0"].sum()), int(col["class0_gfc"].sum())
    return agreement(both=hg, hansen_only=h - hg, gfc_only=g0, neither=n0 - g0)


def s2c_reading(plant_nonforest, forest_forest, map_name="GFC2020") -> str:
    """PROTOCOL_E9 analysis 4: agree if both shares >= 0.70, disagree if either <= 0.50, else mixed."""
    if plant_nonforest is None or forest_forest is None:
        return "too few cells"
    if plant_nonforest <= 0.5 or forest_forest <= 0.5:
        return f"S2C and {map_name} disagree"
    if plant_nonforest >= 0.7 and forest_forest >= 0.7:
        return f"S2C and {map_name} agree"
    return "mixed"


PRIMARY = "V3"   # PROTOCOL_E9: GFC2020 version of the primary analysis


def load(gfc_dir, version, name):
    """A saved GFC2020 forest-percentage grid, or None if it was not fetched."""
    p = os.path.join(gfc_dir, version, f"{name}.npy")
    return np.load(p) if os.path.exists(p) else None


def flag_block(col, m=None):
    """Flag counts and the share of Hansen flags that do not survive GFC gating, over farms m (all if None)."""
    hflag, gflag = gated_flags(col["class2"], col["class2_gfc"])
    m = np.ones(len(hflag), bool) if m is None else m
    return {"hansen_flagged": int(hflag[m].sum()), "gated_flagged": int(gflag[m].sum()),
            "share_flags_not_surviving": ratio_ci((hflag & ~gflag)[m].astype(float), hflag[m].astype(float))}


def summarise(table, group):
    """PROTOCOL_E9 analyses 1-3 over a {farm: farm_counts} table, overall and by group."""
    keys = sorted(table)
    col = {c: np.array([table[k][c] for k in keys]) for c in table[keys[0]]}
    by_group = {}
    for g in sorted({group[k] for k in keys}):
        m = np.array([group[k] == g for k in keys])
        by_group[g] = {"n_farms": int(m.sum()), "loss_cells": int(col["class2"][m].sum()),
                       "share_loss_on_gfc_forest": ratio_ci(col["class2_gfc"][m], col["class2"][m])} | flag_block(col, m)
    return {"n_farms": len(keys), "loss_cells": int(col["class2"].sum()),
            "share_loss_on_gfc_forest": ratio_ci(col["class2_gfc"], col["class2"]),
            "share_class1_on_gfc_forest": ratio_ci(col["class1_gfc"], col["class1"]),
            "share_class0_on_gfc_forest": ratio_ci(col["class0_gfc"], col["class0"]),
            **flag_block(col), "agreement_forest_2020": agreement_from_counts(col), "by_group": by_group}
