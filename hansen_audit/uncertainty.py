"""Per-farm uncertainty: ensemble spread and split conformal."""
from __future__ import annotations

import numpy as np


def ensemble_summary(member_area: np.ndarray, min_cells: float) -> dict:
    """member_area: (M, F) predicted loss cells per farm for each member."""
    m = member_area.shape[0]
    sd = member_area.std(0, ddof=1) if m > 1 else np.zeros(member_area.shape[1])
    return dict(mean=member_area.mean(0), std=sd, flag_frac=(member_area >= min_cells).mean(0))


def split_conformal_halfwidth(scores_cal: np.ndarray, alpha: float = 0.1) -> float:
    n = len(scores_cal)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores_cal)[k - 1])


def decision_with_interval(pred, halfwidth, min_cells):
    """0 = not flagged (interval entirely below), 1 = flagged (entirely at/above), 2 = uncertain."""
    pred = np.asarray(pred, float)
    lo, hi = pred - halfwidth, pred + halfwidth
    return np.where(lo >= min_cells, 1, np.where(hi < min_cells, 0, 2))
