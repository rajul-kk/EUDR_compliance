"""Farm-level compliance decisions and cluster-bootstrap rates."""
from __future__ import annotations

import numpy as np

CELL_M2 = 900.0
DEFAULT_MIN_CELLS = 6      # 5,400 m2 ~ 0.54 ha (PROTOCOL)


def farm_flag(loss_cells, min_cells: int = DEFAULT_MIN_CELLS) -> np.ndarray:
    return np.asarray(loss_cells) >= min_cells


def rate_ci(num, den, n_boot: int = 2000, seed: int = 0):
    """Ratio sum(num)/sum(den) over clusters (farms) with a cluster bootstrap. Returns (est, lo, hi)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    n = len(num)
    est = num.sum() / den.sum() if den.sum() > 0 else np.nan
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        d = den[i].sum()
        vals[b] = num[i].sum() / d if d > 0 else np.nan
    return est, float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def false_flag_rate(flag, truth_positive, n_boot=2000, seed=0):
    """P(flag | truly clean) over farms."""
    flag, tp = np.asarray(flag, bool), np.asarray(truth_positive, bool)
    clean = ~tp
    return rate_ci((flag & clean).astype(float), clean.astype(float), n_boot, seed)


def missed_flag_rate(flag, truth_positive, n_boot=2000, seed=0):
    """P(not flagged | truly positive) over farms."""
    flag, tp = np.asarray(flag, bool), np.asarray(truth_positive, bool)
    return rate_ci((~flag & tp).astype(float), tp.astype(float), n_boot, seed)


def expected_false_flag_prob(n_loss_cells: int, precision: float, min_cells: int = DEFAULT_MIN_CELLS) -> float:
    """For a farm Hansen flags (n_loss_cells >= min_cells): P(fewer than min_cells of them are truly loss),
    cells independent with success probability `precision`. Optimistic when errors are spatially
    clustered; used for transported sensitivity only."""
    from scipy.stats import binom
    if n_loss_cells < min_cells:
        return 0.0
    return float(binom.cdf(min_cells - 1, n_loss_cells, precision))
