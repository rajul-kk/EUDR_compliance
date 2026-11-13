"""Farm-level compliance decisions and cluster-bootstrap rates.

`plot_decision` is the one decision rule, shared by the audit and the product (`src/change_detector.py`). It flags a
plot when forest loss since the 2020 baseline covers more than 0.5 ha, the minimum area of "forest" in Art. 2(4) of
Regulation (EU) 2023/1115 (docs/regulation_check.md; the text sets no percentage tolerance per plot). With an interval
on the loss area it adds a third outcome, UNCERTAIN, when the interval straddles the threshold (PROTOCOL.md).
"""
from __future__ import annotations

import numpy as np

CELL_M2 = 900.0
MIN_LOSS_HA = 0.5

FLAGGED = "FLAGGED"
CLEAR = "CLEAR"
UNCERTAIN = "UNCERTAIN"
NO_BASELINE_FOREST = "NO_BASELINE_FOREST"


def min_cells_for(cell_m2: float = CELL_M2, min_loss_ha: float = MIN_LOSS_HA) -> int:
    """Smallest number of whole cells whose area is more than `min_loss_ha` (6 at 30 m: 5,400 m2)."""
    return int(min_loss_ha * 1e4 // cell_m2) + 1


DEFAULT_MIN_CELLS = min_cells_for()


def plot_decision(loss_m2: float, forest_2020_m2: float | None = None, halfwidth_m2: float = 0.0,
                  min_loss_ha: float = MIN_LOSS_HA) -> str:
    """FLAGGED if forest loss is more than `min_loss_ha`, CLEAR if not, UNCERTAIN if loss +/- halfwidth straddles it.

    NO_BASELINE_FOREST when the plot had no forest in the baseline (forest_2020_m2 given and <= 0)."""
    if forest_2020_m2 is not None and forest_2020_m2 <= 0:
        return NO_BASELINE_FOREST
    t = min_loss_ha * 1e4
    if loss_m2 - halfwidth_m2 > t:
        return FLAGGED
    if loss_m2 + halfwidth_m2 <= t:
        return CLEAR
    return UNCERTAIN


def farm_flag(loss_cells, min_cells: int = DEFAULT_MIN_CELLS) -> np.ndarray:
    """Vectorised `plot_decision` on 30 m cell counts with no interval: True where FLAGGED."""
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
