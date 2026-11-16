"""PROTOCOL_E13: Hansen loss against the Descals et al. (2024) oil palm planting-year layer. Pure functions.

Planting year (YoP) tiles: the year oil palm was planted, 1990-2021, on palm; anything else is not palm. On each label grid
the share of a cell covered by palm planted in or before 2020 (and, separately, in 2021) is an area-weighted average of the
indicator; a cell is "palm by 2020" when that share is at least PALM_SHARE.
"""
import numpy as np

from .decisions import farm_flag

FIRST_YEAR, CUTOFF_YEAR, LAST_YEAR = 1990, 2020, 2021
PALM_SHARE = 0.5
NODATA = 255


def encoding_ok(values, nodata) -> tuple[bool, str]:
    """PROTOCOL_E13 encoding check on the distinct values seen in the tiles: planting years 1990-2021 plus at most one
    other value (zero or the declared no-data) for land that is not palm."""
    vals = {int(v) for v in values}
    years = {v for v in vals if FIRST_YEAR <= v <= LAST_YEAR}
    other = vals - years
    if not years:
        return False, "no planting years found"
    if len(other) > 1 or (other and not other <= {0, None if nodata is None else int(nodata)}):
        return False, f"unexpected non-year values {sorted(other)[:10]}"
    return True, f"years {min(years)}-{max(years)}, non-palm value {sorted(other)}"


def indicators(yop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(palm planted 1990-2020, palm planted 2021) as float32 0/1 arrays, ready for average resampling."""
    yop = np.asarray(yop)
    by2020 = (yop >= FIRST_YEAR) & (yop <= CUTOFF_YEAR)
    return by2020.astype(np.float32), (yop == LAST_YEAR).astype(np.float32)


def palm(share: np.ndarray, threshold: float = PALM_SHARE) -> np.ndarray:
    return np.asarray(share) >= threshold


def farm_counts(labels: np.ndarray, palm2020: np.ndarray, palm2021: np.ndarray) -> dict:
    """Loss-cell counts per farm: all, palm by 2020, palm planted 2021 (label no-data excluded)."""
    loss = (np.asarray(labels) == 2).ravel()
    p20, p21 = np.asarray(palm2020, bool).ravel(), np.asarray(palm2021, bool).ravel()
    return {"class2": int(loss.sum()), "class2_palm2020": int((loss & p20).sum()), "class2_palm2021": int((loss & p21).sum())}


def gated_flags(loss_cells, loss_palm2020_cells):
    """(Hansen flag, Descals-gated flag): the gated count leaves out loss cells that were palm by 2020."""
    loss = np.asarray(loss_cells)
    return farm_flag(loss), farm_flag(loss - np.asarray(loss_palm2020_cells))


def map_reading(share) -> str:
    """PROTOCOL_E13 analysis 3: share of palm-by-2020 loss cells that GFC2020 calls non-forest."""
    if share is None:
        return "too few cells"
    if share >= 0.7:
        return "the two maps agree that this loss was not on 2020 forest"
    if share <= 0.5:
        return "they disagree"
    return "mixed"
