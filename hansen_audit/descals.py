"""PROTOCOL_E13: Hansen loss against the Descals et al. (2024) oil palm planting-year layer. Pure functions.

Planting year (YoP) tiles: the year the current (2021) oil palm plantation was established, 1990-2021, plus two values the
record does not define, read per the owner's decision (PROTOCOL_E13 deviation, 15 Nov): 1989 = planted in 1989 or earlier,
2022 = its own category, never palm by 2020. On each label grid the share of a cell covered by each category is an
area-weighted average of its indicator; a cell is "palm by 2020" when the share of 1989 and 1990-2020 together is at least
PALM_SHARE (sensitivity: 1990-2020 alone).
"""
import numpy as np

from .decisions import farm_flag

FIRST_YEAR, CUTOFF_YEAR, LAST_YEAR = 1990, 2020, 2021
EARLY, LATE = 1989, 2022          # undocumented values, read as above
CATEGORIES = ("y1989", "y1990_2020", "y2021", "y2022")
PALM_SHARE = 0.5


def encoding_ok(values, nodata, extra=(EARLY, LATE)) -> tuple[bool, str]:
    """PROTOCOL_E13 encoding check on the distinct tile values: planting years 1990-2021, the owner-read `extra` values, and
    at most one other value (zero or the declared no-data) for land that is not palm."""
    vals = {int(v) for v in values}
    years = {v for v in vals if FIRST_YEAR <= v <= LAST_YEAR}
    other = vals - years - set(extra)
    if not years:
        return False, "no planting years found"
    if len(other) > 1 or (other and not other <= {0, None if nodata is None else int(nodata)}):
        return False, f"unexpected non-year values {sorted(other)[:10]}"
    return True, f"years {min(years)}-{max(years)}, read as extra {sorted(set(extra) & vals)}, non-palm value {sorted(other)}"


def indicators(yop: np.ndarray) -> dict:
    """{category: float32 0/1 array} for CATEGORIES, ready for average resampling."""
    yop = np.asarray(yop)
    masks = {"y1989": yop == EARLY, "y1990_2020": (yop >= FIRST_YEAR) & (yop <= CUTOFF_YEAR),
             "y2021": yop == LAST_YEAR, "y2022": yop == LATE}
    return {k: m.astype(np.float32) for k, m in masks.items()}


def palm_by_2020(shares: dict, threshold: float = PALM_SHARE, count_1989: bool = True) -> np.ndarray:
    """Bool per cell from percentage shares (0-100) by category: palm planted by 2020."""
    s = np.asarray(shares["y1990_2020"], float) + (np.asarray(shares["y1989"], float) if count_1989 else 0.0)
    return s >= threshold * 100


def palm_share_at_least(share_pct: np.ndarray, threshold: float = PALM_SHARE) -> np.ndarray:
    return np.asarray(share_pct, float) >= threshold * 100


def farm_counts(labels: np.ndarray, palm2020: np.ndarray, palm2021: np.ndarray, palm2022=None) -> dict:
    """Loss-cell counts per farm: all, palm by 2020, palm planted 2021 (and 2022 if given); label no-data excluded."""
    loss = (np.asarray(labels) == 2).ravel()
    out = {"class2": int(loss.sum()), "class2_palm2020": int((loss & np.asarray(palm2020, bool).ravel()).sum()),
           "class2_palm2021": int((loss & np.asarray(palm2021, bool).ravel()).sum())}
    if palm2022 is not None:
        out["class2_palm2022"] = int((loss & np.asarray(palm2022, bool).ravel()).sum())
    return out


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
