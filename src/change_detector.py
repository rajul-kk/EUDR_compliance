from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from hansen_audit.decisions import MIN_LOSS_HA, plot_decision


class RiskLevel(str, Enum):
    FLAGGED = "FLAGGED"
    CLEAR = "CLEAR"
    UNCERTAIN = "UNCERTAIN"
    NO_BASELINE_FOREST = "NO_BASELINE_FOREST"
    UNASSESSED = "UNASSESSED"
    COMPLIANT = "COMPLIANT"
    WARNING = "WARNING"
    VIOLATION = "VIOLATION"


KNOWN_LEVELS = tuple(level.value for level in RiskLevel)
FLAG_LEVELS = ("FLAGGED", "VIOLATION")

AREA_RULE = "area_0.5ha"
LEGACY_RULE = "legacy_percent"
RULES = (AREA_RULE, LEGACY_RULE)

VIOLATION_THRESHOLD = 10.0  # legacy: % of baseline forest lost -> VIOLATION
WARNING_THRESHOLD = 5.0     # legacy: % of baseline forest lost -> WARNING


def decision_rule(rule: Optional[str] = None) -> str:
    """The rule in force: the argument, else EUDR_DECISION_RULE, else the area rule."""
    r = rule or os.environ.get("EUDR_DECISION_RULE") or AREA_RULE
    if r not in RULES:
        raise ValueError(f"unknown decision rule {r!r}; expected one of {RULES}")
    return r


def classify_forest_loss(forest_2020: int, forest_lost: int):
    """Legacy rule (EUDR_DECISION_RULE=legacy_percent): loss as a share of the farm's 2020 forest, COMPLIANT up to 5 %,
    WARNING above 5 %, VIOLATION above 10 %. The regulation sets no such tolerance (docs/regulation_check.md); kept
    for continuity only. Returns (percent, RiskLevel)."""
    if forest_2020 <= 0:
        return 0.0, RiskLevel.NO_BASELINE_FOREST
    pct = forest_lost / forest_2020 * 100
    if pct > VIOLATION_THRESHOLD:
        return pct, RiskLevel.VIOLATION
    if pct > WARNING_THRESHOLD:
        return pct, RiskLevel.WARNING
    return pct, RiskLevel.COMPLIANT


def classify(forest_2020: int, forest_lost: int, pixel_m2: Optional[float] = None, rule: Optional[str] = None,
             halfwidth_px: float = 0.0):
    """The one alert rule for every model path. Returns (percent, RiskLevel, loss_ha).

    Area rule (default): `hansen_audit.decisions.plot_decision` on the lost area, FLAGGED above 0.5 ha. It needs the
    pixel area; without it the plot is UNASSESSED. The percentage is returned in both cases for information."""
    r = decision_rule(rule)
    pct = forest_lost / forest_2020 * 100 if forest_2020 > 0 else 0.0
    loss_ha = forest_lost * pixel_m2 / 1e4 if pixel_m2 else float("nan")
    if r == LEGACY_RULE:
        return classify_forest_loss(forest_2020, forest_lost) + (loss_ha,)
    if forest_2020 <= 0:
        return 0.0, RiskLevel.NO_BASELINE_FOREST, loss_ha
    if not pixel_m2:
        return pct, RiskLevel.UNASSESSED, loss_ha
    level = plot_decision(forest_lost * pixel_m2, forest_2020 * pixel_m2, halfwidth_px * pixel_m2, MIN_LOSS_HA)
    return pct, RiskLevel(level), loss_ha


def pixel_area_m2(transform, crs, height: int) -> Optional[float]:
    """Ground area of one pixel. Projected CRS: |a * e| in square metres. Geographic CRS: degrees converted at the
    raster's centre latitude (110,574 m per degree of latitude, 111,320 cos(lat) per degree of longitude; under 1 %
    error at these sizes). None when the transform or CRS is missing."""
    if transform is None or crs is None:
        return None
    a, e = float(transform.a), float(transform.e)
    if getattr(crs, "is_geographic", False) is True:
        lat = float(transform.f) + e * height / 2
        return abs(a) * 111_320.0 * math.cos(math.radians(lat)) * abs(e) * 110_574.0
    return abs(a * e)


@dataclass
class ChangeResult:
    farm_id: str
    forest_pixels_2020: int
    forest_pixels_2024: int
    forest_pixels_lost: int
    deforestation_percent: float
    alert_level: RiskLevel
    vector_report: Optional[str] = field(default=None)

    forest_loss_ha: float = float("nan")

    @property
    def is_compliant(self) -> bool:
        """No flag under the rule in force (CLEAR, or COMPLIANT under the legacy rule)."""
        return self.alert_level in (RiskLevel.CLEAR, RiskLevel.COMPLIANT)


class ChangeDetector:
    """Compares baseline (2020) and current (2024) segmentation masks to detect forest loss.

    Forest class label follows the hybrid mask convention:
      0 = Non-vegetation / other
      1 = Forest / dense canopy  ← default forest_class
      2 = Cropland
      3 = Water / wetland
    """

    VIOLATION_THRESHOLD = VIOLATION_THRESHOLD
    WARNING_THRESHOLD = WARNING_THRESHOLD

    def __init__(self, forest_class: int = 1) -> None:
        self.forest_class = forest_class

    def analyze(
        self,
        mask_2020: np.ndarray,
        mask_2024: np.ndarray,
        farm_id: str = "",
        pixel_m2: Optional[float] = None,
        rule: Optional[str] = None,
    ) -> ChangeResult:
        """Compare two segmentation masks and return a ChangeResult.

        Args:
            mask_2020: 2-D integer array of baseline class labels.
            mask_2024: 2-D integer array of current class labels (same shape).
            farm_id: Identifier string for the farm.
            pixel_m2: Ground area of one pixel; the area rule needs it (UNASSESSED without it).
            rule: Decision rule (see `decision_rule`); default the area rule.

        Returns:
            ChangeResult with loss metrics and risk level.
        """
        if mask_2020.shape != mask_2024.shape:
            raise ValueError(
                f"Mask shape mismatch: {mask_2020.shape} vs {mask_2024.shape}"
            )

        forest_2020 = int(np.sum(mask_2020 == self.forest_class))
        forest_2024 = int(np.sum(mask_2024 == self.forest_class))
        lost = max(0, forest_2020 - forest_2024)

        pct, level, loss_ha = classify(forest_2020, lost, pixel_m2, rule)
        pct = round(pct, 2)

        return ChangeResult(
            farm_id=farm_id,
            forest_pixels_2020=forest_2020,
            forest_pixels_2024=forest_2024,
            forest_pixels_lost=lost,
            deforestation_percent=pct,
            alert_level=level,
            forest_loss_ha=loss_ha,
        )

    def batch_analyze(
        self,
        baseline_dir: str,
        prediction_dir: str,
        output_report_path: str,
        vector_dir: Optional[str] = None,
    ):
        """Delegates to detect_deforestation.batch_detect_deforestation.

        Returns the report DataFrame or None if no pairs were found.
        """
        from src.detect_deforestation import batch_detect_deforestation

        return batch_detect_deforestation(
            baseline_dir,
            prediction_dir,
            output_report_path,
            vector_dir=vector_dir,
        )
