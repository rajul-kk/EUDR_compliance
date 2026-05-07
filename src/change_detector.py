from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class RiskLevel(str, Enum):
    COMPLIANT = "COMPLIANT"
    WARNING = "WARNING"
    VIOLATION = "VIOLATION"
    NO_BASELINE_FOREST = "NO_BASELINE_FOREST"


@dataclass
class ChangeResult:
    farm_id: str
    forest_pixels_2020: int
    forest_pixels_2024: int
    forest_pixels_lost: int
    deforestation_percent: float
    alert_level: RiskLevel
    vector_report: Optional[str] = field(default=None)

    @property
    def is_compliant(self) -> bool:
        return self.alert_level == RiskLevel.COMPLIANT


class ChangeDetector:
    """Compares baseline (2020) and current (2024) segmentation masks to detect forest loss.

    Forest class label follows the hybrid mask convention:
      0 = Non-vegetation / other
      1 = Forest / dense canopy  ← default forest_class
      2 = Cropland
      3 = Water / wetland
    """

    VIOLATION_THRESHOLD = 10.0  # % forest loss → VIOLATION
    WARNING_THRESHOLD = 5.0     # % forest loss → WARNING

    def __init__(self, forest_class: int = 1) -> None:
        self.forest_class = forest_class

    def analyze(
        self,
        mask_2020: np.ndarray,
        mask_2024: np.ndarray,
        farm_id: str = "",
    ) -> ChangeResult:
        """Compare two segmentation masks and return a ChangeResult.

        Args:
            mask_2020: 2-D integer array of baseline class labels.
            mask_2024: 2-D integer array of current class labels (same shape).
            farm_id: Identifier string for the farm.

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

        if forest_2020 == 0:
            pct = 0.0
            level = RiskLevel.NO_BASELINE_FOREST
        else:
            pct = round((lost / forest_2020) * 100, 2)
            if pct > self.VIOLATION_THRESHOLD:
                level = RiskLevel.VIOLATION
            elif pct > self.WARNING_THRESHOLD:
                level = RiskLevel.WARNING
            else:
                level = RiskLevel.COMPLIANT

        return ChangeResult(
            farm_id=farm_id,
            forest_pixels_2020=forest_2020,
            forest_pixels_2024=forest_2024,
            forest_pixels_lost=lost,
            deforestation_percent=pct,
            alert_level=level,
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
