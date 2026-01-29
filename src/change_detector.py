"""
Change Detector - Compares 2020 baseline vs current imagery to flag deforestation.
Uses NDVI differencing and threshold analysis.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum
import numpy as np


class RiskLevel(Enum):
    """Risk classification for deforestation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ChangeResult:
    """Results from change detection analysis."""
    farm_id: str
    lat: float
    lon: float
    baseline_ndvi: float
    current_ndvi: float
    ndvi_change: float
    forest_loss_percentage: float
    risk_level: RiskLevel
    is_compliant: bool
    details: str


class ChangeDetector:
    """Detect vegetation changes between two time periods."""
    
    # EUDR cutoff date is December 31, 2020
    EUDR_CUTOFF_DATE = "2020-12-31"
    
    # Risk thresholds for NDVI change (negative = vegetation loss)
    THRESHOLD_LOW = -0.05
    THRESHOLD_MEDIUM = -0.15
    THRESHOLD_HIGH = -0.25
    
    # Maximum acceptable forest loss for EUDR compliance
    MAX_FOREST_LOSS_PERCENT = 5.0
    
    def __init__(self):
        """Initialize the change detector."""
        from .vegetation_index import VegetationIndex
        self.vi_calculator = VegetationIndex()
    
    def calculate_ndvi_change(
        self,
        baseline_ndvi: np.ndarray,
        current_ndvi: np.ndarray
    ) -> np.ndarray:
        """
        Calculate NDVI change between baseline and current.
        
        Negative values indicate vegetation loss.
        Positive values indicate vegetation gain.
        
        Args:
            baseline_ndvi: NDVI array from baseline period (2020)
            current_ndvi: NDVI array from current period
            
        Returns:
            Change array (current - baseline)
        """
        return current_ndvi - baseline_ndvi
    
    def detect_deforestation(
        self,
        baseline_ndvi: np.ndarray,
        current_ndvi: np.ndarray,
        forest_threshold: float = 0.5
    ) -> Tuple[np.ndarray, float]:
        """
        Detect areas where forest has been converted to non-forest.
        
        Args:
            baseline_ndvi: NDVI array from baseline period
            current_ndvi: NDVI array from current period
            forest_threshold: NDVI threshold for forest classification
            
        Returns:
            Tuple of (deforestation mask, deforestation percentage)
        """
        # Create forest masks for both periods
        baseline_forest = self.vi_calculator.is_forested(baseline_ndvi, forest_threshold)
        current_forest = self.vi_calculator.is_forested(current_ndvi, forest_threshold)
        
        # Deforestation = was forest in baseline, not forest now
        deforestation_mask = (baseline_forest == 1) & (current_forest == 0)
        
        # Calculate percentage of baseline forest that was lost
        total_baseline_forest = np.sum(baseline_forest)
        if total_baseline_forest == 0:
            return deforestation_mask, 0.0
        
        deforestation_pct = (np.sum(deforestation_mask) / total_baseline_forest) * 100
        
        return deforestation_mask, deforestation_pct
    
    def classify_risk(self, ndvi_change: float, forest_loss_pct: float) -> RiskLevel:
        """
        Classify the risk level based on vegetation changes.
        
        Args:
            ndvi_change: Mean NDVI change value
            forest_loss_pct: Percentage of forest lost
            
        Returns:
            Risk level classification
        """
        if forest_loss_pct > 20 or ndvi_change < self.THRESHOLD_HIGH:
            return RiskLevel.CRITICAL
        elif forest_loss_pct > 10 or ndvi_change < self.THRESHOLD_MEDIUM:
            return RiskLevel.HIGH
        elif forest_loss_pct > 5 or ndvi_change < self.THRESHOLD_LOW:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def is_eudr_compliant(self, forest_loss_pct: float) -> bool:
        """
        Check if the farm meets EUDR compliance requirements.
        
        EUDR requires that commodities are not sourced from land that
        was deforested after December 31, 2020.
        
        Args:
            forest_loss_pct: Percentage of forest lost since baseline
            
        Returns:
            True if compliant (minimal/no deforestation)
        """
        return forest_loss_pct <= self.MAX_FOREST_LOSS_PERCENT
    
    def analyze_farm(
        self,
        farm_id: str,
        lat: float,
        lon: float,
        baseline_image: np.ndarray,
        current_image: np.ndarray
    ) -> ChangeResult:
        """
        Perform complete change detection analysis for a farm.
        
        Args:
            farm_id: Unique identifier for the farm
            lat: Latitude of farm center
            lon: Longitude of farm center
            baseline_image: Multi-band image from 2020
            current_image: Multi-band image from current period
            
        Returns:
            ChangeResult with complete analysis
        """
        # Extract bands and calculate NDVI
        # Assuming bands are in order: [B2, B3, B4, B8, ...]
        baseline_red = baseline_image[:, :, 2]  # B4
        baseline_nir = baseline_image[:, :, 3]  # B8
        current_red = current_image[:, :, 2]
        current_nir = current_image[:, :, 3]
        
        baseline_ndvi = self.vi_calculator.calculate_ndvi(baseline_red, baseline_nir)
        current_ndvi = self.vi_calculator.calculate_ndvi(current_red, current_nir)
        
        # Calculate changes
        ndvi_change = self.calculate_ndvi_change(baseline_ndvi, current_ndvi)
        mean_ndvi_change = np.nanmean(ndvi_change)
        
        # Detect deforestation
        _, forest_loss_pct = self.detect_deforestation(baseline_ndvi, current_ndvi)
        
        # Classify risk and compliance
        risk_level = self.classify_risk(mean_ndvi_change, forest_loss_pct)
        is_compliant = self.is_eudr_compliant(forest_loss_pct)
        
        # Generate details
        if is_compliant:
            details = f"Farm shows minimal vegetation change ({mean_ndvi_change:.3f} NDVI change). EUDR compliant."
        else:
            details = f"WARNING: Significant vegetation loss detected ({forest_loss_pct:.1f}% forest loss). Requires investigation."
        
        return ChangeResult(
            farm_id=farm_id,
            lat=lat,
            lon=lon,
            baseline_ndvi=float(np.nanmean(baseline_ndvi)),
            current_ndvi=float(np.nanmean(current_ndvi)),
            ndvi_change=float(mean_ndvi_change),
            forest_loss_percentage=float(forest_loss_pct),
            risk_level=risk_level,
            is_compliant=is_compliant,
            details=details
        )
    
    def generate_change_map(
        self,
        ndvi_change: np.ndarray
    ) -> np.ndarray:
        """
        Generate a colorized change map for visualization.
        
        Colors:
        - Red: Significant vegetation loss
        - Yellow: Minor vegetation loss
        - Green: Vegetation gain
        - Gray: No significant change
        
        Args:
            ndvi_change: NDVI change array
            
        Returns:
            RGB array for visualization
        """
        height, width = ndvi_change.shape
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Gray - no change
        no_change = (ndvi_change >= -0.05) & (ndvi_change <= 0.05)
        rgb[no_change] = [128, 128, 128]
        
        # Green - vegetation gain
        gain = ndvi_change > 0.05
        rgb[gain] = [0, 200, 0]
        
        # Yellow - minor loss
        minor_loss = (ndvi_change < -0.05) & (ndvi_change >= -0.15)
        rgb[minor_loss] = [255, 255, 0]
        
        # Red - significant loss
        major_loss = ndvi_change < -0.15
        rgb[major_loss] = [255, 0, 0]
        
        return rgb


if __name__ == "__main__":
    # Example usage with synthetic data
    detector = ChangeDetector()
    
    # Create synthetic baseline and current data
    np.random.seed(42)
    
    # Simulate a scenario with some deforestation
    baseline_ndvi = np.random.uniform(0.5, 0.8, (100, 100))  # Dense vegetation
    current_ndvi = baseline_ndvi.copy()
    current_ndvi[20:40, 30:50] = 0.1  # Deforested area
    
    # Calculate change
    change = detector.calculate_ndvi_change(baseline_ndvi, current_ndvi)
    _, forest_loss = detector.detect_deforestation(baseline_ndvi, current_ndvi)
    
    print(f"Mean NDVI change: {np.nanmean(change):.3f}")
    print(f"Forest loss: {forest_loss:.1f}%")
    print(f"EUDR Compliant: {detector.is_eudr_compliant(forest_loss)}")
