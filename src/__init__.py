"""
EUDR Compliance AI - Source Package

This package provides tools for analyzing satellite imagery to verify
EUDR (EU Deforestation Regulation) compliance for agricultural supply chains.
"""

from .change_detector import ChangeDetector, ChangeResult, RiskLevel
from .cloud_filter import CloudFilter
from .sentinel_client import download_all_farms
from .vegetation_index import VegetationIndex

__all__ = [
    "download_all_farms",
    "CloudFilter",
    "VegetationIndex",
    "ChangeDetector",
    "ChangeResult",
    "RiskLevel",
]

__version__ = "0.1.0"
