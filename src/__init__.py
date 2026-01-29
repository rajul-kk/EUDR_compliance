"""
EUDR Compliance AI - Source Package

This package provides tools for analyzing satellite imagery to verify
EUDR (EU Deforestation Regulation) compliance for agricultural supply chains.
"""

from .sentinel_client import SentinelClient
from .cloud_filter import CloudFilter
from .vegetation_index import VegetationIndex
from .change_detector import ChangeDetector, ChangeResult, RiskLevel

__all__ = [
    "SentinelClient",
    "CloudFilter",
    "VegetationIndex",
    "ChangeDetector",
    "ChangeResult",
    "RiskLevel",
]

__version__ = "0.1.0"
