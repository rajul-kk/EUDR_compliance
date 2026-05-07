"""Re-exports VegetationIndex from the NDVI_pred module for use within the src package."""
from __future__ import annotations

import os
import sys

_ndvi_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "NDVI_pred")
if _ndvi_dir not in sys.path:
    sys.path.insert(0, os.path.normpath(_ndvi_dir))

from vegetation_index import VegetationIndex  # noqa: E402  (path set above)

__all__ = ["VegetationIndex"]
