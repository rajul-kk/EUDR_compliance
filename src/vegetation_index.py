"""Re-exports VegetationIndex from the NDVI_pred module for use within the src package."""
from __future__ import annotations

import importlib.util
import os

# Load by file path so the module name never collides with this file's own name,
# which would cause a self-import cycle when src/ is on sys.path.
_ndvi_path = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "NDVI_pred", "vegetation_index.py")
)
_spec = importlib.util.spec_from_file_location("ndvi_pred.vegetation_index", _ndvi_path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
VegetationIndex = _mod.VegetationIndex

__all__ = ["VegetationIndex"]
