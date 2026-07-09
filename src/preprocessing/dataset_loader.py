"""Re-exports FarmSegmentationDataset from GEE_dynamic for src-package consumers."""
import importlib.util as _ilu
import os as _os

_loader_path = _os.path.normpath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "GEE_dynamic", "preprocessing", "dataset_loader.py")
)
_spec = _ilu.spec_from_file_location("gee_dynamic.preprocessing.dataset_loader", _loader_path)
_mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
FarmSegmentationDataset = _mod.FarmSegmentationDataset

__all__ = ["FarmSegmentationDataset"]
