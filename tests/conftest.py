"""Stub out native deps so tests run without torch/rasterio/GDAL installed.

A package is stubbed only when it is not installed. Stubbing an installed torch made
torch.utils.data.Dataset a MagicMock, so every subclass of it (ChangeDetectionDataset)
became a mock too, and the metric and model tests ran against mocks.
"""
import importlib.machinery
import importlib.util
import sys
from unittest.mock import MagicMock

_STUBS = [
    "torch", "torch.nn", "torch.nn.functional", "torch.optim",
    "torch.utils", "torch.utils.data",
    "torchvision", "torchvision.models", "torchvision.models.segmentation",
    "rasterio", "rasterio.warp", "rasterio.features",
    "affine", "cv2", "ee", "s2cloudless", "geotessera",
    "geopandas", "osmnx", "lightgbm",
    "sklearn", "sklearn.metrics", "sklearn.model_selection",
    "pystac_client", "dask", "dask.delayed",
    "folium", "reportlab", "reportlab.lib", "reportlab.lib.colors",
    "reportlab.lib.pagesizes", "reportlab.lib.styles",
    "reportlab.lib.units", "reportlab.platypus",
    "skimage", "skimage.morphology", "skimage.exposure",
]

def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name.split(".")[0]) is not None
    except (ImportError, ValueError):
        return False


for mod in _STUBS:
    if not _installed(mod):
        stub = MagicMock()
        stub.__spec__ = importlib.machinery.ModuleSpec(mod, None)
        sys.modules.setdefault(mod, stub)

sys.path.insert(0, ".")
sys.path.insert(0, "src")
