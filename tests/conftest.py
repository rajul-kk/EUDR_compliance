"""Stub out native deps so tests run without torch/rasterio/GDAL installed."""
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

for mod in _STUBS:
    sys.modules.setdefault(mod, MagicMock())

sys.path.insert(0, ".")
sys.path.insert(0, "src")
