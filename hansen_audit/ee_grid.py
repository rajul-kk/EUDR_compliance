"""Earth Engine pixels on an exact local grid (PROTOCOL_E9, PROTOCOL_E10)."""
import numpy as np

CRS = "EPSG:4326"


def compute_pixels(img, transform, shape, crs: str = CRS) -> np.ndarray:
    """ee.data.computePixels of `img` on the grid (affine a, b, c, d, e, f; shape H, W): a NumPy structured array, one field
    per band. The grid's CRS must be given explicitly; every label raster here is EPSG:4326."""
    import ee

    a, b, c, d, e, f = transform
    return ee.data.computePixels({
        "expression": img, "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": int(shape[1]), "height": int(shape[0])},
                 "affineTransform": {"scaleX": a, "shearX": b, "translateX": c, "shearY": d, "scaleY": e, "translateY": f},
                 "crsCode": crs}})
