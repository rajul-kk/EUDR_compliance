import ee
try:
    from GEE_dynamic.config import HEIGHT_THRESHOLD
except ImportError:
    try:
        from config import HEIGHT_THRESHOLD
    except ImportError:
         # Fallback for relative import if run as a module within src
        from ..config import HEIGHT_THRESHOLD


def get_height_mask(image):
    """
    Returns a boolean mask where canopy height > HEIGHT_THRESHOLD.
    
    Args:
        image (ee.Image): The canopy height image (or image with height band).
        
    Returns:
        ee.Image: A boolean mask (1 where height > threshold, 0 otherwise).
    """
    return image.gt(HEIGHT_THRESHOLD)
