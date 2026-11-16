import ee

from GEE_dynamic.config import (
    CLASS_CROPS,
    CLASS_FOREST,
    CLASS_OTHER,
    CLASS_SHRUB,
    DW_CROPS_LABEL,
    DW_SHRUB_LABEL,
    DW_TREES_LABEL,
    HEIGHT_THRESHOLD,
)


def compute_hybrid_classification(dw_image, height_image):
    """
    Computes the hybrid classification based on Dynamic World and Canopy Height.
    
    Logic:
    - Class 1 (True Forest): DW='Trees' AND Height > HEIGHT_THRESHOLD
    - Class 2 (Crops/Plantation): DW='Crops' OR (DW='Trees' AND Height <= HEIGHT_THRESHOLD)
    - Class 3 (Shrubbery): DW='Shrub and scrub'
    - Class 0 (Other): Everything else
    
    Args:
        dw_image (ee.Image): Dynamic World image (class band).
        height_image (ee.Image): Canopy height image.
        
    Returns:
        ee.Image: Reclassified image.
    """
    
    is_forest = dw_image.eq(DW_TREES_LABEL).And(height_image.gt(HEIGHT_THRESHOLD))
    
    is_crops = dw_image.eq(DW_CROPS_LABEL).Or(
        dw_image.eq(DW_TREES_LABEL).And(height_image.lte(HEIGHT_THRESHOLD))
    )
    
    is_shrub = dw_image.eq(DW_SHRUB_LABEL)
    
    classification = ee.Image.constant(CLASS_OTHER).toByte()
    
    
    classification = classification.where(is_shrub, CLASS_SHRUB)
    classification = classification.where(is_crops, CLASS_CROPS)
    classification = classification.where(is_forest, CLASS_FOREST)
    
    return classification.rename('hybrid_class')
