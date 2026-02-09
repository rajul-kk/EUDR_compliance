import ee
try:
    from GEE_dynamic.config import (
        HEIGHT_THRESHOLD,
        DW_TREES_LABEL,
        DW_CROPS_LABEL,
        DW_SHRUB_LABEL,
        CLASS_FOREST,
        CLASS_CROPS,
        CLASS_SHRUB,
        CLASS_OTHER
    )
except ImportError:
    try:
        from config import (
            HEIGHT_THRESHOLD,
            DW_TREES_LABEL,
            DW_CROPS_LABEL,
            DW_SHRUB_LABEL,
            CLASS_FOREST,
            CLASS_CROPS,
            CLASS_SHRUB,
            CLASS_OTHER
        )
    except ImportError:
         # Fallback for relative import
        from ..config import (
            HEIGHT_THRESHOLD,
            DW_TREES_LABEL,
            DW_CROPS_LABEL,
            DW_SHRUB_LABEL,
            CLASS_FOREST,
            CLASS_CROPS,
            CLASS_SHRUB,
            CLASS_OTHER
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
    
    # 1. True Forest: DW is Trees AND Height > Threshold
    is_forest = dw_image.eq(DW_TREES_LABEL).And(height_image.gt(HEIGHT_THRESHOLD))
    
    # 2. Crops/Plantation: DW is Crops OR (DW is Trees AND Height <= Threshold)
    is_crops = dw_image.eq(DW_CROPS_LABEL).Or(
        dw_image.eq(DW_TREES_LABEL).And(height_image.lte(HEIGHT_THRESHOLD))
    )
    
    # 3. Shrubbery: DW is Shrub
    is_shrub = dw_image.eq(DW_SHRUB_LABEL)
    
    # 4. Combine into a single image
    # Start with Other (0)
    classification = ee.Image.constant(CLASS_OTHER).toByte()
    
    # Update with classes based on conditions (Order matters if masks overlap, but here cases are mutually exclusive mostly)
    # Actually, is_forest and is_crops for 'Trees' case are mutually exclusive by height threshold.
    # is_shrub is exclusive to others by DW label.
    
    classification = classification.where(is_shrub, CLASS_SHRUB)
    classification = classification.where(is_crops, CLASS_CROPS)
    classification = classification.where(is_forest, CLASS_FOREST)
    
    # Rename the band for clarity
    return classification.rename('hybrid_class')
