
import cv2
import numpy as np


def refine_mask(mask, kernel_size=3):
    """
    Apply morphological opening and closing to refine the segmentation mask.
    Removes small noise and fills small holes.
    
    Args:
        mask: Numpy array (H, W) with class labels
        kernel_size: Size of the structuring element
        
    Returns:
        refined_mask: Refined numpy array
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # We apply this per class or on the binary forest mask?
    # Usually better on the binary forest mask for deforestation detection.

    # Example: If forest_class=1
    forest_mask = (mask == 1).astype(np.uint8)

    # Opening: Removal of small objects
    refined = cv2.morphologyEx(forest_mask, cv2.MORPH_OPEN, kernel)
    # Closing: Filling of small holes
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)

    # Put back into mask
    # This assumes we only care about forest refinement for now
    refined_mask = mask.copy()
    refined_mask[mask == 1] = 0 # Clear old forest
    refined_mask[refined == 1] = 1 # Set refined forest

    return refined_mask
