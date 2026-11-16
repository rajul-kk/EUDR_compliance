
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


    forest_mask = (mask == 1).astype(np.uint8)

    refined = cv2.morphologyEx(forest_mask, cv2.MORPH_OPEN, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)

    refined_mask = mask.copy()
    refined_mask[mask == 1] = 0
    refined_mask[refined == 1] = 1

    return refined_mask
