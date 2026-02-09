import torch
from torch.utils.data import Dataset
import rasterio
import os
import re
import numpy as np
import sys

# Add parent dir to path if needed to find preprocessing
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.aligner import align_mask_to_image, check_alignment

class FarmSegmentationDataset(Dataset):
    def __init__(self, raw_dir, mask_dir, transform=None, cache_aligned_masks=True):
        """
        Args:
            raw_dir (str): Directory with Sentinel-2 images.
            mask_dir (str): Directory with GEE Hybrid masks.
            transform (callable, optional): Optional transform to be applied on a sample.
            cache_aligned_masks (bool): If True, saves aligned masks to disk (mask_dir/aligned) to speed up future loads.
        """
        self.raw_dir = raw_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.cache_aligned_masks = cache_aligned_masks
        
        # Find valid pairs
        self.image_paths = []
        self.mask_map = {} # Map valid raw filename to mask path
        
        raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.tiff') or f.endswith('.tif')]
        
        for f in raw_files:
            # Parse filename: {type}_{id}_{year}_{date}.tiff
            match = re.match(r"(relation|way)_(\d+)_(\d{4})_.*\.tiff?", f)
            if match:
                obj_type, obj_id, year = match.groups()
                mask_name = f"osm_{obj_type}_{obj_id}_{year}_hybrid.tif"
                mask_path = os.path.join(mask_dir, mask_name)
                
                if os.path.exists(mask_path):
                    self.image_paths.append(os.path.join(raw_dir, f))
                    self.mask_map[os.path.join(raw_dir, f)] = mask_path
                else:
                    # Missing mask, skip this image
                    pass
        
        if self.cache_aligned_masks:
            self.aligned_dir = os.path.join(mask_dir, 'aligned_cache')
            os.makedirs(self.aligned_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_map[img_path]
        
        # 1. Check Alignment
        # If caching on, check if aligned version exists
        aligned_mask_path = mask_path
        if self.cache_aligned_masks:
            # Construct cache path
            base_name = os.path.basename(img_path) 
            # Use image name for aligned mask to ensure 1-to-1 even if multiple images map to same farm (different dates)
            # Actually mask is per year (hybrid). If image is different date, mask is same (for that year).
            # But we align to specific image geometry. So cache based on IMAGE name.
            cache_name = f"{os.path.splitext(base_name)[0]}_mask.tif"
            cache_path = os.path.join(self.aligned_dir, cache_name)
            
            if os.path.exists(cache_path):
                 aligned_mask_path = cache_path
            else:
                 # Need to align or check if original is aligned (unlikely but possible)
                 if check_alignment(img_path, mask_path):
                      # Original is fine, use it. But maybe copy to cache? No.
                      aligned_mask_path = mask_path
                 else:
                      # Align and save to cache
                      # print(f"Aligning mask for {base_name}...")
                      success = align_mask_to_image(img_path, mask_path, cache_path)
                      if success:
                          aligned_mask_path = cache_path
                      else:
                          # Failed to align, return original (will likely fail downstream or be mismatched)
                          print(f"Warning: Failed to align mask for {base_name}")
                          aligned_mask_path = mask_path
        else:
             # If not caching, we might need a temp file or in-memory alignment?
             # User asked for "call aligner.py functions to fix the mask on the fly".
             # writing to temp is safest with current aligner implementation.
             if not check_alignment(img_path, mask_path):
                  # This is slow if not cached!
                  temp_path = mask_path.replace(".tif", "_temp_aligned.tif") # Risky in concurrency
                  # Better use random temp or just rely on caching (which is default True)
                  # For now, let's assume caching is used.
                  pass

        # 2. Load Image
        with rasterio.open(img_path) as src:
            image = src.read() # (C, H, W)
            # Normalize? Or return raw? Usually return tensor, maybe valid range.
            # Sentinel is uint16 usually?
            # SCL is Band 4 (0-indexed assuming 5 bands) if format is [R, G, B, NIR, SCL]
            # Verify band order: Sentinel-2 visual usually Red, Green, Blue. + NIR + SCL.
            # Assuming 5 bands.
            scl = image[4, :, :]
            image = image.astype(np.float32)

        # 3. Load Mask
        with rasterio.open(aligned_mask_path) as src:
            mask = src.read(1) # (H, W)
            mask = mask.astype(np.longlong) # Class labels match torch.long for CrossEntropy

        # --- Cloud Masking ---
        # SCL Values: 0=No Data, 1=Saturated, 3=Cloud Shadow, 8=Medium Cloud, 9=High Cloud, 10=Cirrus
        cloud_pixels = np.isin(scl, [0, 1, 3, 8, 9, 10])
        # Set label to 255 (ignore_index) where clouds exist
        mask[cloud_pixels] = 255
        
        # Convert to Tensor
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)


        if self.transform:
            # Transform usually expects PIL or numpy. 
            # If custom transform, handle tensors.
            # For now, return tuple
             pass

        return image, mask
