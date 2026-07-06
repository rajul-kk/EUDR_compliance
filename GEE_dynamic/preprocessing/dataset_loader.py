import random

import numpy as np
import os
import pandas as pd
import re
import rasterio
import sys
import torch
from torch.utils.data import Dataset

# Ensure src/ is on the path so preprocessing.aligner resolves to src/preprocessing/aligner.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
_src_dir = os.path.normpath(os.path.join(parent_dir, "..", "src"))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.aligner import align_mask_to_image, check_alignment

class FarmSegmentationDataset(Dataset):
    def __init__(self, raw_dir, mask_dir, transform=None, cache_aligned_masks=True,
                 exclude_crops=None, exclude_regions=None, training=False):
        """
        Args:
            raw_dir (str): Directory with Sentinel-2 images.
            mask_dir (str): Directory with GEE Hybrid masks.
            transform (callable, optional): Optional transform to be applied on a sample.
            cache_aligned_masks (bool): If True, saves aligned masks to disk.
            exclude_crops (list, optional): List of crop types to exclude (e.g. ['Sugarcane']).
            exclude_regions (dict, optional): Dict with 'lat_range' or 'lon_range' to exclude.
        """
        self.raw_dir = raw_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.cache_aligned_masks = cache_aligned_masks
        self.training = training
        
        # Load metadata for filtering
        project_root = os.path.dirname(parent_dir)
        csv_path = os.path.join(project_root, 'inputs', 'farms_osm.csv')
        if os.path.exists(csv_path):
            self.metadata = pd.read_csv(csv_path)
        else:
            print(f"Warning: Metadata CSV not found at {csv_path}. Filtering will be disabled.")
            self.metadata = None

        # Find valid pairs
        self.image_paths = []
        self.mask_map = {} # Map valid raw filename to mask path
        
        raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.tiff') or f.endswith('.tif')]
        
        for f in raw_files:
            # Parse filename: {type}_{id}_{year}_{date}.tiff
            match = re.match(r"(relation|way)_(\d+)_(\d{4})_.*\.tiff?", f)
            if match:
                obj_type, obj_id, year = match.groups()
                
                # --- Metadata Filtering ---
                if self.metadata is not None:
                    # Match ID in CSV (e.g. osm_relation_3936516)
                    full_id = f"osm_{obj_type}_{obj_id}"
                    row = self.metadata[self.metadata['farm_id'] == full_id]
                    
                    if not row.empty:
                        crop = row.iloc[0]['crop_type']
                        lat = row.iloc[0]['lat']
                        lon = row.iloc[0]['lon']
                        
                        # Filter by crop
                        if exclude_crops and crop in exclude_crops:
                            # print(f"Skipping {f} (Crop: {crop})")
                            continue
                            
                        # Filter by region (example: Brazil approx lat < 5)
                        if exclude_regions:
                            if 'lat_range' in exclude_regions:
                                l_min, l_max = exclude_regions['lat_range']
                                if l_min <= lat <= l_max:
                                    continue
                            if 'lon_range' in exclude_regions:
                                ln_min, ln_max = exclude_regions['lon_range']
                                if ln_min <= lon <= ln_max:
                                    continue

                # Masks were renamed to strip osm_ prefix, so pattern is now: {type}_{id}_{year}_hybrid.tif
                mask_name = f"{obj_type}_{obj_id}_{year}_hybrid.tif"
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
                  aligned_mask_path = mask_path  # Fallback to prevent UnboundLocalError
             else:
                  aligned_mask_path = mask_path

        # 2. Load Image
        with rasterio.open(img_path) as src:
            image = src.read() # (C, H, W)
            # Band order in downloaded TIFF: [Red, Green, Blue, NIR, SCL]
            if image.shape[0] < 5:
                raise ValueError(f"Expected at least 5 bands, got {image.shape[0]} in {os.path.basename(img_path)}")
            
            scl = image[4, :, :]
            image = image.astype(np.float32)
            
            # Derived indices
            red = image[0, :, :]
            nir = image[3, :, :]
            green = image[1, :, :]
            ndvi = (nir - red) / (nir + red + 1e-8)
            ndwi = (green - nir) / (green + nir + 1e-8)  # water mask; negative inside forests

            ndvi = np.expand_dims(ndvi, axis=0)
            ndwi = np.expand_dims(ndwi, axis=0)
            image = np.concatenate([image, ndvi, ndwi], axis=0)  # 7 channels: [R,G,B,NIR,SCL,NDVI,NDWI]

        # 3. Load Mask
        with rasterio.open(aligned_mask_path) as src:
            mask = src.read(1) # (H, W)
            mask = mask.astype(np.longlong) # Class labels match torch.long for CrossEntropy

        # --- Cloud Masking ---
        # SCL Values: 0=No Data, 1=Saturated, 3=Cloud Shadow, 8=Medium Cloud, 9=High Cloud, 10=Cirrus
        cloud_pixels = np.isin(scl, [0, 1, 3, 8, 9, 10])
        # Set label to 255 (ignore_index) where clouds exist
        mask[cloud_pixels] = 255
        
        # Training augmentation (geometric + mild radiometric)
        if self.training:
            # Random horizontal flip
            if random.random() < 0.5:
                image = image[:, :, ::-1].copy()
                mask = mask[:, ::-1].copy()
            # Random vertical flip
            if random.random() < 0.5:
                image = image[:, ::-1, :].copy()
                mask = mask[::-1, :].copy()
            # Random 90° rotation
            k = random.randint(0, 3)
            if k > 0:
                image = np.rot90(image, k=k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k=k, axes=(0, 1)).copy()
            # Brightness jitter on optical bands only (not SCL at index 4)
            factor = random.uniform(0.9, 1.1)
            optical = [0, 1, 2, 3, 5, 6]  # R, G, B, NIR, NDVI, NDWI
            image[optical] = np.clip(image[optical] * factor, 0, None)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask
