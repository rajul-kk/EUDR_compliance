import os
import random
import re

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

from src.preprocessing.aligner import align_mask_to_image, check_alignment


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
        
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        csv_path = os.path.join(project_root, 'inputs', 'farms_osm.csv')
        if os.path.exists(csv_path):
            self.metadata = pd.read_csv(csv_path)
        else:
            print(f"Warning: Metadata CSV not found at {csv_path}. Filtering will be disabled.")
            self.metadata = None

        self.image_paths = []
        self.mask_map = {}
        
        raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.tiff') or f.endswith('.tif')]
        
        for f in raw_files:
            # Parse filename: {type}_{id}_{year}_{date}.tiff
            match = re.match(r"(relation|way)_(\d+)_(\d{4})_.*\.tiff?", f)
            if match:
                obj_type, obj_id, year = match.groups()
                
                if self.metadata is not None:
                    # Match ID in CSV (e.g. osm_relation_3936516)
                    full_id = f"osm_{obj_type}_{obj_id}"
                    row = self.metadata[self.metadata['farm_id'] == full_id]
                    
                    if not row.empty:
                        crop = row.iloc[0]['crop_type']
                        lat = row.iloc[0]['lat']
                        lon = row.iloc[0]['lon']
                        
                        if exclude_crops and crop in exclude_crops:
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
                    pass
        
        if self.cache_aligned_masks:
            self.aligned_dir = os.path.join(mask_dir, 'aligned_cache')
            os.makedirs(self.aligned_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_map[img_path]
        
        aligned_mask_path = mask_path
        if self.cache_aligned_masks:
            base_name = os.path.basename(img_path) 
            # Use image name for aligned mask to ensure 1-to-1 even if multiple images map to same farm (different dates)
            cache_name = f"{os.path.splitext(base_name)[0]}_mask.tif"
            cache_path = os.path.join(self.aligned_dir, cache_name)
            
            if os.path.exists(cache_path):
                 aligned_mask_path = cache_path
            else:
                 if check_alignment(img_path, mask_path):
                      aligned_mask_path = mask_path
                 else:
                      success = align_mask_to_image(img_path, mask_path, cache_path)
                      if success:
                          aligned_mask_path = cache_path
                      else:
                          # Failed to align, return original (will likely fail downstream or be mismatched)
                          print(f"Warning: Failed to align mask for {base_name}")
                          aligned_mask_path = mask_path
        else:
            aligned_mask_path = mask_path   # without the cache, a misaligned mask is used as it is

        with rasterio.open(img_path) as src:
            image = src.read()
            # Band order in downloaded TIFF: [Red, Green, Blue, NIR, SCL]
            if image.shape[0] < 5:
                raise ValueError(f"Expected at least 5 bands, got {image.shape[0]} in {os.path.basename(img_path)}")
            
            scl = image[4, :, :]
            image = image.astype(np.float32)
            
            red = image[0, :, :]
            nir = image[3, :, :]
            green = image[1, :, :]
            ndvi = (nir - red) / (nir + red + 1e-8)
            ndwi = (green - nir) / (green + nir + 1e-8)  # water mask; negative inside forests

            ndvi = np.expand_dims(ndvi, axis=0)
            ndwi = np.expand_dims(ndwi, axis=0)
            image = np.concatenate([image, ndvi, ndwi], axis=0)  # 7 channels: [R,G,B,NIR,SCL,NDVI,NDWI]

        with rasterio.open(aligned_mask_path) as src:
            mask = src.read(1)
            mask = mask.astype(np.longlong) # Class labels match torch.long for CrossEntropy

        # SCL Values: 0=No Data, 1=Saturated, 3=Cloud Shadow, 8=Medium Cloud, 9=High Cloud, 10=Cirrus
        cloud_pixels = np.isin(scl, [0, 1, 3, 8, 9, 10])
        mask[cloud_pixels] = 255
        
        if self.training:
            if random.random() < 0.5:
                image = image[:, :, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if random.random() < 0.5:
                image = image[:, ::-1, :].copy()
                mask = mask[::-1, :].copy()
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
