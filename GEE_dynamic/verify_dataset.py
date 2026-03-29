import torch
from preprocessing.dataset_loader import FarmSegmentationDataset
import os
import sys

# Ensure imports work
sys.path.append(os.getcwd())

raw_dir = r'd:\Work\EUDR-compliance\data\raw_satellite\2020_baseline'
mask_dir = r'd:\Work\EUDR-compliance\data\hybrid_masks'

print("Initializing Dataset...")
dataset = FarmSegmentationDataset(raw_dir, mask_dir, cache_aligned_masks=True)

print(f"Dataset Size: {len(dataset)}")

if len(dataset) > 0:
    print("Testing __getitem__ for first 3 items...")
    for i in range(min(5, len(dataset))): # Checking more items to find a cloud
        try:
            img, mask = dataset[i]
            print(f"Item {i}: Image Shape {img.shape}, Mask Shape {mask.shape}")
            
            # Check shapes
            if img.shape[1:] == mask.shape:
                print(" -> Shapes Match ✅")
            else:
                print(" -> Shapes Mismatch ❌")
                
            # Check values
            unique_vals = torch.unique(mask)
            print(f" -> Mask Values: {unique_vals}")
            if 255 in unique_vals:
                print(" -> Cloud Masking Active (Found 255) ☁️")
            else:
                print(" -> No Clouds Detected in this sample")
            
        except Exception as e:
            print(f"Error loading item {i}: {e}")
else:
    print("Dataset empty. Check paths matching logic.")
