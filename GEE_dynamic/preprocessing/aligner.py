
import rasterio
from rasterio.warp import reproject, Resampling
import os
import glob
import re
import numpy as np


def align_mask_to_image(master_path, slave_path, output_path):
    """
    Reprojects and aligns the slave image (mask) to match the master image (satellite) exactly.
    
    Args:
        master_path (str): Path to the Sentinel-2 image (Master).
        slave_path (str): Path to the GEE Hybrid Mask (Slave).
        output_path (str): Output path for the aligned mask.
    """
    try:
        with rasterio.open(master_path) as master:
            master_transform = master.transform
            master_width = master.width
            master_height = master.height
            master_crs = master.crs
            master_profile = master.profile

        with rasterio.open(slave_path) as slave:
            source_data = slave.read(1)
            source_transform = slave.transform
            source_crs = slave.crs

            # Prepare output array
            # We assume mask is single band (or we fuse/flatten before calling this, but here it's typically 1 band)
            # Sentinel might be multi-band, but we just need spatial alignment so shape matches
            destination_data = np.zeros((master_height, master_width), dtype=rasterio.uint8) # Class labels fit in uint8

            # Update output profile
            output_profile = master_profile.copy()
            output_profile.update({
                'driver': 'GTiff',
                'height': master_height,
                'width': master_width,
                'transform': master_transform,
                'crs': master_crs,
                'count': 1, # Mask is single band
                'dtype': rasterio.uint8,
                'nodata': 0 # Assuming 0 is Other or Nodata, or we keep what came in if compatible
            })
            
            # Reproject
            reproject(
                source=rasterio.band(slave, 1),
                destination=destination_data,
                src_transform=source_transform,
                src_crs=source_crs,
                dst_transform=master_transform,
                dst_crs=master_crs,
                resampling=Resampling.nearest # Vital for categorical data
            )
            
            # Save
            with rasterio.open(output_path, 'w', **output_profile) as dst:
                dst.write(destination_data, 1)
                
            print(f"Aligned: {os.path.basename(output_path)}")
            return True

    except Exception as e:
        print(f"Alignment failed for {os.path.basename(slave_path)} -> {os.path.basename(master_path)}: {e}")
        return False




def batch_align_masks(raw_dir, mask_dir, output_dir):
    """
    Aligns all matching masks in mask_dir to images in raw_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Pattern for raw files: {type}_{id}_{year}_{date}.tiff
    # Example: way_12345_2020_2020-06-01.tiff
    # Farm ID in mask: osm_{type}_{id}  (Note the prefix 'osm_')
    # Or just {type}_{id}?
    # Let's check the verify_generation output: osm_relation_17898310_2020_hybrid.tif
    # But inputs/farms_osm.csv had ids like "osm_relation_..."
    # Wait, the filenames in raw_save lists: "relation_10018408_2020_2020-06-07.tiff"
    # So raw files DON'T have "osm_" prefix usually? Or is "relation" the type? 
    # Let's look at list_dir output from Step 205:
    # "relation_10741190_2020_2020-06-14.tiff"
    # "way_1013629879_2020_2020-06-29.tiff"
    
    # Mask filenames from Step 189:
    # "osm_relation_17898310_2020_hybrid.tif"
    
    # So Mapping is:
    # Raw: "{type}_{id}_{year}_{date}.tiff"
    # Mask: "osm_{type}_{id}_{year}_hybrid.tif"
    
    raw_files = glob.glob(os.path.join(raw_dir, "*.tiff")) + glob.glob(os.path.join(raw_dir, "*.tif"))
    
    print(f"Found {len(raw_files)} raw Sentinel images.")
    
    for raw_path in raw_files:
        filename = os.path.basename(raw_path)
        
        # Regex to parse raw filename
        # Expecting: (type)_(id)_(year)_(date).tiff
        # e.g. relation_10741190_2020_2020-06-14.tiff
        match = re.match(r"(relation|way)_(\d+)_(\d{4})_.*\.tiff?", filename)
        
        if match:
            obj_type, obj_id, year = match.groups()
            
            # Construct expected mask filename
            # osm_relation_10741190_2020_hybrid.tif
            mask_filename = f"osm_{obj_type}_{obj_id}_{year}_hybrid.tif"
            mask_path = os.path.join(mask_dir, mask_filename)
            
            if os.path.exists(mask_path):
                # We align specifically to this image
                output_filename = f"{os.path.splitext(filename)[0]}_mask_aligned.tif"
                output_path = os.path.join(output_dir, output_filename)
                
                # Check if already exists to save time? Or overwrite. 
                # Let's overwrite or skip. Overwrite is safer for dev.
                align_mask_to_image(raw_path, mask_path, output_path)
            else:
                 # It's possible we downloaded a mask but didn't have raw data, or vice versa.
                 # Actually raw data exists (we are iterating it). Mask might be missing if generation failed.
                 # Or maybe existing raw data naming convention is slightly different.
                 # Let's log if not found? No, too noisy if datasets mismatch.
                 pass
        else:
            print(f"Skipping unrecognized filename format: {filename}")

if __name__ == "__main__":
    # For testing, we can point to specific dirs
    # But usually this is imported.
    # We can add a simple CLI runner
    pass

def check_alignment(master_path, slave_path):
    """
    Checks if slave is aligned to master (same transform, dimensions, CRS).
    Returns True if aligned, False otherwise.
    """
    try:
        with rasterio.open(master_path) as master:
            m_profile = master.profile
            m_transform = master.transform
            m_crs = master.crs
            m_width = master.width
            m_height = master.height
            
        with rasterio.open(slave_path) as slave:
            s_transform = slave.transform
            s_crs = slave.crs
            s_width = slave.width
            s_height = slave.height

        # Compare critical attributes
        # CRS could be equivalent but different string representation, maybe skip strict CRS check if transform and shape match?
        # But safest is to check everything.
        # But allow small float diffs in transform? rasterio transform equality handles this usually.
        
        if (m_width != s_width) or (m_height != s_height):
            return False
        if (m_crs != s_crs):
             # Try simple equality first. If fails, we align.
             return False
        if (m_transform != s_transform):
            return False
            
        return True
    except Exception as e:
        print(f"Error checking alignment: {e}")
        return False
