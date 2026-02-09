import ee
import pandas as pd
import os
import requests
import sys

# Add parent directory to path to import authenticators and config if running directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from auth import initialize_gee
    from config import DYNAMIC_WORLD_ASSET_ID, CANOPY_HEIGHT_ASSET_ID
    from src.fusion_engine import compute_hybrid_classification
except ImportError:
    # Fallback if running from parent dir or installed as package
    from GEE_dynamic.auth import initialize_gee
    from GEE_dynamic.config import DYNAMIC_WORLD_ASSET_ID, CANOPY_HEIGHT_ASSET_ID
    from GEE_dynamic.src.fusion_engine import compute_hybrid_classification

import zipfile
import shutil

def download_file(url, output_path):
    """Downloads a file from a URL. If it's a zip, extracts the TIFF."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save successfully logic
        # We don't know if it is a zip yet from headers sometimes, but GEE usually sends zip
        # Let's save to a temp file
        temp_path = output_path + ".temp"
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        # Check if zip
        if zipfile.is_zipfile(temp_path):
            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                # Look for the .tif inside
                # Usually it's named matching the name param in getDownloadURL + .tif
                # or just look for the first .tif
                tiff_files = [f for f in zip_ref.namelist() if f.endswith('.tif') or f.endswith('.tiff')]
                if tiff_files:
                    # Extract to same dir
                    # But we want to rename it to output_path
                    source_file = zip_ref.extract(tiff_files[0], path=os.path.dirname(output_path))
                    # Move/Rename
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    os.rename(os.path.join(os.path.dirname(output_path), tiff_files[0]), output_path)
                    print(f"Extracted and Saved: {output_path}")
                else:
                    print("No TIFF found in zip.")
            os.remove(temp_path)
        else:
            # Assume it's the file itself (though verify_generation found PK header, so it was a zip)
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_path, output_path)
            print(f"Saved: {output_path}")
            
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def download_mask(lat, lon, farm_id, output_dir):
    """
    Generates hybrid classification for a farm and downloads it as GeoTIFF.
    
    Args:
        lat (float): Latitude of center point.
        lon (float): Longitude of center point.
        farm_id (str): Identifier for the farm.
        output_dir (str): Directory to save the downloaded files.
    """
    point = ee.Geometry.Point([lon, lat])
    # 2.5km buffer around the point
    region = point.buffer(2500).bounds()
    
    # Canopy height is static (2020 version most common), loading once
    # Try to select 'b1' band, but fall back to using image directly if it doesn't exist
    try:
        height_image = ee.Image(CANOPY_HEIGHT_ASSET_ID).select(['b1'], ['height'])
    except:
        # If 'b1' doesn't exist, use the first band or the image as-is
        height_image = ee.Image(CANOPY_HEIGHT_ASSET_ID).rename('height')

    # If the asset is 'users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1', it is often a mosaic. 
    # Let's try to select the first band if unknown, or just use the image if single band.
    # Safest is usually to not select unless we know name. ETH map usually has 1 band (height in meters).
    
    years = [2020, 2024]
    
    for year in years:
        try:
            # Time range for the year
            start_date = f'{year}-01-01'
            end_date = f'{year}-12-31'
            
            # Fetch Dynamic World
            dw_collection = ee.ImageCollection(DYNAMIC_WORLD_ASSET_ID)\
                .filterBounds(region)\
                .filterDate(start_date, end_date)
            
            # Create a mode composite (most common class)
            # The 'label' band contains the class index
            dw_image = dw_collection.select('label').mode().clip(region)
            
            # Compute Hybrid Classification
            # We use the static height image for both years as 2020 is the best baseline
            hybrid_image = compute_hybrid_classification(dw_image, height_image.clip(region))
            
            # Prepare download URL
            # Scale: 10m is native for DW and Sentinel
            url = hybrid_image.getDownloadURL({
                'name': f'{farm_id}_{year}_hybrid',
                'scale': 10,
                'crs': 'EPSG:4326', # WGS84
                'region': region.getInfo() # GeoJSON geometry
            })
            
            # Download file
            filename = f"{farm_id}_{year}_hybrid.tif"
            file_path = os.path.join(output_dir, filename)
            download_file(url, file_path)
            
        except Exception as e:
            print(f"Error processing {farm_id} for {year}: {e}")


def download_mask_years(lat, lon, farm_id, output_dir, years):
    """
    Generates hybrid classification for specific years.
    """
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(2500).bounds()
    
    # Load Canopy Height (Static 2020)
    height_image = ee.Image(CANOPY_HEIGHT_ASSET_ID).select(['b1'], ['height']) 

    for year in years:
        target_file = os.path.join(output_dir, f"{farm_id}_{year}_hybrid.tif")
        if os.path.exists(target_file):
             print(f"   ⏭️ Mask exists for {year}, skipping.")
             continue

        try:
            print(f"   ⚙️ Generating {year} mask...")
            start_date = f'{year}-01-01'
            end_date = f'{year}-12-31'
            
            dw_collection = ee.ImageCollection(DYNAMIC_WORLD_ASSET_ID)\
                .filterBounds(region)\
                .filterDate(start_date, end_date)
            
            # Mosaic method: mode() is good for categorical class stability
            dw_image = dw_collection.select('label').mode().clip(region)
            
            # Compute Hybrid
            hybrid_image = compute_hybrid_classification(dw_image, height_image.clip(region))
            
            url = hybrid_image.getDownloadURL({
                'name': f'{farm_id}_{year}_hybrid',
                'scale': 10,
                'crs': 'EPSG:4326',
                'region': region.getInfo()
            })
            
            download_file(url, target_file)
            
        except Exception as e:
            print(f"   ❌ Error processing {farm_id} {year}: {e}")

def main():
    # Initialize GEE
    initialize_gee()
    
    # Input CSV Path
    csv_path = os.path.join(os.path.dirname(parent_dir), 'inputs', 'farms_osm.csv')
    if not os.path.exists(csv_path):
        # Fallback hardcoded path based on user environment
        csv_path = r'd:\Work\EUDR-compliance\inputs\farms_osm.csv'
    
    # Output Directory
    output_dir = os.path.join(os.path.dirname(parent_dir), 'data', 'hybrid_masks')
    if not os.path.exists(output_dir):
         # Create it
         try:
             os.makedirs(output_dir, exist_ok=True)
         except:
             # Try absolute path fallback
             output_dir = r'd:\Work\EUDR-compliance\data\hybrid_masks'
             os.makedirs(output_dir, exist_ok=True)
             
    print(f"Reading farms from: {csv_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    # --- FILTER: Existing Images + Top 100 Per Crop ---
    
    # helper to check existence
    # helper to check existence
    # HARDCODED FIX to ensure paths are correct
    raw_dir_2020 = r'd:\Work\EUDR-compliance\data\raw_satellite\2020_baseline'
    raw_dir_2024 = r'd:\Work\EUDR-compliance\data\raw_satellite\2024_current'
    
    print(f"DEBUG: Checking for images in:")
    print(f"  2020: {raw_dir_2020}")
    print(f"  2024: {raw_dir_2024}")
    
    import glob

    # 1. Identify valid farms (those with at least one image)
    print("🔍 Scanning local images to build process list...")
    
    # Pre-load all tiff files
    print("   Listing files in 2020 baseline...")
    files_2020 = set(os.path.basename(f) for f in glob.glob(os.path.join(raw_dir_2020, "*.tiff")))
    print(f"   -> Found {len(files_2020)} files. Examples: {list(files_2020)[:3]}")
    
    print("   Listing files in 2024 current...")
    files_2024 = set(os.path.basename(f) for f in glob.glob(os.path.join(raw_dir_2024, "*.tiff")))
    print(f"   -> Found {len(files_2024)} files.")
    
    valid_rows = []
    
    match_count = 0
    for index, row in df.iterrows():
        farm_id = str(row['farm_id']).strip()
        
        # Check if ID is a substring of any filename? 
        # Filename format: {farm_id}_{year}_{date}.tiff
        # Efficient check: 
        # We can reconstruct the farm_id from filenames and store THAT in a set.
        
        # But let's stick to checking if we can find a match.
        # Optimized: Pre-process filenames into a set of IDs
        # Instead of looping, let's do this ONCE before the loop.
        pass

    # Optimization: Extract IDs from filenames
    def extract_ids_from_files(file_list):
        ids = set()
        for fname in file_list:
            # fname: osm_relation_3936516_2020_2020-06-23.tiff
            # Split by '_'
            # But "osm_relation_3936516" contains underscores.
            # Strategy: The ID is everything up to the year? 
            # Or simpler: The ID is the CSV ID.
            # Let's map CSV IDs to files.
            pass
        return ids
    
    # We will just iterate and doing a quick check if we assume consistent naming
    # The safest way given underscores is:
    # Does any file start with f"{farm_id}_"?
    # To avoid O(N*M), let's build a lookup dictionary: {id_prefix: has_file} ? No.
    
    # Let's try to map filenames back to IDs.
    # We know the suffix is like "_2020_YYYY-MM-DD.tiff" or "_2024_YYYY-MM-DD.tiff".
    # We can split by "_" and take parts?
    # No, assuming "osm_way_123" -> 3 parts. "osm_relation_456" -> 3 parts.
    # What if we just iterate the CSV and check?
    
    # Let's build a Set of farm_ids present in the folders.
    present_ids_2020 = set()
    for f in files_2020:
        # Try to match to CSV IDs? No, just heuristic.
        # Remove extension
        name = f.replace('.tiff', '')
        # Remove date and year suffix? 
        # Standard format: {ID}_{YEAR}_{DATE}
        # Split by '_'
        parts = name.split('_')
        # We expect at least YEAR and DATE at end.
        if len(parts) > 2:
            # ID is everything except last 2?
            # e.g. osm_relation_123_2020_2020-06-01 -> osm_relation_123
            guessed_id = "_".join(parts[:-2])
            present_ids_2020.add(guessed_id)
            
    present_ids_2024 = set()
    for f in files_2024:
        name = f.replace('.tiff', '')
        parts = name.split('_')
        if len(parts) > 2:
            guessed_id = "_".join(parts[:-2])
            present_ids_2024.add(guessed_id)
            
    print(f"   -> Identified {len(present_ids_2020)} unique IDs in 2020 folder.")
    print(f"      Samples: {list(present_ids_2020)[:3]}")
    print(f"   -> Identified {len(present_ids_2024)} unique IDs in 2024 folder.")

    # Debug CSV IDs
    print(f"   CSV Sample IDs: {df['farm_id'].head(3).tolist()}")
    
    # helper to normalize ID (strip 'osm_' prefix common in CSV but often missing in files)
    def normalize_id(raw_id):
        if raw_id.startswith("osm_"):
            return raw_id[4:] # Strip "osm_"
        return raw_id
    
    match_count = 0
    valid_rows = []
    
    for index, row in df.iterrows():
        raw_farm_id = str(row['farm_id']).strip()
        search_id = normalize_id(raw_farm_id)
        
        # Check 2020
        # Does any file in files_2020 start with {search_id}_ ?
        # This is O(N*M) but N=2400 and M=600, so 1.4M ops. Fast enough.
        
        has_2020 = any(f.startswith(search_id + "_") for f in files_2020)
        has_2024 = any(f.startswith(search_id + "_") for f in files_2024)
        
        if has_2020 or has_2024:
            row_dict = row.to_dict()
            row_dict['has_2020'] = has_2020
            row_dict['has_2024'] = has_2024
            valid_rows.append(row_dict)
            match_count += 1

    print(f"   -> Matched {match_count} CSV rows to images.")

    if not valid_rows:
        print("⚠️ No downloaded images found. Run the image downloader first.")
        return

    valid_df = pd.DataFrame(valid_rows)
    
    # 2. Limit to Top 100 per crop_type
    print("🔻 Filtering: Keeping max 100 farms per crop type...")
    if 'crop_type' in valid_df.columns:
        final_df = valid_df.groupby('crop_type').head(100).reset_index(drop=True)
        
        # Print breakdown
        counts = final_df['crop_type'].value_counts()
        print(f"   Processing Breakdown:\n{counts}")
    else:
        final_df = valid_df.head(100) # Fallback if no crop_type
        
    print(f"🚀 Starting Mask Generation for {len(final_df)} farms...")
    
    for index, row in final_df.iterrows():
        farm_id = row['farm_id']
        lat = row['lat']
        lon = row['lon']
        
        # Normalize the farm_id to match filename convention
        search_id = normalize_id(farm_id)
        
        # Respect file existence per year
        process_years = []
        if row['has_2020']: process_years.append(2020)
        if row['has_2024']: process_years.append(2024)
        
        print(f"[{index+1}/{len(final_df)}] Farm {farm_id} ({row.get('crop_type','?')}) -> Years: {process_years}")
        
        # Modified download_mask call to accept specific years
        # Pass search_id (normalized) as filenames should match raw images (stripped osm_)
        download_mask_years(lat, lon, search_id, output_dir, process_years)


if __name__ == "__main__":
    main()
