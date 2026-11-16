import os
import sys

import ee
import pandas as pd
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if __package__ in (None, ""):   # run by path (python GEE_dynamic/src/x.py): make the repo root importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import zipfile

from GEE_dynamic.auth import initialize_gee
from GEE_dynamic.config import CANOPY_HEIGHT_ASSET_ID, DYNAMIC_WORLD_ASSET_ID
from GEE_dynamic.src.fusion_engine import compute_hybrid_classification


def download_file(url, output_path):
    """Downloads a file from a URL. If it's a zip, extracts the TIFF."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # We don't know if it is a zip yet from headers sometimes, but GEE usually sends zip
        temp_path = output_path + ".temp"
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        if zipfile.is_zipfile(temp_path):
            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                tiff_files = [f for f in zip_ref.namelist() if f.endswith('.tif') or f.endswith('.tiff')]
                if tiff_files:
                    zip_ref.extract(tiff_files[0], path=os.path.dirname(output_path))
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    os.rename(os.path.join(os.path.dirname(output_path), tiff_files[0]), output_path)
                    print(f"Extracted and Saved: {output_path}")
                else:
                    print("No TIFF found in zip.")
            os.remove(temp_path)
        else:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_path, output_path)
            print(f"Saved: {output_path}")
            
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def download_mask_years(lat, lon, farm_id, output_dir, years):
    """
    Generates hybrid classification for specific years.
    """
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(2500).bounds()
    
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
    initialize_gee()
    
    root = os.path.dirname(parent_dir)
    csv_path = os.path.join(root, 'inputs', 'farms_osm.csv')
    output_dir = os.path.join(root, 'data', 'hybrid_masks')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading farms from: {csv_path}")
    print(f"Output directory: {output_dir}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        sys.exit(f"Failed to read CSV: {e}")

    raw_dir_2020 = os.path.join(root, 'data', 'raw_satellite', '2020_baseline')
    raw_dir_2024 = os.path.join(root, 'data', 'raw_satellite', '2024_current')
    
    print("DEBUG: Checking for images in:")
    print(f"  2020: {raw_dir_2020}")
    print(f"  2024: {raw_dir_2024}")
    
    import glob

    print("🔍 Scanning local images to build process list...")
    
    print("   Listing files in 2020 baseline...")
    files_2020 = set(os.path.basename(f) for f in glob.glob(os.path.join(raw_dir_2020, "*.tiff")))
    print(f"   -> Found {len(files_2020)} files. Examples: {list(files_2020)[:3]}")
    
    print("   Listing files in 2024 current...")
    files_2024 = set(os.path.basename(f) for f in glob.glob(os.path.join(raw_dir_2024, "*.tiff")))
    print(f"   -> Found {len(files_2024)} files.")
    
    present_ids_2020 = set()
    for f in files_2020:
        name = f.replace('.tiff', '')
        parts = name.split('_')
        if len(parts) > 2:
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

    print(f"   CSV Sample IDs: {df['farm_id'].head(3).tolist()}")
    
    # helper to normalize ID (strip 'osm_' prefix common in CSV but often missing in files)
    def normalize_id(raw_id):
        if raw_id.startswith("osm_"):
            return raw_id[4:]
        return raw_id
    
    match_count = 0
    valid_rows = []
    
    for index, row in df.iterrows():
        raw_farm_id = str(row['farm_id']).strip()
        search_id = normalize_id(raw_farm_id)
        
        
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
        sys.exit("No downloaded images found. Run the image downloader first.")

    valid_df = pd.DataFrame(valid_rows)
    
    print("🔻 Filtering: Keeping max 100 farms per crop type...")
    if 'crop_type' in valid_df.columns:
        final_df = valid_df.groupby('crop_type').head(100).reset_index(drop=True)
        
        counts = final_df['crop_type'].value_counts()
        print(f"   Processing Breakdown:\n{counts}")
    else:
        final_df = valid_df.head(100)
        
    print(f"🚀 Starting Mask Generation for {len(final_df)} farms...")
    
    for index, row in final_df.iterrows():
        farm_id = row['farm_id']
        lat = row['lat']
        lon = row['lon']
        
        search_id = normalize_id(farm_id)
        
        process_years = []
        if row['has_2020']:
            process_years.append(2020)
        if row['has_2024']:
            process_years.append(2024)
        
        print(f"[{index+1}/{len(final_df)}] Farm {farm_id} ({row.get('crop_type','?')}) -> Years: {process_years}")
        
        download_mask_years(lat, lon, search_id, output_dir, process_years)


if __name__ == "__main__":
    main()
