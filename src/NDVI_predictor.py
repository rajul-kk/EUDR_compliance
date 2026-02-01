
import os
import shutil
import numpy as np
import rasterio
import matplotlib.pyplot as plt

def calculate_ndvi(red, nir):
    """
    Calculate NDVI: (NIR - Red) / (NIR + Red)
    Handles division by zero.
    """
    numerator = nir - red
    denominator = nir + red
    
    # Avoid division by zero
    ndvi = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    return ndvi

def generate_debug_mask(ndvi, output_path):
    """
    Save a colormap of the NDVI values for visual inspection.
    """
    plt.figure(figsize=(6, 6))
    # cmap 'RdYlGn' goes from Red (low NDVI) to Green (high NDVI)
    plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(label='NDVI')
    plt.title("NDVI Categorization Mask")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def get_process_stats(folder):
    if not os.path.exists(folder):
        return 0
    return len([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])

def process_images_paired(input_dir, output_base_dir):
    """
    Walks through input_dir finds TIFFs, groups by farm_id,
    and calculates NDVI.
    Output: output_base_dir/{farm_id}/{year}_{label}.tiff
    """
    
    stats = {"forest": 0, "deforested": 0, "needs_review": 0, "error": 0, "processed_farms": 0}
    
    print(f"🚀 Starting Paired Auto-Labeling from: {input_dir}")
    
    # helper to find files
    all_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".tiff"):
                all_files.append(os.path.join(root, file))

    print(f"   📂 Found {len(all_files)} TIFF files total.")

    # Group by Farm ID
    # Filename format expected: {farm_id}_{year}_{date}.tiff
    # e.g. way_12345_2020_2020-06-01.tiff
    
    farm_data = {}
    
    for file_path in all_files:
        basename = os.path.basename(file_path)
        parts = basename.split('_')
        
        # Heuristic: find where the year is to split farm_id
        # Usually farm_id is everything before the year suffix
        
        # Search for 2020 or 2024 in parts
        year_idx = -1
        year = None
        
        for i, part in enumerate(parts):
            if part in ["2020", "2024"]:
                year_idx = i
                year = part
                break
        
        if year:
            farm_id = "_".join(parts[:year_idx])
            
            if farm_id not in farm_data:
                farm_data[farm_id] = {}
            
            farm_data[farm_id][year] = file_path

    print(f"   🚜 Identified {len(farm_data)} unique farms.")
    
    # Process each farm
    for farm_id, years in farm_data.items():
        
        # Create Farm Folder
        farm_dir = os.path.join(output_base_dir, farm_id)
        
        # Check if already done (optimization)
        # If folder exists and has contents, we can skip? 
        # User requested duplicate check, let's look inside.
        if os.path.exists(farm_dir):
            existing = os.listdir(farm_dir)
            if len(existing) >= 2: # heuristic: at least 2 files (image + mask?)
                 # Actually verify if we have coverage for the years present in input
                 # But simplistic check: if folder exists, duplicate check per file below
                 pass
        
        os.makedirs(farm_dir, exist_ok=True)
        
        stats["processed_farms"] += 1

        for year, file_path in years.items():
             try:
                base_name = os.path.basename(file_path)
                
                with rasterio.open(file_path) as src:
                    if src.count < 4:
                        print(f"⚠️ Skipping {base_name}: Not enough bands ({src.count})")
                        stats["error"] += 1
                        continue
                        
                    red = src.read(1).astype('float32')
                    nir = src.read(4).astype('float32')
                    
                    ndvi = calculate_ndvi(red, nir)
                    avg_ndvi = np.mean(ndvi)
                    
                    # Categorize
                    label = "needs_review"
                    if avg_ndvi >= 0.6:
                        label = "forest"
                    elif avg_ndvi <= 0.3:
                        label = "deforested"
                    
                    stats[label] += 1
                    
                    # New Filename: {year}_{label}.tiff
                    # If we have multiple dates for same year/farm, we might conflict.
                    # Appending date from original filename to be safe if needed.
                    # Original: way_123_2020_2020-06-01.tiff
                    # Let's keep it simple as requested: {year}_{label}.tiff 
                    # BUT checks for collisions if multiple images per year.
                    
                    date_part = base_name.replace(f"{farm_id}_{year}_", "").replace(".tiff", "")
                    
                    # Target Name: {year}_{label}.tiff
                    # To allow multiple images per year, maybe: {year}_{date}_{label}.tiff
                    target_name = f"{year}_{label}.tiff"
                    
                    dest_file = os.path.join(farm_dir, target_name)
                    mask_file = os.path.join(farm_dir, f"{year}_{label}_mask.png")
                    
                    # DUPLICATE CHECK
                    if os.path.exists(dest_file):
                        # print(f"   ⏭️ {farm_id}: {year} already labeled as {label}")
                        continue

                    # Copy
                    shutil.copy2(file_path, dest_file)
                    
                    # Mask
                    generate_debug_mask(ndvi, mask_file)
                    
                    print(f"[{label.upper()}] {farm_id} ({year}): Avg NDVI {avg_ndvi:.2f}")
                    
             except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                stats["error"] += 1

    print("\n🎉 Paired Auto-Labeling Complete!")
    print(f"🚜 Farms Processed: {stats['processed_farms']}")
    print(f"🌲 Forest Images: {stats['forest']}")
    print(f"🪓 Deforested Images: {stats['deforested']}")
    print(f"👀 Needs Review: {stats['needs_review']}")
    print(f"⚠️ Errors: {stats['error']}")

def clean_old_labels(label_dir):
    """
    Deletes the old directory structure (forest/deforested/needs_review) 
    checking if it exists first.
    """
    if os.path.exists(label_dir):
        print(f"\n🧹 Cleaning up old labeled directory: {label_dir}")
        try:
            shutil.rmtree(label_dir)
            print("   ✅ Deleted.")
        except Exception as e:
            print(f"   ❌ Failed to delete: {e}")

if __name__ == "__main__":
    input_folder = "data/raw_satellite"
    
    # New Output Folder
    output_folder_paired = "data/labeled_paired"
    
    # Old Output Folder to Delete
    output_folder_old = "data/labeled"
    
    # 1. Run New Process
    process_images_paired(input_folder, output_folder_paired)
    
    # 2. Cleanup Old
    clean_old_labels(output_folder_old)
