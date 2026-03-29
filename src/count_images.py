
import os
import pandas as pd
import glob

def count_images():
    # Paths
    csv_path = "inputs/farms_osm.csv"
    dir_2020 = "data/raw_satellite/2020_baseline"
    dir_2024 = "data/raw_satellite/2024_current"
    report_path = "count_report.md"

    # 1. Read Farms CSV
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} farms from CSV.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Ensure crop_type column exists
    if 'crop_type' not in df.columns:
        print("'crop_type' column missing in CSV.")
        return

    # 2. Get Farm IDs present in image directories
    
    def get_farm_ids_from_dir(directory):
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return set()
        
        files = glob.glob(os.path.join(directory, "*.tiff"))
        farm_ids = set()
        for f in files:
            basename = os.path.basename(f)
            # Assuming format: farm_id_date.tiff
            # Split by "_" from the right once
            parts = basename.rsplit("_", 1)
            if len(parts) == 2:
                # Extracted ID is like 'relation_17898310_2020'
                # We need to remove the trailing _2020 or _2024
                extracted_id = parts[0]
                if extracted_id.endswith("_2020"):
                    farm_ids.add(extracted_id[:-5])
                elif extracted_id.endswith("_2024"):
                    farm_ids.add(extracted_id[:-5])
                else:
                    farm_ids.add(extracted_id)
            else:
                 print(f"Skipping odd filename: {basename}")
        
        return farm_ids

    ids_2020 = get_farm_ids_from_dir(dir_2020)
    ids_2024 = get_farm_ids_from_dir(dir_2024)

    print(f"Found {len(ids_2020)} images in 2020 folder.")
    print(f"Found {len(ids_2024)} images in 2024 folder.")

    # 3. Map Farm IDs to Crop Types (Normalize IDs first)
    
    def process_id(raw_id):
        return str(raw_id).replace("osm_", "").replace("('", "").replace("', ", "_").replace(")", "")

    df['processed_id'] = df['farm_id'].apply(process_id)

    # 4. Count
    df['has_2020'] = df['processed_id'].isin(ids_2020)
    df['has_2024'] = df['processed_id'].isin(ids_2024)

    # Group by crop_type and sum
    summary = df.groupby('crop_type')[['has_2020', 'has_2024']].sum().reset_index()
    summary.rename(columns={'has_2020': 'Count 2020', 'has_2024': 'Count 2024'}, inplace=True)
    
    # Add Total Row
    total_2020 = summary['Count 2020'].sum()
    total_2024 = summary['Count 2024'].sum()
    
    output_str = "\n=== Image Headcount by Crop Type ===\n"
    output_str += summary.to_string(index=False)
    output_str += f"\n\nTotal 2020: {total_2020}\n"
    output_str += f"Total 2024: {total_2024}\n"
    
    print(output_str)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Image Headcount Report\n\n")
        f.write("```\n")
        f.write(summary.to_string(index=False))
        f.write("\n```")
        f.write(f"\n\n**Total 2020**: {total_2020}\n")
        f.write(f"**Total 2024**: {total_2024}\n")

if __name__ == "__main__":
    count_images()
