import osmnx as ox
import pandas as pd
import os
import time

# CONFIGURATION
# Extensive list of regions to ensure we fill the dataset
# CONFIGURATION
# Optimized regions (Town/Municipality level) to avoid timeouts
# and ensuring better OSM data availability
TARGETS = {
    "Cattle": [
        {"region": "Alta Floresta, Mato Grosso, Brazil", "tags": {"landuse": "meadow"}},
        {"region": "Ji-Parana, Rondonia, Brazil", "tags": {"landuse": "meadow"}},
        {"region": "Amarillo, Texas, USA", "tags": {"landuse": "meadow"}}, 
        {"region": "Rockhampton, Queensland, Australia", "tags": {"landuse": "meadow"}},
        {"region": "La Pampa, Argentina", "tags": {"landuse": "meadow"}}, # New Backup
        {"region": "Cork, Ireland", "tags": {"landuse": "meadow"}} # New Backup
    ],
    "Rubber": [
        {"region": "Surat Thani, Thailand", "tags": {"landuse": "farmland"}},
        {"region": "Binh Phuoc, Vietnam", "tags": {"landuse": "plantation"}},
        {"region": "Hat Yai, Thailand", "tags": {"landuse": "farmland"}},
        {"region": "Kottayam, India", "tags": {"landuse": "farmland"}},
        {"region": "Bong County, Liberia", "tags": {"landuse": "plantation"}} # Firestone plantation
    ],
    "Rice": [
        {"region": "Vercelli, Italy", "tags": {"landuse": "farmland"}}
    ],
    "Soy": [
        {"region": "Sorriso, Mato Grosso, Brazil", "tags": {"landuse": "farmland"}},
        {"region": "Rio Verde, Goias, Brazil", "tags": {"landuse": "farmland"}},
        {"region": "Venado Tuerto, Argentina", "tags": {"landuse": "farmland"}},
        {"region": "Ames, Iowa, USA", "tags": {"landuse": "farmland"}},
        {"region": "Toledo, Parana, Brazil", "tags": {"landuse": "farmland"}}
    ],
    "Coffee": [
        {"region": "Manhuaçu, Minas Gerais, Brazil", "tags": {"landuse": "farmland"}},
        {"region": "Pitalito, Huila, Colombia", "tags": {"landuse": "farmland"}}, 
        {"region": "Buon Ma Thuot, Vietnam", "tags": {"landuse": "farm"}},
        {"region": "Chinchina, Caldas, Colombia", "tags": {"landuse": "farmland"}},
        {"region": "Jimma, Ethiopia", "tags": {"landuse": "farmland"}}
    ],
    "Cocoa": [
        {"region": "San Pedro, Ivory Coast", "tags": {"landuse": "orchard"}},
        {"region": "Soubré, Ivory Coast", "tags": {"landuse": "orchard"}},
        {"region": "Kumasi, Ghana", "tags": {"landuse": "farmland"}},
        {"region": "Ilheus, Bahia, Brazil", "tags": {"landuse": "farmland"}}
    ],
    "Oil Palm": [
        {"region": "Pekanbaru, Indonesia", "tags": {"landuse": "farmland"}},
        {"region": "Sandakan, Malaysia", "tags": {"landuse": "plantation"}},
        {"region": "Lahad Datu, Malaysia", "tags": {"landuse": "plantation"}},
        {"region": "Jambi, Indonesia", "tags": {"landuse": "farmland"}}
    ]
}

TARGET_COUNT_PER_CROP = 600
OUTPUT_FILE = "inputs/farms_osm.csv"
GLOBAL_TIMEOUT_SEC = 600 # 10 Minutes Limit

def build_farm_csv():
    print(f"🌍 Starting Multi-Crop scouting with FALLBACKS...")
    start_time = time.time()
    
    # Initialize/Load logic
    if not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0:
        pd.DataFrame(columns=["farm_id", "lat", "lon", "crop_type"]).to_csv(OUTPUT_FILE, index=False)
        existing_ids = set()
        print("   Created new database file.")
    else:
        try:
            df_exist = pd.read_csv(OUTPUT_FILE)
            if "farm_id" in df_exist.columns:
                existing_ids = set(df_exist["farm_id"].astype(str))
                # Count current per crop
                counts = df_exist["crop_type"].value_counts().to_dict()
                print(f"📖 Loaded {len(existing_ids)} existing farms.")
                print(f"   Current counts: {counts}")
            else:
                existing_ids = set()
                counts = {}
        except:
            existing_ids = set()
            counts = {}

    total_added = 0

    for crop, regions_list in TARGETS.items():
        if time.time() - start_time > GLOBAL_TIMEOUT_SEC:
            print(f"⏰ Global time limit ({GLOBAL_TIMEOUT_SEC}s) reached. Stopping.")
            break

        current_count = counts.get(crop, 0)
        needed = TARGET_COUNT_PER_CROP - current_count
        
        if needed <= 0:
            print(f"\n✅ {crop}: Already have {current_count} (Target {TARGET_COUNT_PER_CROP}). Skipping.")
            continue
            
        print(f"\n🔎 Scouting for {crop} (Need {needed})...")
        
        for info in regions_list:
            if needed <= 0:
                break
            if time.time() - start_time > GLOBAL_TIMEOUT_SEC:
                break
                
            region = info["region"]
            tags = info["tags"]
            print(f"   📍 Checking {region}...")
            
            try:
                gdf = ox.features_from_place(region, tags=tags)
                print(f"      ✅ Found {len(gdf)} candidates.")
                
                batch_farms = []
                for idx, row in gdf.iterrows():
                    if needed <= 0:
                        break
                    
                    farm_id = f"osm_{idx}".replace("('", "").replace("', ", "_").replace(")", "")
                    
                    if farm_id in existing_ids:
                        continue 
                    
                    centroid = row.geometry.centroid
                    batch_farms.append({
                        "farm_id": farm_id,
                        "lat": centroid.y,
                        "lon": centroid.x,
                        "crop_type": crop
                    })
                    existing_ids.add(farm_id)
                    needed -= 1
                    total_added += 1

                if batch_farms:
                    pd.DataFrame(batch_farms).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
                    print(f"      💾 Appended {len(batch_farms)} farms.")
                
            except Exception as e:
                print(f"      ⚠️ Failed/Empty: {e}")
                
        print(f"   📦 {crop} finished. New total: {TARGET_COUNT_PER_CROP - needed}")

    print(f"\n🎉 Process Complete!")
    print(f"   Total NEW farms added: {total_added}")

if __name__ == "__main__":
    build_farm_csv()