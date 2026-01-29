import osmnx as ox
import pandas as pd

# CONFIGURATION
# Pick a region (e.g., a province in Italy or a district in Brazil)
REGION = "Vercelli, Italy" 
TAGS = {"landuse": "farmland"} # We want things tagged as farms

def build_farm_csv():
    print(f"🌍 Scouting for farms in {REGION}...")
    
    # 1. Download geometries from OpenStreetMap
    gdf = ox.features_from_place(REGION, tags=TAGS)
    
    print(f"✅ Found {len(gdf)} farm plots.")
    
    # 2. Extract Center Points (Centroids)
    # We only need one GPS point per farm to aim the satellite
    farms = []
    
    for idx, row in gdf.iterrows():
        # Get the center of the shape
        centroid = row.geometry.centroid
        
        farms.append({
            "farm_id": f"osm_{idx}", # Unique ID from OSM
            "lat": centroid.y,
            "lon": centroid.x
        })
        
    # 3. Save to CSV
    df = pd.DataFrame(farms)
    
    # Optional: Limit to first 50 to save disk space
    df = df.head(50)
    
    output_file = "inputs/farms_osm.csv"
    df.to_csv(output_file, index=False)
    
    print(f"🎉 Saved {len(df)} farms to {output_file}")
    print("Now run 'bulk_download.py' using this file!")

if __name__ == "__main__":
    build_farm_csv()