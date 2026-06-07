import logging
import os
import time

import osmnx as ox
import pandas as pd

logger = logging.getLogger(__name__)

# Use a reliable Overpass mirror — overpass-api.de times out frequently
ox.settings.overpass_url = "https://overpass.kumi.systems/api/interpreter"

# CONFIGURATION
# Extensive list of regions to ensure we fill the dataset
# CONFIGURATION
# Optimized regions (Town/Municipality level) to avoid timeouts
# and ensuring better OSM data availability
TARGETS = {
    "Oil Palm": [
        {"region": "Minas, Siak, Riau, Indonesia", "tags": {"landuse": "farmland"}},
        {"region": "Mandau, Bengkalis, Riau, Indonesia", "tags": {"landuse": "farmland"}},
        {"region": "Kotawaringin Barat, Kalimantan Tengah, Indonesia", "tags": {"landuse": "farmland"}},
        {"region": "Muaro Jambi, Jambi, Indonesia", "tags": {"landuse": "farmland"}},
        {"region": "Ujung Batu, Rokan Hulu, Riau, Indonesia", "tags": {"landuse": "farmland"}},
        {"region": "Keningau, Sabah, Malaysia", "tags": {"landuse": "farmland"}},
        {"region": "Beaufort, Sabah, Malaysia", "tags": {"landuse": "farmland"}},
    ],
    "Cocoa": [
        {"region": "San Pedro, Ivory Coast", "tags": {"landuse": "farmland"}},
        {"region": "Soubre, Ivory Coast", "tags": {"landuse": "farmland"}},
        {"region": "Issia, Ivory Coast", "tags": {"landuse": "farmland"}},
        {"region": "Kumasi Metropolitan, Ashanti, Ghana", "tags": {"landuse": "farmland"}},
        {"region": "Ilheus, Bahia, Brazil", "tags": {"landuse": "farmland"}}
    ],
    "Coffee": [
        {"region": "Manhuaçu, Minas Gerais, Brazil", "tags": {"landuse": "farmland"}},
        {"region": "Pitalito, Huila, Colombia", "tags": {"landuse": "farmland"}},
        {"region": "Buon Ho, Dak Lak, Vietnam", "tags": {"landuse": "farmland"}},
        {"region": "Armenia, Quindio, Colombia", "tags": {"landuse": "orchard"}},
        {"region": "Jimma, Ethiopia", "tags": {"landuse": "farmland"}},
        {"region": "Coorg, India", "tags": {"landuse": "plantation"}},
        {"region": "Boquete, Panama", "tags": {"landuse": "farmland"}},
        {"region": "Alajuela, Costa Rica", "tags": {"landuse": "farmland"}},
        {"region": "Nyeri, Kenya", "tags": {"landuse": "farmland"}},
        {"region": "Bener Meriah, Aceh, Indonesia", "tags": {"landuse": "farmland"}}
    ],
    "Cattle": [
        {"region": "Alta Floresta, Mato Grosso, Brazil", "tags": {"landuse": "meadow"}},
        {"region": "Ji-Parana, Rondonia, Brazil", "tags": {"landuse": "meadow"}},
        {"region": "Amarillo, Texas, USA", "tags": {"landuse": "meadow"}},
        {"region": "Rockhampton, Queensland, Australia", "tags": {"landuse": "meadow"}},
        {"region": "La Pampa, Argentina", "tags": {"landuse": "meadow"}},
        {"region": "Cork, Ireland", "tags": {"landuse": "meadow"}},
        # New Regions
        {"region": "Greeley, Colorado, USA", "tags": {"landuse": "farm"}}, # Feedlots often marked as farm/industrial
        {"region": "Garden City, Kansas, USA", "tags": {"landuse": "meadow"}},
        {"region": "Uberaba, Minas Gerais, Brazil", "tags": {"landuse": "meadow"}},
        {"region": "Waikato, New Zealand", "tags": {"landuse": "meadow"}},
        {"region": "Omaheke, Namibia", "tags": {"landuse": "meadow"}}
    ],
    "Rubber": [
        {"region": "Surat Thani, Thailand", "tags": {"landuse": "farmland"}},
        {"region": "Loc Ninh, Binh Phuoc, Vietnam", "tags": {"landuse": "farmland"}},
        {"region": "Hat Yai District, Songkhla, Thailand", "tags": {"landuse": "farmland"}},
        {"region": "Kottayam, India", "tags": {"landuse": "farmland"}},
        {"region": "Bong County, Liberia", "tags": {"landuse": "plantation"}},
        # New Regions
        {"region": "Xishuangbanna, China", "tags": {"landuse": "plantation"}},
        {"region": "North Sumatra, Indonesia", "tags": {"landuse": "plantation"}},
        {"region": "Phuket, Thailand", "tags": {"landuse": "farmland"}},
        {"region": "Mon, Myanmar", "tags": {"landuse": "plantation"}},
        {"region": "Edo State, Nigeria", "tags": {"landuse": "plantation"}}
    ],
    "Soy": [
        # Keeping existing
        {"region": "Sorriso, Mato Grosso, Brazil", "tags": {"landuse": "farmland"}},
        {"region": "Rio Verde, Goias, Brazil", "tags": {"landuse": "farmland"}},
        {"region": "Casilda, Santa Fe, Argentina", "tags": {"landuse": "farmland"}},
        {"region": "Ames, Iowa, USA", "tags": {"landuse": "farmland"}},
        {"region": "Toledo, Parana, Brazil", "tags": {"landuse": "farmland"}},
        {"region": "Braila, Romania", "tags": {"landuse": "farmland"}},
    ],
}

TARGET_COUNT_PER_CROP = {
    "Oil Palm": 200,
    "Cocoa":    250,
    "Coffee":   250,
    "Soy":      200,
    "Cattle":   200,
    "Rubber":   150,
}
OUTPUT_FILE = "inputs/farms_osm.csv"
GLOBAL_TIMEOUT_SEC = 1200 # 20 Minutes Limit

def build_farm_csv():
    logger.info("Starting multi-crop scouting")
    start_time = time.time()

    if not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0:
        pd.DataFrame(columns=["farm_id", "lat", "lon", "crop_type"]).to_csv(OUTPUT_FILE, index=False)
        existing_ids = set()
        counts = {}
        logger.info("Created new farm database file")
    else:
        try:
            df_exist = pd.read_csv(OUTPUT_FILE)
            if "farm_id" in df_exist.columns:
                existing_ids = set(df_exist["farm_id"].astype(str))
                counts = df_exist["crop_type"].value_counts().to_dict()
                logger.info("Loaded %d existing farms | counts=%s", len(existing_ids), counts)
            else:
                existing_ids = set()
                counts = {}
        except Exception:
            existing_ids = set()
            counts = {}

    total_added = 0

    for crop, regions_list in TARGETS.items():
        if time.time() - start_time > GLOBAL_TIMEOUT_SEC:
            logger.warning("Global time limit (%ds) reached — stopping", GLOBAL_TIMEOUT_SEC)
            break

        current_count = counts.get(crop, 0)
        target = TARGET_COUNT_PER_CROP.get(crop, 200)
        needed = target - current_count

        if needed <= 0:
            logger.info("%s: already have %d (target %d) — skipping", crop, current_count, target)
            continue

        logger.info("Scouting for %s (need %d)", crop, needed)

        for info in regions_list:
            if needed <= 0:
                break
            if time.time() - start_time > GLOBAL_TIMEOUT_SEC:
                break

            region = info["region"]
            tags = info["tags"]
            logger.debug("Checking %s", region)

            try:
                gdf = ox.features_from_place(region, tags=tags)
                logger.debug("Found %d candidates in %s", len(gdf), region)

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
                    logger.debug("Appended %d farms from %s", len(batch_farms), region)

            except Exception as e:
                logger.warning("OSM query failed for %s: %s", region, e)

        logger.info("%s finished — new total: %d", crop, target - needed)

    logger.info("Farm discovery complete | total new farms added: %d", total_added)

if __name__ == "__main__":
    build_farm_csv()
