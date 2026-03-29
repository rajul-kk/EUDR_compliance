import time
import os
import glob
import json
import random
import argparse
import threading
import pandas as pd
import requests
import dask
from dask import delayed
from dotenv import load_dotenv
from pystac_client import Client

load_dotenv()

# CONFIGURATION
USERNAME = os.getenv("CDSE_EMAIL")
PASSWORD = os.getenv("CDSE_PASSWORD")

# API ENDPOINTS
AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL = "https://stac.dataspace.copernicus.eu/v1"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# GLOBAL TOKEN CACHE
ACCESS_TOKEN = None
TOKEN_EXPIRY = 0
STATE_LOCK = threading.Lock()

DEFAULT_PROFILE = {
    "default": {
        "month_windows": [["05-01", "09-30"]],
        "max_cloud": 15
    },
    "tropical": {
        "month_windows": [["01-01", "12-31"]],
        "max_cloud": 20
    }
}

EU_COUNTRY_CODES = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR",
    "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
    "SI", "ES", "SE"
}


def request_with_retry(method, url, max_retries=4, retryable_statuses=None, timeout=60, **kwargs):
    """HTTP wrapper with bounded retries for transient API failures."""
    if retryable_statuses is None:
        retryable_statuses = {429, 500, 502, 503, 504}

    for attempt in range(max_retries):
        try:
            response = requests.request(method=method, url=url, timeout=timeout, **kwargs)
            if response.status_code in retryable_statuses:
                raise requests.HTTPError(f"Transient API status {response.status_code}", response=response)
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = min(30, (2 ** attempt) + random.random())
            print(f"⚠️ Request failed ({e}). Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)


def normalize_farm_id(raw_id):
    return str(raw_id).replace("osm_", "").replace("('", "").replace("', ", "_").replace(")", "")


def load_json_file(path, default):
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load JSON {path}: {e}. Using default.")
        return default


def save_json_file(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_country_code(row):
    if "country_iso2" not in row:
        return ""
    value = str(row.get("country_iso2", "")).strip().upper()
    return value[:2] if value else ""


def get_profile_for_country(country_code, profile):
    if country_code and country_code in EU_COUNTRY_CODES:
        return profile.get("default", DEFAULT_PROFILE["default"])
    return profile.get("tropical", profile.get("default", DEFAULT_PROFILE["default"]))


def get_year_windows(year, country_code, profile):
    cfg = get_profile_for_country(country_code, profile)
    windows = cfg.get("month_windows", DEFAULT_PROFILE["default"]["month_windows"])
    year_windows = []
    for start_mmdd, end_mmdd in windows:
        year_windows.append((f"{year}-{start_mmdd}", f"{year}-{end_mmdd}"))
    return year_windows


def get_farm_bbox(row, delta=0.005):
    """Build AOI bbox from row columns. Uses bbox columns when present, else point buffer."""
    lat = float(row["lat"])
    lon = float(row["lon"])

    min_lon = row.get("min_lon")
    min_lat = row.get("min_lat")
    max_lon = row.get("max_lon")
    max_lat = row.get("max_lat")

    if pd.notna(min_lon) and pd.notna(min_lat) and pd.notna(max_lon) and pd.notna(max_lat):
        return [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]

    return [lon - delta, lat - delta, lon + delta, lat + delta]


def update_manifest(manifest_path, farm_id, year, status, message=""):
    """Store per-farm/year state for resume-safe downloads."""
    with STATE_LOCK:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

        if os.path.exists(manifest_path):
            df = pd.read_csv(manifest_path)
        else:
            df = pd.DataFrame(columns=["farm_id", "year", "status", "message", "updated_at"])

        mask = (df["farm_id"] == farm_id) & (df["year"] == int(year))
        now_ts = pd.Timestamp.utcnow().isoformat()

        if mask.any():
            df.loc[mask, ["status", "message", "updated_at"]] = [status, message, now_ts]
        else:
            df = pd.concat([
                df,
                pd.DataFrame([
                    {
                        "farm_id": farm_id,
                        "year": int(year),
                        "status": status,
                        "message": message,
                        "updated_at": now_ts,
                    }
                ])
            ], ignore_index=True)

        df.to_csv(manifest_path, index=False)

def get_auth_token():
    global ACCESS_TOKEN, TOKEN_EXPIRY
    
    current_time = time.time()
    
    # 1. Return cached token if valid (buffer of 60s)
    if ACCESS_TOKEN and current_time < (TOKEN_EXPIRY - 60):
        return ACCESS_TOKEN
        
    print("   🔑 Refreshing Access Token...")
    
    payload = {
        "client_id": "cdse-public",
        "username": USERNAME,
        "password": PASSWORD,
        "grant_type": "password",
    }
    
    for attempt in range(3):
        try:
            response = request_with_retry("POST", AUTH_URL, data=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            ACCESS_TOKEN = data["access_token"]
            # Default to 600s if expires_in not provided
            expires_in = data.get("expires_in", 600) 
            TOKEN_EXPIRY = current_time + expires_in
            
            return ACCESS_TOKEN
            
        except Exception as e:
            if attempt < 2:
                wait_time = (attempt + 1) * 2
                print(f"⚠️ Auth Attempt {attempt+1} failed ({e}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ Auth Failed after 3 attempts: {e}")
                if hasattr(e, 'response') and e.response:
                    print(f"Server Response: {e.response.text}")
                raise e

def find_cleanest_date(bbox, start_date, end_date, max_cloud=15):
    """
    Step 1: Search the Catalog to find the EXACT date of a clear flyover.
    """
    print(f"🔎 Searching for clear images between {start_date} and {end_date}...")
    
    client = Client.open(CATALOG_URL)
    
    # Search for Sentinel-2 Level-2A (Cloud corrected)
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}"
    )
    
    items = list(search.items())
    
    # Filter for low clouds
    clean_items = [i for i in items if i.properties.get("eo:cloud_cover", 100) < max_cloud]
    
    if not clean_items:
        print("⚠️ No cloud-free images found in this range. Trying all images...")
        clean_items = items
        
    if not clean_items:
        print("❌ No satellite passes found at all. Check coordinates.")
        return None

    # Sort by least cloudy
    best_item = sorted(clean_items, key=lambda x: x.properties.get("eo:cloud_cover"))[0]
    best_date = best_item.datetime.strftime("%Y-%m-%d")
    
    print(f"✅ Best Match Found: {best_date} (Cloud Cover: {best_item.properties['eo:cloud_cover']}%)")
    return best_date


def get_best_date_for_year(row, year, profile, scene_cache, scene_cache_path):
    farm_id = normalize_farm_id(row["farm_id"])
    country_code = get_country_code(row)
    cache_key = f"{farm_id}:{year}"
    if cache_key in scene_cache:
        return scene_cache[cache_key]

    bbox = get_farm_bbox(row)
    cfg = get_profile_for_country(country_code, profile)
    max_cloud = cfg.get("max_cloud", 15)

    for start_date, end_date in get_year_windows(year, country_code, profile):
        selected = find_cleanest_date(bbox=bbox, start_date=start_date, end_date=end_date, max_cloud=max_cloud)
        if selected:
            with STATE_LOCK:
                scene_cache[cache_key] = selected
                save_json_file(scene_cache_path, scene_cache)
            return selected

    return None

def download_farm_image(row, date, farm_id):
    # 1. Determine Output Filename First (for Checkpointing)
    year = date.split("-")[0]
    if year == "2020":
        output_dir = "data/raw_satellite/2020_baseline"
    elif year == "2024":
        output_dir = "data/raw_satellite/2024_current"
    else:
        output_dir = f"data/raw_satellite/{year}_other"
        
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{farm_id}_{date}.tiff"
    
    # 2. Check if already exists
    if os.path.exists(filename):
        print(f"   ⏭️ Image exists, skipping download: {filename}")
        return filename

    # 3. Proceed with Download
    print(f"⬇️ Downloading tile for {farm_id} on {date}...")
    token = get_auth_token()
    bbox = get_farm_bbox(row)

    # EVALSCRIPT: Ask for 5 Bands (Red, Green, Blue, NIR, SCL)
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["B04", "B03", "B02", "B08", "SCL"], 
        output: { bands: 5, sampleType: "FLOAT32" }
      };
    }
    function evaluatePixel(sample) {
      return [sample.B04, sample.B03, sample.B02, sample.B08, sample.SCL];
    }
    """

    # --- PAYLOAD STRUCTURE ---
    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{date}T00:00:00Z",
                        "to": f"{date}T23:59:59Z"
                    }
                }
            }]
        },
        "output": {
            "width": 512,
            "height": 512,
            "responses": [{
                "identifier": "default", 
                "format": {"type": "image/tiff"}
            }]
        },
        "evalscript": evalscript
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = request_with_retry("POST", PROCESS_URL, json=payload, headers=headers, timeout=60)

    # Re-auth once if needed
    if response.status_code == 401:
        token = get_auth_token()
        headers["Authorization"] = f"Bearer {token}"
        response = request_with_retry("POST", PROCESS_URL, json=payload, headers=headers, timeout=60)

    # ERROR HANDLING
    if response.status_code != 200:
        print(f"❌ API Error: {response.status_code}")
        print(f"📜 Server Message: {response.text}")
        return None
    
    print(f"   💾 Saving to: {filename}")
    with open(filename, "wb") as f:
        f.write(response.content)
        
    print(f"✅ Success! Saved raw TIFF to: {filename}")
    return filename

# --- MAIN WORKFLOW ---

def process_single_farm(row, skip_count, index, profile, scene_cache, scene_cache_path, manifest_path):
    """
    Worker function to process a single farm.
    """
    try:
        crop = row.get('crop_type', 'Unknown')
        display_index = index + skip_count + 1
        
        farm_id = normalize_farm_id(row['farm_id'])
        
        print(f"\n--- Farm {display_index} [{crop}]: ID {farm_id} ---")

        # Check if already processed
        path_2020 = f"data/raw_satellite/2020_baseline/{farm_id}_*.tiff"
        path_2024 = f"data/raw_satellite/2024_current/{farm_id}_*.tiff"
        
        has_2020 = len(glob.glob(path_2020)) > 0
        has_2024 = len(glob.glob(path_2024)) > 0
        
        if has_2020 and has_2024:
            print(f"   ⏭️ Fully processed (2020 & 2024 found). Skipping.")
            update_manifest(manifest_path, farm_id, 2020, "downloaded", "existing file")
            update_manifest(manifest_path, farm_id, 2024, "downloaded", "existing file")
            return True

        for year in [2020, 2024]:
            expected_glob = f"data/raw_satellite/{year}_baseline/{farm_id}_*.tiff" if year == 2020 else f"data/raw_satellite/{year}_current/{farm_id}_*.tiff"
            if len(glob.glob(expected_glob)) > 0:
                update_manifest(manifest_path, farm_id, year, "downloaded", "existing file")
                continue

            update_manifest(manifest_path, farm_id, year, "pending", "selecting scene")
            date_selected = get_best_date_for_year(row, year, profile, scene_cache, scene_cache_path)
            if not date_selected:
                update_manifest(manifest_path, farm_id, year, "failed", "no suitable scene date")
                print(f"   ⚠️ No clear {year} image found.")
                return False

            update_manifest(manifest_path, farm_id, year, "selected_scene", date_selected)
            out_file = download_farm_image(row, date_selected, f"{farm_id}_{year}")
            if out_file and os.path.exists(out_file):
                update_manifest(manifest_path, farm_id, year, "downloaded", out_file)
            else:
                update_manifest(manifest_path, farm_id, year, "failed", f"download failed for {date_selected}")
                return False

        return True

    except Exception as e:
        print(f"Skipping row {index}: {e}")
        try:
            farm_id = normalize_farm_id(row['farm_id'])
            update_manifest(manifest_path, farm_id, 2020, "failed", str(e))
            update_manifest(manifest_path, farm_id, 2024, "failed", str(e))
        except Exception:
            pass
        return False

def download_all_farms(
    csv_path: str,
    skip_count: int = 0,
    use_dask: bool = True,
    max_workers: int = 4,
    countries=None,
    profile_path: str = "inputs/acquisition_profiles.json",
    manifest_path: str = "reports/download_manifest.csv",
    scene_cache_path: str = "cache/scene_candidates.json",
    limit_per_crop: int = 100,
):
    """
    Batch download satellite imagery for all farms in the CSV.
    Uses Dask for parallelism if use_dask is True.
    """
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return

    print(f"📖 Reading farms from: {csv_path}")
    df = pd.read_csv(csv_path)

    if countries and 'country_iso2' in df.columns:
        normalized = {c.strip().upper()[:2] for c in countries if c.strip()}
        df = df[df['country_iso2'].astype(str).str.upper().str[:2].isin(normalized)].reset_index(drop=True)
        print(f"🌍 Country filter active: {sorted(normalized)} -> {len(df)} farms")

    if 'crop_type' in df.columns:
        df = df.groupby('crop_type').head(limit_per_crop).reset_index(drop=True)
    
    if skip_count > 0:
        df = df.iloc[skip_count:].reset_index(drop=True)

    profile = load_json_file(profile_path, DEFAULT_PROFILE)
    scene_cache = load_json_file(scene_cache_path, {})

    print(f"🚜 Found {len(df)} total farms to process.")

    if use_dask:
        print("⚡ Using Dask for parallel downloads...")
        tasks = []
        for index, row in df.iterrows():
            tasks.append(
                delayed(process_single_farm)(
                    row,
                    skip_count,
                    index,
                    profile,
                    scene_cache,
                    scene_cache_path,
                    manifest_path,
                )
            )
        
        # Run tasks with a limited number of workers to respect rate limits
        results = dask.compute(*tasks, scheduler='threads', num_workers=max_workers)
        success_count = sum(results)
        fail_count = len(results) - success_count
    else:
        print("🚶 Processing sequentially...")
        success_count = 0
        fail_count = 0
        for index, row in df.iterrows():
            if process_single_farm(row, skip_count, index, profile, scene_cache, scene_cache_path, manifest_path):
                success_count += 1
            else:
                fail_count += 1

    print(f"\n🎉 Batch processing complete!")
    print(f"✅ Successful PAIRS: {int(success_count)}")
    print(f"❌ Failed/Parity Removed: {int(fail_count)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Sentinel-2 farm imagery with retries and resume support.")
    parser.add_argument("--csv-path", default="inputs/farms_osm.csv", help="Input farm CSV path")
    parser.add_argument("--skip-count", type=int, default=0, help="Rows to skip from start")
    parser.add_argument("--no-dask", action="store_true", help="Disable threaded dask execution")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of dask thread workers")
    parser.add_argument("--countries", default="", help="Optional comma-separated ISO2 country filter, e.g. DE,FR,ES")
    parser.add_argument("--profile-path", default="inputs/acquisition_profiles.json", help="Country acquisition profile JSON")
    parser.add_argument("--manifest-path", default="reports/download_manifest.csv", help="Resume/manifest CSV")
    parser.add_argument("--scene-cache-path", default="cache/scene_candidates.json", help="Scene date cache JSON")
    parser.add_argument("--limit-per-crop", type=int, default=100, help="Maximum farms per crop type")

    args = parser.parse_args()
    country_list = [c.strip() for c in args.countries.split(",") if c.strip()]

    download_all_farms(
        csv_path=args.csv_path,
        skip_count=args.skip_count,
        use_dask=not args.no_dask,
        max_workers=args.max_workers,
        countries=country_list,
        profile_path=args.profile_path,
        manifest_path=args.manifest_path,
        scene_cache_path=args.scene_cache_path,
        limit_per_crop=args.limit_per_crop,
    )