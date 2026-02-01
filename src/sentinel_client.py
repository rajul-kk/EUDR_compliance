import time
import os
import glob
import pandas as pd
import requests
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
            response = requests.post(AUTH_URL, data=payload, timeout=60)
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

def find_cleanest_date(lat, lon, start_date, end_date):
    """
    Step 1: Search the Catalog to find the EXACT date of a clear flyover.
    """
    print(f"🔎 Searching for clear images between {start_date} and {end_date}...")
    
    client = Client.open(CATALOG_URL)
    bbox = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
    
    # Search for Sentinel-2 Level-2A (Cloud corrected)
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}"
    )
    
    items = list(search.items())
    
    # Filter for low clouds (< 10%)
    clean_items = [i for i in items if i.properties.get("eo:cloud_cover", 100) < 10]
    
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

def download_farm_image(lat, lon, date, farm_id):
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
    
    # 500m x 500m Box
    delta = 0.005 
    bbox = [lon - delta, lat - delta, lon + delta, lat + delta]

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

    response = requests.post(PROCESS_URL, json=payload, headers=headers, timeout=60)

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

def download_all_farms(csv_path: str, skip_count: int = 0):
    """
    Batch download satellite imagery for all farms in the CSV.
    Limits to 100 per crop type and enforces rate limiting.
    Allows skipping known completed rows.
    """
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return

    print(f"📖 Reading farms from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Filter: Top 100 per crop type
    if 'crop_type' in df.columns:
        print("   🔻 Filtering: Selecting top 100 farms per crop type...")
        df = df.groupby('crop_type').head(100).reset_index(drop=True)
    
    # Explicit Skip
    if skip_count > 0:
        print(f"   ⏭️ Skipping first {skip_count} farms as requested...")
        df = df.iloc[skip_count:].reset_index(drop=True)

    print(f"🚜 Found {len(df)} total farms to process. Starting batch download (<60 req/min)...")
    
    # Statistics
    success_count = 0
    fail_count = 0

    for index, row in df.iterrows():
        try:
            crop = row.get('crop_type', 'Unknown')
            # Adjust index for display to match original count if possible, or just local
            display_index = index + skip_count + 1
            
            farm_id = str(row['farm_id']).replace("osm_", "").replace("('", "").replace("', ", "_").replace(")", "")
            lat = float(row['lat'])
            lon = float(row['lon'])
            
            print(f"\n--- Farm {display_index} [{crop}]: ID {farm_id} ---")

            # Check if already processed (both 2020 and 2024 exist)
            # Pattern: {dir}/{farm_id}_{date}.tiff
            # We don't know the date, so we glob.
            path_2020 = f"data/raw_satellite/2020_baseline/{farm_id}_*.tiff"
            path_2024 = f"data/raw_satellite/2024_current/{farm_id}_*.tiff"
            
            has_2020 = len(glob.glob(path_2020)) > 0
            has_2024 = len(glob.glob(path_2024)) > 0
            
            if has_2020 and has_2024:
                print(f"   ⏭️ Fully processed (2020 & 2024 found). Skipping.")
                continue

            file_2020 = None
            
            # 1. Process 2020 (Baseline)
            try:
                # Search for best date in June 2020
                print("   🔎 Searching 2020...")
                date_2020 = find_cleanest_date(lat, lon, "2020-06-01", "2020-06-30")
                time.sleep(1) # Rate limit
                
                if date_2020:
                    print(f"   ⬇️ Downloading 2020 ({date_2020})...")
                    file_2020 = download_farm_image(lat, lon, date_2020, f"{farm_id}_2020")
                    time.sleep(2) # Rate limit
                else:
                    print("   ⚠️ No clear 2020 image found. Skipping pair.")
                    continue

            except Exception as e:
                print(f"   ❌ Failed 2020 Search/Download: {e}")
                continue

            # 2. Process 2024 (Current) - ONLY if 2020 succeeded
            if file_2020 and os.path.exists(file_2020):
                try:
                    print("   🔎 Searching 2024...")
                    date_2024 = find_cleanest_date(lat, lon, "2024-06-01", "2024-06-30")
                    time.sleep(1) 
                    
                    if date_2024:
                        print(f"   ⬇️ Downloading 2024 ({date_2024})...")
                        file_2024 = download_farm_image(lat, lon, date_2024, f"{farm_id}_2024")
                        time.sleep(2)
                        
                        if file_2024:
                            success_count += 1
                        else:
                            raise Exception("Download 2024 returned None")
                    else:
                        raise Exception("No clear 2024 image found")

                except Exception as e:
                    print(f"   ❌ Failed 2024: {e}")
                    print(f"   🧹 Parity Check: Removing orphan 2020 file: {file_2020}")
                    try:
                        os.remove(file_2020)
                        print("      🗑️ Deleted.")
                    except:
                        pass
                    fail_count += 1
            else:
                print("   ⚠️ 2020 download failed? Skipping 2024.")

        except Exception as e:
            print(f"Skipping row {index}: {e}")

    print(f"\n🎉 Batch processing complete!")
    print(f"✅ Successful PAIRS: {int(success_count)}")
    print(f"❌ Failed/Parity Removed: {int(fail_count)}")


if __name__ == "__main__":
    # Path to your farms CSV
    csv_path = "inputs/farms_osm.csv"
    
    # Run the batch process
    # Skipping 229 as requested
    download_all_farms(csv_path, skip_count=229)
    
    # Example Single Download (Commented out)
    # lat = 44.42  # Example: Genoa, Italy
    # lon = 8.95
    # download_farm_image(lat, lon, "2024-06-01", "genoa_test")