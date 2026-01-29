import os
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

def get_auth_token():
    payload = {
        "client_id": "cdse-public",
        "username": USERNAME,
        "password": PASSWORD,
        "grant_type": "password",
    }
    response = requests.post(AUTH_URL, data=payload)
    response.raise_for_status()
    return response.json()["access_token"]

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
    print(f"⬇️ Downloading tile for {farm_id} on {date}...")
    token = get_auth_token()
    
    # 500m x 500m Box
    delta = 0.005 
    bbox = [lon - delta, lat - delta, lon + delta, lat + delta]

    # EVALSCRIPT: Ask for 4 Raw Bands (Red, Green, Blue, NIR)
    # We use sampleType: "FLOAT32" to get the raw scientific numbers
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["B04", "B03", "B02", "B08"], 
        output: { bands: 4, sampleType: "FLOAT32" }
      };
    }
    function evaluatePixel(sample) {
      return [sample.B04, sample.B03, sample.B02, sample.B08];
    }
    """

    # --- PAYLOAD STRUCTURE (MUST MATCH EXACTLY) ---
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
            "width": 512,  # Keep this power of 2 (e.g., 512, 1024)
            "height": 512,
            "responses": [{
                "identifier": "default", 
                "format": {"type": "image/tiff"} # <--- Requesting TIFF
            }]
        },
        "evalscript": evalscript
    }
    # ----------------------------------------------

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # DEBUG: Print payload if it fails again
    # import json
    # print(json.dumps(payload, indent=2))

    response = requests.post(PROCESS_URL, json=payload, headers=headers)

    # ERROR HANDLING
    if response.status_code != 200:
        print(f"❌ API Error: {response.status_code}")
        print(f"📜 Server Message: {response.text}")
        return None

    filename = f"data/raw_satellite/2024_current/{farm_id}_{date}.tiff"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, "wb") as f:
        f.write(response.content)
        
    print(f"✅ Success! Saved raw TIFF to: {filename}")
    return filename

# --- MAIN WORKFLOW ---
if __name__ == "__main__":
    # 1. Define your farm
    FARM_LAT = 44.42
    FARM_LON = 8.95
    FARM_NAME = "genoa_test"
    
    # 2. Find the best date in 2023
    best_date = find_cleanest_date(FARM_LAT, FARM_LON, "2023-06-01", "2023-08-30")
    
    # 3. Download that specific date
    if best_date:
        download_farm_image(FARM_LAT, FARM_LON, best_date, FARM_NAME)