import io
import json
import logging
import os
import random
import glob
import argparse
import threading
import time

import numpy as np
import pandas as pd
import requests
import rasterio
import dask
from dask import delayed
from dotenv import load_dotenv
from pystac_client import Client

load_dotenv()

logger = logging.getLogger(__name__)

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

# SCL classes treated as cloud / shadow
_CLOUD_SCL = {0, 1, 3, 8, 9, 10}

DEFAULT_PROFILE = {
    "default": {
        "month_windows": [["05-01", "09-30"]],
        "max_cloud": 30,
        "max_scenes": 8,
    },
    "tropical": {
        "month_windows": [["01-01", "12-31"]],
        "max_cloud": 50,
        "max_scenes": 10,
    },
}

EU_COUNTRY_CODES = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR",
    "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
    "SI", "ES", "SE",
}

# JavaScript evalscript — requests 5 bands: R, G, B, NIR, SCL
_EVALSCRIPT = """
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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def request_with_retry(method, url, max_retries=4, retryable_statuses=None, timeout=60, **kwargs):
    """HTTP wrapper with bounded exponential-backoff retries."""
    if retryable_statuses is None:
        retryable_statuses = {429, 500, 502, 503, 504}

    for attempt in range(max_retries):
        try:
            response = requests.request(method=method, url=url, timeout=timeout, **kwargs)
            if response.status_code in retryable_statuses:
                raise requests.HTTPError(f"Transient {response.status_code}", response=response)
            return response
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = min(30, (2 ** attempt) + random.random())
            logger.warning("Request failed (%s). Retry %d/%d in %.1fs", exc, attempt + 1, max_retries, wait)
            time.sleep(wait)


def normalize_farm_id(raw_id):
    return str(raw_id).replace("osm_", "").replace("('", "").replace("', ", "_").replace(")", "")


def load_json_file(path, default):
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load JSON %s: %s. Using default.", path, exc)
        return default


def save_json_file(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_country_code(row):
    value = str(row.get("country_iso2", "")).strip().upper()
    return value[:2] if value else ""


def get_profile_for_country(country_code, profile):
    if country_code and country_code in EU_COUNTRY_CODES:
        return profile.get("default", DEFAULT_PROFILE["default"])
    return profile.get("tropical", profile.get("default", DEFAULT_PROFILE["default"]))


def get_year_windows(year, country_code, profile):
    cfg = get_profile_for_country(country_code, profile)
    windows = cfg.get("month_windows", DEFAULT_PROFILE["default"]["month_windows"])
    return [(f"{year}-{s}", f"{year}-{e}") for s, e in windows]


def get_farm_bbox(row, delta=0.005):
    lat, lon = float(row["lat"]), float(row["lon"])
    min_lon, min_lat = row.get("min_lon"), row.get("min_lat")
    max_lon, max_lat = row.get("max_lon"), row.get("max_lat")
    if pd.notna(min_lon) and pd.notna(min_lat) and pd.notna(max_lon) and pd.notna(max_lat):
        return [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]
    return [lon - delta, lat - delta, lon + delta, lat + delta]


def update_manifest(manifest_path, farm_id, year, status, message=""):
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
            df = pd.concat([df, pd.DataFrame([{
                "farm_id": farm_id, "year": int(year),
                "status": status, "message": message, "updated_at": now_ts,
            }])], ignore_index=True)
        df.to_csv(manifest_path, index=False)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def get_auth_token():
    global ACCESS_TOKEN, TOKEN_EXPIRY
    current_time = time.time()
    if ACCESS_TOKEN and current_time < (TOKEN_EXPIRY - 60):
        return ACCESS_TOKEN

    logger.debug("Refreshing Copernicus access token")
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
            TOKEN_EXPIRY = current_time + data.get("expires_in", 600)
            return ACCESS_TOKEN
        except Exception as exc:
            if attempt < 2:
                wait = (attempt + 1) * 2
                logger.warning("Auth attempt %d failed (%s). Retrying in %ds", attempt + 1, exc, wait)
                time.sleep(wait)
            else:
                logger.error("Auth failed after 3 attempts: %s", exc)
                raise


# ---------------------------------------------------------------------------
# STAC scene search
# ---------------------------------------------------------------------------

def search_scenes(bbox, start_date, end_date, max_cloud=50):
    """Return all STAC items in the window at or below max_cloud, sorted ascending by cloud cover."""
    logger.debug("Searching scenes %s to %s (max cloud %.0f%%)", start_date, end_date, max_cloud)
    client = Client.open(CATALOG_URL)
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
    )
    items = list(search.items())
    items = [i for i in items if i.properties.get("eo:cloud_cover", 100) <= max_cloud]
    items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
    logger.debug("Found %d scenes under %.0f%% cloud", len(items), max_cloud)
    return items


# ---------------------------------------------------------------------------
# Single-scene download (in-memory array)
# ---------------------------------------------------------------------------

def _download_scene_bytes(bbox, date):
    """Call the Sentinel Hub Process API for one date. Returns raw bytes or None."""
    token = get_auth_token()
    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {
                "timeRange": {"from": f"{date}T00:00:00Z", "to": f"{date}T23:59:59Z"},
            }}],
        },
        "output": {
            "width": 512, "height": 512,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": _EVALSCRIPT,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = request_with_retry("POST", PROCESS_URL, json=payload, headers=headers, timeout=60)

    if response.status_code == 401:
        token = get_auth_token()
        headers["Authorization"] = f"Bearer {token}"
        response = request_with_retry("POST", PROCESS_URL, json=payload, headers=headers, timeout=60)

    if response.status_code != 200:
        logger.warning("Process API error %d for date %s: %s", response.status_code, date, response.text[:200])
        return None

    return response.content


def _bytes_to_array(data):
    """Open GeoTIFF bytes in memory. Returns (array, profile) or (None, None)."""
    try:
        with rasterio.open(io.BytesIO(data)) as src:
            return src.read().astype(np.float32), src.profile.copy()
    except Exception as exc:
        logger.warning("Failed to parse scene bytes: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# Cloud-free median compositing
# ---------------------------------------------------------------------------

def build_median_composite(bbox, start_date, end_date, max_cloud=30, max_scenes=8):
    """
    Download up to max_scenes scenes and produce a cloud-free composite.

    Strategy:
      - Bands 0-3 (R, G, B, NIR): per-pixel nanmedian across all observations
        where that pixel was not cloud/shadow in any scene.
      - Band 4 (SCL): taken from the least-cloudy scene; pixels that were cloudy
        there but clear in at least one other scene are overwritten with SCL=4.

    Falls back to a relaxed cloud threshold (80%) when nothing is found.

    Returns:
        (composite_array, rasterio_profile)  shape (5, H, W), float32
        or (None, None) on failure.
    """
    scenes = search_scenes(bbox, start_date, end_date, max_cloud=max_cloud)

    if not scenes:
        logger.warning("No scenes under %.0f%% cloud — relaxing threshold to 80%%", max_cloud)
        scenes = search_scenes(bbox, start_date, end_date, max_cloud=80)

    if not scenes:
        logger.error("No scenes found between %s and %s", start_date, end_date)
        return None, None

    scenes = scenes[:max_scenes]
    logger.info("Compositing %d scene(s) [%s – %s]", len(scenes), start_date, end_date)

    spectral_stack = []   # list of (4, H, W) arrays with NaN for clouds
    scl_clearest = None   # SCL from least-cloudy scene (first in sorted list)
    best_profile = None

    for item in scenes:
        date_str = item.datetime.strftime("%Y-%m-%d")
        raw = _download_scene_bytes(bbox, date_str)
        if raw is None:
            continue

        arr, profile = _bytes_to_array(raw)
        if arr is None:
            continue

        if best_profile is None:
            best_profile = profile

        scl = arr[4].astype(np.int16)
        cloud_mask = np.isin(scl, list(_CLOUD_SCL))

        spectral = arr[:4].copy()
        spectral[:, cloud_mask] = np.nan
        spectral_stack.append(spectral)

        if scl_clearest is None:
            scl_clearest = scl.astype(np.float32)

    if not spectral_stack:
        logger.error("All scene downloads failed for window %s – %s", start_date, end_date)
        return None, None

    # Spectral composite
    stacked = np.stack(spectral_stack, axis=0)          # (N, 4, H, W)
    composite_spectral = np.nanmedian(stacked, axis=0)  # (4, H, W)
    composite_spectral = np.nan_to_num(composite_spectral, nan=0.0)

    # SCL: mark pixels that were cloudy in clearest scene but have data elsewhere
    had_any_clear = ~np.all(np.isnan(stacked), axis=(0, 1))  # (H, W)
    was_cloud = np.isin(scl_clearest.astype(np.int16), list(_CLOUD_SCL))
    scl_clearest[had_any_clear & was_cloud] = 4.0  # reclassify as clear vegetation

    composite = np.concatenate(
        [composite_spectral, scl_clearest[np.newaxis]], axis=0
    )  # (5, H, W)

    valid_pct = float(np.mean(composite_spectral[0] != 0) * 100)
    logger.info("Composite complete — %.1f%% valid pixels (non-zero in band 0)", valid_pct)

    return composite, best_profile


# ---------------------------------------------------------------------------
# Farm image download (uses compositing)
# ---------------------------------------------------------------------------

def download_farm_image(row, year, farm_id, start_date, end_date, profile_cfg):
    """
    Build a cloud-free composite for one farm/year and save as GeoTIFF.

    Args:
        row:         Farm metadata row (needs lat, lon, and optional bbox columns).
        year:        Integer year (2020 or 2024).
        farm_id:     Normalised farm identifier string.
        start_date:  Composite window start (YYYY-MM-DD).
        end_date:    Composite window end (YYYY-MM-DD).
        profile_cfg: Profile dict with max_cloud, max_scenes keys.

    Returns:
        Output file path on success, None on failure.
    """
    year_str = str(year)
    if year_str == "2020":
        output_dir = "data/raw_satellite/2020_baseline"
    elif year_str == "2024":
        output_dir = "data/raw_satellite/2024_current"
    else:
        output_dir = f"data/raw_satellite/{year_str}_other"

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{farm_id}_{year}.tiff")

    if os.path.exists(filename):
        logger.info("Skipping %s — composite already exists", filename)
        return filename

    bbox = get_farm_bbox(row)
    max_cloud = profile_cfg.get("max_cloud", 30)
    max_scenes = profile_cfg.get("max_scenes", 8)

    composite, profile = build_median_composite(
        bbox, start_date, end_date,
        max_cloud=max_cloud, max_scenes=max_scenes,
    )

    if composite is None:
        logger.error("Compositing failed for farm %s year %d", farm_id, year)
        return None

    profile.update(count=5, dtype="float32", driver="GTiff", compress="lzw")
    with rasterio.open(filename, "w", **profile) as dst:
        dst.write(composite)

    logger.info("Saved composite (%d scenes) to %s", max_scenes, filename)
    return filename


# ---------------------------------------------------------------------------
# Per-farm orchestration
# ---------------------------------------------------------------------------

def process_single_farm(row, skip_count, index, profile, manifest_path):
    try:
        crop = row.get("crop_type", "Unknown")
        farm_id = normalize_farm_id(row["farm_id"])
        display_index = index + skip_count + 1
        country_code = get_country_code(row)
        profile_cfg = get_profile_for_country(country_code, profile)

        logger.info("Farm %d [%s]: %s", display_index, crop, farm_id)

        path_2020 = f"data/raw_satellite/2020_baseline/{farm_id}_*.tiff"
        path_2024 = f"data/raw_satellite/2024_current/{farm_id}_*.tiff"

        if glob.glob(path_2020) and glob.glob(path_2024):
            logger.info("Farm %s already fully downloaded — skipping", farm_id)
            update_manifest(manifest_path, farm_id, 2020, "downloaded", "existing file")
            update_manifest(manifest_path, farm_id, 2024, "downloaded", "existing file")
            return True

        for year in [2020, 2024]:
            suffix = "baseline" if year == 2020 else "current"
            if glob.glob(f"data/raw_satellite/{year}_{suffix}/{farm_id}_*.tiff"):
                update_manifest(manifest_path, farm_id, year, "downloaded", "existing file")
                continue

            update_manifest(manifest_path, farm_id, year, "pending", "selecting window")

            windows = get_year_windows(year, country_code, profile)
            success = False
            for start_date, end_date in windows:
                out = download_farm_image(row, year, farm_id, start_date, end_date, profile_cfg)
                if out and os.path.exists(out):
                    update_manifest(manifest_path, farm_id, year, "downloaded", out)
                    success = True
                    break

            if not success:
                update_manifest(manifest_path, farm_id, year, "failed", "no composite produced")
                logger.warning("No composite produced for farm %s year %d", farm_id, year)
                return False

        return True

    except Exception as exc:
        logger.exception("Unexpected error processing farm at index %d: %s", index, exc)
        try:
            fid = normalize_farm_id(row["farm_id"])
            update_manifest(manifest_path, fid, 2020, "failed", str(exc))
            update_manifest(manifest_path, fid, 2024, "failed", str(exc))
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def download_all_farms(
    csv_path: str,
    skip_count: int = 0,
    use_dask: bool = True,
    max_workers: int = 4,
    countries=None,
    profile_path: str = "inputs/acquisition_profiles.json",
    manifest_path: str = "reports/download_manifest.csv",
    limit_per_crop: int = 100,
):
    """Download composite imagery for all farms in the CSV."""
    if not os.path.exists(csv_path):
        logger.error("CSV not found: %s", csv_path)
        return

    logger.info("Reading farms from %s", csv_path)
    df = pd.read_csv(csv_path)

    if countries and "country_iso2" in df.columns:
        normalized = {c.strip().upper()[:2] for c in countries if c.strip()}
        df = df[df["country_iso2"].astype(str).str.upper().str[:2].isin(normalized)].reset_index(drop=True)
        logger.info("Country filter %s -> %d farms", sorted(normalized), len(df))

    if "crop_type" in df.columns:
        df = df.groupby("crop_type").head(limit_per_crop).reset_index(drop=True)

    if skip_count > 0:
        df = df.iloc[skip_count:].reset_index(drop=True)

    profile = load_json_file(profile_path, DEFAULT_PROFILE)

    logger.info("Processing %d farms", len(df))

    if use_dask:
        logger.info("Using Dask (%d workers)", max_workers)
        tasks = [
            delayed(process_single_farm)(row, skip_count, idx, profile, manifest_path)
            for idx, row in df.iterrows()
        ]
        results = dask.compute(*tasks, scheduler="threads", num_workers=max_workers)
    else:
        results = [
            process_single_farm(row, skip_count, idx, profile, manifest_path)
            for idx, row in df.iterrows()
        ]

    success = sum(results)
    logger.info("Batch complete — %d succeeded, %d failed", success, len(results) - success)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Download Sentinel-2 composite imagery for farms.")
    parser.add_argument("--csv-path", default="inputs/farms_osm.csv")
    parser.add_argument("--skip-count", type=int, default=0)
    parser.add_argument("--no-dask", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--countries", default="")
    parser.add_argument("--profile-path", default="inputs/acquisition_profiles.json")
    parser.add_argument("--manifest-path", default="reports/download_manifest.csv")
    parser.add_argument("--limit-per-crop", type=int, default=100)
    args = parser.parse_args()

    download_all_farms(
        csv_path=args.csv_path,
        skip_count=args.skip_count,
        use_dask=not args.no_dask,
        max_workers=args.max_workers,
        countries=[c.strip() for c in args.countries.split(",") if c.strip()],
        profile_path=args.profile_path,
        manifest_path=args.manifest_path,
        limit_per_crop=args.limit_per_crop,
    )
