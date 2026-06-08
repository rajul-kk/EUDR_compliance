"""
Generate per-farm forest loss labels from Hansen Global Forest Change (GFC).

Label encoding (saved as uint8):
  0 = Non-forest / Other
  1 = Forest in 2020 (no loss by end of 2020)
  2 = Post-EUDR deforestation (loss in 2021-2024)

EUDR baseline date: 31 December 2020.
Hansen asset: UMD/hansen/global_forest_change_2023_v1_11
  - treecover2000: % tree canopy cover in year 2000
  - lossyear:      year of first loss (1=2001 … 23=2023); 0 = no loss detected
"""

import logging
import os
import re
import sys
import zipfile

import ee
import pandas as pd
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from auth import initialize_gee
except ImportError:
    from GEE_dynamic.auth import initialize_gee

logger = logging.getLogger(__name__)

HANSEN_ASSET = "UMD/hansen/global_forest_change_2023_v1_11"
TREECOVER_THRESHOLD = 10    # % canopy cover in 2000 to count as forest
BUFFER_METERS = 2500        # 5km × 5km tile around centroid
OUTPUT_SCALE = 30           # Hansen native resolution (metres)
OUTPUT_CRS = "EPSG:4326"

PROJECT_ROOT = os.path.abspath(os.path.join(parent_dir, ".."))
DEFAULT_CSV = os.path.join(PROJECT_ROOT, "inputs", "farms_osm.csv")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "data", "hansen_labels")


def _download_url(url: str, output_path: str) -> bool:
    temp = output_path + ".tmp"
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(temp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        if zipfile.is_zipfile(temp):
            with zipfile.ZipFile(temp) as z:
                tiffs = [n for n in z.namelist() if n.lower().endswith((".tif", ".tiff"))]
                if not tiffs:
                    logger.error("No TIFF in zip for %s", output_path)
                    return False
                extracted = z.extract(tiffs[0], path=os.path.dirname(output_path))
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(extracted, output_path)
        else:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp, output_path)

        if os.path.exists(temp):
            os.remove(temp)
        return True

    except Exception as e:
        logger.error("Download failed for %s: %s", output_path, e)
        if os.path.exists(temp):
            os.remove(temp)
        return False


def generate_label(lat: float, lon: float, farm_id: str, output_dir: str) -> bool:
    """
    Download a Hansen GFC label mask for one farm.

    Output file: {output_dir}/{farm_id}_hansen_label.tif
    Pixel values:
      0 = non-forest or unknown
      1 = forest as of 2020 (no post-2020 loss detected)
      2 = post-EUDR loss (lossyear 21-23, i.e. 2021-2023)
    """
    out_path = os.path.join(output_dir, f"{farm_id}_hansen_label.tif")
    if os.path.exists(out_path):
        logger.debug("Label exists for %s — skipping", farm_id)
        return True

    try:
        region = ee.Geometry.Point([lon, lat]).buffer(BUFFER_METERS).bounds()
        hansen = ee.Image(HANSEN_ASSET)

        treecover = hansen.select("treecover2000")
        lossyear = hansen.select("lossyear")

        # Was this pixel forested in 2000?
        was_forest = treecover.gte(TREECOVER_THRESHOLD)

        # Post-EUDR loss: lossyear 21, 22, or 23 (2021, 2022, 2023)
        post_eudr_loss = lossyear.gte(21).And(lossyear.lte(23))

        # Forest in 2020: was forested in 2000 AND not lost by end of 2020
        # lossyear 0 = no loss; lossyear >= 21 = loss after 2020
        no_pre2021_loss = lossyear.eq(0).Or(lossyear.gte(21))
        forest_2020 = was_forest.And(no_pre2021_loss)

        # Build label: start at 0, set forest_2020=1, then overwrite with post_eudr_loss=2
        label = (
            ee.Image.constant(0).uint8()
            .where(forest_2020, 1)
            .where(was_forest.And(post_eudr_loss), 2)
            .clip(region)
            .rename("hansen_label")
        )

        url = label.getDownloadURL({
            "name": f"{farm_id}_hansen_label",
            "scale": OUTPUT_SCALE,
            "crs": OUTPUT_CRS,
            "region": region.getInfo(),
            "format": "GEO_TIFF",
        })

        success = _download_url(url, out_path)
        if success:
            logger.info("Saved: %s", out_path)
        return success

    except Exception as e:
        logger.error("GEE error for farm %s: %s", farm_id, e)
        return False


def _extract_base_id(farm_id: str) -> str:
    """Strip 'osm_' prefix if present (matches existing TIFF filename convention)."""
    return farm_id[4:] if farm_id.startswith("osm_") else farm_id


def _has_satellite_image(farm_id: str, sat_dirs: list[str]) -> bool:
    base = _extract_base_id(farm_id)
    for d in sat_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            # Match old format {base}_{year}_{date}.tiff and new format {base}_{year}.tiff
            if re.match(rf"^{re.escape(base)}_(2020|2024)", fname):
                return True
    return False


def main(csv_path: str = DEFAULT_CSV, output_dir: str = DEFAULT_OUT):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    initialize_gee()

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d farms from %s", len(df), csv_path)

    sat_dirs = [
        os.path.join(PROJECT_ROOT, "data", "raw_satellite", "2020_baseline"),
        os.path.join(PROJECT_ROOT, "data", "raw_satellite", "2024_current"),
    ]

    # Only process farms that have at least one satellite image downloaded
    farms = [
        row for _, row in df.iterrows()
        if _has_satellite_image(str(row["farm_id"]), sat_dirs)
    ]
    logger.info("%d/%d farms have satellite images — generating labels", len(farms), len(df))

    already_done = sum(
        1 for row in farms
        if os.path.exists(os.path.join(output_dir, f"{_extract_base_id(str(row['farm_id']))}_hansen_label.tif"))
    )
    logger.info("%d labels already exist — %d remaining", already_done, len(farms) - already_done)

    ok = fail = 0
    for i, row in enumerate(farms, 1):
        farm_id = str(row["farm_id"])
        base_id = _extract_base_id(farm_id)
        crop = row.get("crop_type", "?")
        logger.info("[%d/%d] %s (%s)", i, len(farms), farm_id, crop)

        success = generate_label(float(row["lat"]), float(row["lon"]), base_id, output_dir)
        if success:
            ok += 1
        else:
            fail += 1

    logger.info("Done — %d succeeded, %d failed", ok, fail)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Hansen GFC labels for EUDR farms")
    parser.add_argument("--csv-path", default=DEFAULT_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    main(csv_path=args.csv_path, output_dir=args.output_dir)
