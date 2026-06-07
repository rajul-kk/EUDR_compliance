"""
Enrich farms_osm.csv with ISO2 country codes using boundary heuristics.
Simple lookup based on lat/lon ranges for major agricultural regions.
"""
import logging
import os
import sys

import pandas as pd

logger = logging.getLogger(__name__)

# Country boundary lookup: (lat_min, lat_max, lon_min, lon_max, iso2)
# Covers major agricultural regions in the dataset
COUNTRY_BOUNDS = [
    # Africa
    (2, 13, 8, 16, 'CM'),      # Cameroon (Cocoa)
    (4, 12, -4, 2, 'GH'),      # Ghana (Cocoa)
    (4, 11, -9, -2, 'CI'),     # Ivory Coast (Cocoa)
    (-3, 0, 25, 35, 'TZ'),     # Tanzania
    (-15, -5, 20, 35, 'ZM'),   # Zambia (Coffee)
    (4, 14, 2, 15, 'NG'),      # Nigeria
    (3, 15, 33, 48, 'ET'),     # Ethiopia (Coffee)
    (-5, 5, 33, 42, 'KE'),     # Kenya (Coffee)
    (-2, 4, 29, 35, 'UG'),     # Uganda (Coffee)
    (4, 8, -12, -8, 'LR'),     # Liberia (Rubber)
    (-29, -17, 12, 26, 'NA'),  # Namibia (Cattle)
    # South Asia
    (8, 37, 68, 97, 'IN'),     # India (Rubber, Coffee)
    # Southeast Asia — specific countries before generic Indonesia
    (5, 22, 97, 106, 'TH'),    # Thailand (Rubber)
    (8, 24, 102, 110, 'VN'),   # Vietnam (Rubber, Coffee)
    (10, 29, 92, 102, 'MM'),   # Myanmar (Rubber)
    (0, 8, 109, 120, 'MY'),    # Malaysia (Oil Palm, Sabah)
    (1, 8, 99, 109, 'MY'),     # Malaysia (Oil Palm, Peninsular)
    (-5, 6, 95, 141, 'ID'),    # Indonesia (Oil Palm) — after MY
    # East Asia
    (18, 54, 73, 135, 'CN'),   # China
    # South America
    (-5, 13, -80, -67, 'CO'),  # Colombia (Coffee) — expanded north
    (7, 10, -83, -77, 'PA'),   # Panama (Coffee)
    (8, 12, -86, -83, 'CR'),   # Costa Rica (Coffee)
    (-33, -5, -75, -50, 'BR'), # Brazil (Soy, Coffee, Cocoa)
    (-10, -5, -80, -60, 'PE'), # Peru (Coffee, Cocoa)
    (-56, -20, -74, -53, 'AR'),# Argentina (Cattle)
    # North America
    (25, 50, -125, -66, 'US'), # USA (Soy, Cattle)
    (14, 33, -118, -86, 'MX'), # Mexico (Cattle)
    # Oceania
    (-47, -34, 166, 178, 'NZ'),# New Zealand (Cattle)
    (-44, -10, 112, 154, 'AU'),# Australia (Cattle)
    # Europe
    (36, 48, 6, 19, 'IT'),     # Italy (Rice) — expanded
    (43, 49, 22, 30, 'RO'),    # Romania (Soy)
    (45, 52, 5, 8, 'FR'),      # France
    (47, 55, 5, 15, 'DE'),     # Germany
    (36, 44, -9, -6, 'PT'),    # Portugal
    (36, 44, -10, 4, 'ES'),    # Spain
]

def infer_country_iso2(lat, lon):
    """Simple lat/lon boundary lookup for country ISO2."""
    lat = float(lat)
    lon = float(lon)

    for lat_min, lat_max, lon_min, lon_max, iso2 in COUNTRY_BOUNDS:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return iso2

    return ''  # Unknown — do not guess


def enrich_with_heuristic(df):
    """Use simple lat/lon-based country lookup."""
    try:
        logger.info("Using lat/lon boundary lookup for country inference")
        df = df.copy()

        if 'country_iso2' not in df.columns:
            df['country_iso2'] = ''

        for idx, row in df.iterrows():
            raw_current = row.get('country_iso2', '')
            if pd.notna(raw_current) and str(raw_current).strip() and str(raw_current).strip().lower() != 'nan':
                continue
            df.at[idx, 'country_iso2'] = infer_country_iso2(row['lat'], row['lon'])

        return df

    except Exception as e:
        logger.warning("Heuristic enrichment failed: %s", e)
        return None


def main():
    csv_path = 'inputs/farms_osm.csv'
    output_path = 'inputs/farms_osm.csv'

    if not os.path.exists(csv_path):
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    logger.info("Loading %s", csv_path)
    df = pd.read_csv(csv_path)

    before_filled = 0
    if 'country_iso2' in df.columns:
        before_filled = (df['country_iso2'].fillna('').astype(str).str.strip() != '').sum()
        logger.info("country_iso2 already exists — filling missing values (%d/%d filled)", before_filled, len(df))
    else:
        logger.info("Enriching %d farms with country ISO2 codes", len(df))

    result = enrich_with_heuristic(df)

    if result is None:
        logger.error("Enrichment failed — CSV not updated")
        sys.exit(1)

    filled = (result['country_iso2'].fillna('').astype(str).str.strip() != '').sum()
    logger.info("Enrichment complete: %d/%d farms with country_iso2", filled, len(result))
    if before_filled:
        logger.info("Newly filled this run: %d", filled - before_filled)

    if filled > 0:
        dist = result[result['country_iso2'].fillna('').astype(str).str.strip() != '']['country_iso2'].value_counts().head(15)
        logger.info("Country distribution:\n%s", dist.to_string())

    result.to_csv(output_path, index=False)
    logger.info("Saved enriched CSV to %s", output_path)


if __name__ == "__main__":
    main()
