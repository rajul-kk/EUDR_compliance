"""
Enrich farms_osm.csv with ISO2 country codes using boundary heuristics.
Simple lookup based on lat/lon ranges for major agricultural regions.
"""
import pandas as pd
import os
import sys

# Country boundary lookup: (lat_min, lat_max, lon_min, lon_max, iso2)
# Covers major agricultural regions in the dataset
COUNTRY_BOUNDS = [
    # Africa
    (2, 5, 12, 16, 'CM'),      # Cameroon (Cocoa)
    (2, 7, 8, 15, 'GH'),       # Ghana (Cocoa)
    (3, 6, 13, 18, 'CI'),      # Ivory Coast (Cocoa)
    (-3, 0, 25, 35, 'TZ'),     # Tanzania (Cocoa, Coffee)
    (-15, -5, 20, 35, 'ZM'),   # Zambia (Coffee)
    (8, 12, 2, 15, 'NG'),      # Nigeria
    (0, 12, 30, 43, 'ET'),     # Ethiopia (Coffee)
    (-5, 5, 33, 42, 'KE'),     # Kenya (Coffee)
    (-2, 4, 29, 35, 'UG'),     # Uganda (Coffee)
    # Southeast Asia
    (-5, 5, 95, 115, 'ID'),    # Indonesia (Oil Palm)
    (0, 8, 95, 110, 'MY'),     # Malaysia (Oil Palm)
    # South America
    (-25, -5, -75, -50, 'BR'),  # Brazil (Soy, Coffee, Cocoa)
    (-5, 0, -80, -65, 'CO'),    # Colombia (Coffee)
    (-10, -5, -70, -60, 'PE'),  # Peru (Coffee, Cocoa)
    (-42, -20, -74, -53, 'AR'), # Argentina (Cattle)
    # North America
    (35, 50, -105, -66, 'US'),  # USA (Soy, Cattle)
    (14, 33, -118, -86, 'MX'),  # Mexico (Cattle)
    # Oceania
    (-44, -10, 112, 154, 'AU'), # Australia (Cattle)
    # Europe
    (42, 46, 7, 12, 'IT'),      # Italy (Rice)
    (45, 50, 5, 8, 'FR'),       # France
    (48, 53, 5, 15, 'DE'),      # Germany
]

def infer_country_iso2(lat, lon):
    """Simple lat/lon boundary lookup for country ISO2."""
    lat = float(lat)
    lon = float(lon)
    
    for lat_min, lat_max, lon_min, lon_max, iso2 in COUNTRY_BOUNDS:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return iso2
    
    # Default fallback based on major regions
    if -20 <= lat <= 5 and lon > 20:
        return 'ZA'  # Africa (generic)
    if lat > 35 and lon > -10 and lon < 50:
        return 'IT'  # Europe (generic)
    if lat > 0 and 90 < lon < 140:
        return 'ID'  # Asia (generic)
    if -30 < lat < 5 and -80 < lon < -35:
        return 'BR'  # South America (generic)
    
    return ''  # Unknown


def enrich_with_heuristic(df):
    """Use simple lat/lon-based country lookup."""
    try:
        print("🌍 Using lat/lon boundary lookup for country inference...")
        df = df.copy()

        if 'country_iso2' not in df.columns:
            df['country_iso2'] = ''

        # Fill only missing ISO2 values to preserve existing assignments.
        for idx, row in df.iterrows():
            raw_current = row.get('country_iso2', '')
            if pd.notna(raw_current) and str(raw_current).strip() and str(raw_current).strip().lower() != 'nan':
                continue
            df.at[idx, 'country_iso2'] = infer_country_iso2(row['lat'], row['lon'])

        return df
        
    except Exception as e:
        print(f"⚠️ Heuristic enrichment failed: {e}")
        return None


def main():
    csv_path = 'inputs/farms_osm.csv'
    output_path = 'inputs/farms_osm.csv'  # Overwrites in place
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        sys.exit(1)
    
    print(f"📖 Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    before_filled = 0
    if 'country_iso2' in df.columns:
        before_filled = (df['country_iso2'].fillna('').astype(str).str.strip() != '').sum()
        print(f"🔄 country_iso2 already exists. Filling missing values (currently {before_filled}/{len(df)} filled)...")
    else:
        print(f"🔄 Enriching {len(df)} farms with country ISO2 codes...")
    
    # Use heuristic boundary lookup
    result = enrich_with_heuristic(df)
    
    if result is None:
        print("❌ Enrichment failed. CSV not updated.")
        sys.exit(1)
    
    # Report results
    filled = (result['country_iso2'].fillna('').astype(str).str.strip() != '').sum()
    print(f"\n✅ Enrichment complete: {filled}/{len(result)} farms with country_iso2")
    if before_filled:
        print(f"📈 Newly filled this run: {filled - before_filled}")
    
    # Show distribution
    if filled > 0:
        print("\nCountry distribution:")
        print(result[result['country_iso2'].fillna('').astype(str).str.strip() != '']['country_iso2'].value_counts().head(15))
    
    # Save
    result.to_csv(output_path, index=False)
    print(f"\n💾 Saved enriched CSV to {output_path}")


if __name__ == "__main__":
    main()
