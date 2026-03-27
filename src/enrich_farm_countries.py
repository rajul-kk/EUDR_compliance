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
    # Southeast Asia
    (-5, 5, 95, 115, 'ID'),    # Indonesia (Oil Palm)
    (0, 8, 95, 110, 'MY'),     # Malaysia (Oil Palm)
    # South America
    (-25, -5, -75, -50, 'BR'),  # Brazil (Soy, Coffee, Cocoa)
    (-5, 0, -80, -65, 'CO'),    # Colombia (Coffee)
    (-10, -5, -70, -60, 'PE'),  # Peru (Coffee, Cocoa)
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
        iso2_codes = []
        
        for _, row in df.iterrows():
            iso2 = infer_country_iso2(row['lat'], row['lon'])
            iso2_codes.append(iso2)
        
        df['country_iso2'] = iso2_codes
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
    
    # Check if already enriched
    if 'country_iso2' in df.columns:
        print("✅ CSV already has country_iso2 column. Skipping enrichment.")
        return
    
    print(f"🔄 Enriching {len(df)} farms with country ISO2 codes...")
    
    # Use heuristic boundary lookup
    result = enrich_with_heuristic(df)
    
    if result is None:
        print("❌ Enrichment failed. CSV not updated.")
        sys.exit(1)
    
    # Report results
    filled = (result['country_iso2'] != '').sum()
    print(f"\n✅ Enrichment complete: {filled}/{len(result)} farms with country_iso2")
    
    # Show distribution
    if filled > 0:
        print("\nCountry distribution:")
        print(result[result['country_iso2'] != '']['country_iso2'].value_counts().head(15))
    
    # Save
    result.to_csv(output_path, index=False)
    print(f"\n💾 Saved enriched CSV to {output_path}")


if __name__ == "__main__":
    main()
