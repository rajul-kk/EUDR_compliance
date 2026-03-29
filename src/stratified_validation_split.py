"""
Stratified validation set curation script.
Samples farms by (crop_type, country_iso2) to ensure geo and crop diversity.
"""

import argparse
import os
import pandas as pd
import numpy as np
import re


def load_farms_and_imagery(farms_csv: str, tiff_dir: str) -> pd.DataFrame:
    """
    Load farms and filter to only those with imagery on disk.
    
    Args:
        farms_csv: Path to farms_osm.csv
        tiff_dir: Base directory containing 2020_baseline and 2024_current subdirs
    
    Returns:
        DataFrame with farms that have TIFF files available
    """
    farms_df = pd.read_csv(farms_csv)
    
    # Collect all farm IDs present in imagery directories
    available_ids = set()
    for year_dir in ['2020_baseline', '2024_current']:
        year_path = os.path.join(tiff_dir, year_dir)
        if os.path.isdir(year_path):
            for fname in os.listdir(year_path):
                if fname.endswith('.tiff') or fname.endswith('.tif'):
                    # Extract farm id from patterns like:
                    # relation_10018408_2020_2020-06-07.tiff -> relation_10018408
                    # way_12345_2024_2024-06-14.tif -> way_12345
                    match = re.match(r"^(relation|way)_\d+", fname)
                    if not match:
                        continue
                    base = match.group(0)
                    available_ids.add(f"osm_{base}")
    
    # Filter to farms with imagery
    farms_df['has_imagery'] = farms_df['farm_id'].isin(available_ids)
    result = farms_df[farms_df['has_imagery']].copy()
    
    print(f"Found {len(result)} farms with available imagery (out of {len(farms_df)} total)")
    return result


def stratified_sample(
    farms_df: pd.DataFrame,
    validation_ratio: float = 0.15,
    seed: int = 42
) -> pd.DataFrame:
    """
    Stratified sampling by (crop_type, country_iso2).
    
    Args:
        farms_df: DataFrame with farm data
        validation_ratio: Fraction to sample (e.g., 0.15 = ~15%)
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with sampled validation farms
    """
    np.random.seed(seed)
    
    # Create strata as (crop_type, country_iso2) tuples
    farms_df['stratum'] = farms_df['crop_type'] + '|' + farms_df['country_iso2']
    
    # Sample within each stratum
    val_samples = []
    for stratum, group in farms_df.groupby('stratum'):
        group_size = len(group)
        sample_size = max(1, int(group_size * validation_ratio))
        sample = group.sample(n=sample_size, random_state=seed)
        val_samples.append(sample)
        print(f"  {stratum:30s} -> {len(sample):3d} / {group_size:3d}")
    
    result = pd.concat(val_samples, ignore_index=True)
    print(f"\nTotal validation farms: {len(result)}")
    return result


def save_validation_set(val_df: pd.DataFrame, output_path: str) -> None:
    """
    Save validation farm IDs to txt file (one per line).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for farm_id in sorted(val_df['farm_id']):
            f.write(f"{farm_id}\n")
    print(f"\n✅ Saved {len(val_df)} validation farms to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified validation split.")
    parser.add_argument('--farms-csv', required=True, help='Path to farms_osm.csv')
    parser.add_argument('--tiff-dir', required=True, help='Base directory with 2020_baseline/ and 2024_current/')
    parser.add_argument('--output-path', required=True, help='Output path for validation_farms.txt')
    parser.add_argument('--validation-ratio', type=float, default=0.15, help='Fraction for validation set')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("🔍 Loading farms and checking imagery availability...")
    farms_df = load_farms_and_imagery(args.farms_csv, args.tiff_dir)
    
    print(f"\n📊 Stratified sampling (validation_ratio={args.validation_ratio})...")
    val_df = stratified_sample(farms_df, validation_ratio=args.validation_ratio, seed=args.seed)
    
    save_validation_set(val_df, args.output_path)


if __name__ == "__main__":
    main()
