"""
Data quality filtering and label enrichment for training data.
Flags and optionally removes low-quality image-mask pairs.
"""

import argparse
import ast
import glob
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio


def load_image(tiff_path: str) -> Tuple[np.ndarray, int]:
    """
    Load a Sentinel-2 GeoTIFF and return pixel data + band count.
    """
    with rasterio.open(tiff_path) as src:
        data = src.read()  # (C, H, W)
    return data, data.shape[0]


def load_mask(tiff_path: str) -> np.ndarray:
    """
    Load a segmentation mask GeoTIFF (single band).
    """
    with rasterio.open(tiff_path) as src:
        mask = src.read(1)  # (H, W)
    return mask


def estimate_cloud_cover(image: np.ndarray) -> float:
    """
    Estimate cloud cover from SCL band (band 5, 0-indexed).
    SCL values >= 8 are typically clouds/shadows.
    
    Args:
        image: Shape (C, H, W) with C >= 5
    
    Returns:
        Cloud cover percentage (0-100)
    """
    if image.shape[0] < 5:
        return -1.0  # Unable to estimate
    
    scl = image[4, :, :]  # SCL is band 5 (0-indexed)
    cloud_mask = scl >= 8
    cloud_pct = float(np.sum(cloud_mask)) / cloud_mask.size * 100
    return cloud_pct


def estimate_no_data_ratio(image: np.ndarray) -> float:
    """
    Estimate no-data ratio from no-data pixels (value 0 across all bands).
    
    Args:
        image: Shape (C, H, W)
    
    Returns:
        No-data ratio (0-100)
    """
    # Pixels with all bands == 0 are likely no-data
    no_data_mask = np.all(image == 0, axis=0)
    no_data_pct = float(np.sum(no_data_mask)) / no_data_mask.size * 100
    return no_data_pct


def check_alignment(image_shape: Tuple[int, int], mask_shape: Tuple[int, int]) -> bool:
    """
    Check if image and mask spatial dimensions match (alignment check).
    
    Returns:
        True if aligned, False otherwise
    """
    return image_shape == mask_shape


def quality_check(
    image_path: str,
    mask_path: str,
    cloud_threshold: float = 20.0,
    no_data_threshold: float = 30.0,
    fail_on_shape_mismatch: bool = False,
) -> Dict[str, Any]:
    """
    Perform quality checks on an image-mask pair.
    
    Args:
        image_path: Path to Sentinel-2 GeoTIFF
        mask_path: Path to segmentation mask GeoTIFF
        cloud_threshold: Max allowed cloud cover (%)
        no_data_threshold: Max allowed no-data ratio (%)
    
    Returns:
        Dict with quality metrics and pass/fail flags
    """
    result = {
        "image_path": image_path,
        "mask_path": mask_path,
        "pass": True,
        "cloud_cover_pct": -1.0,
        "no_data_ratio_pct": -1.0,
        "alignment_ok": False,
        "failures": []
    }
    
    try:
        image, band_count = load_image(image_path)
        mask = load_mask(mask_path)
        
        # Store metadata
        result["band_count"] = band_count
        result["image_shape"] = image.shape
        result["mask_shape"] = mask.shape
        
        # Cloud cover
        cloud_pct = estimate_cloud_cover(image)
        result["cloud_cover_pct"] = cloud_pct
        if cloud_pct >= 0 and cloud_pct > cloud_threshold:
            result["pass"] = False
            result["failures"].append(f"Cloud cover {cloud_pct:.1f}% > threshold {cloud_threshold}%")
        
        # No-data ratio
        no_data_pct = estimate_no_data_ratio(image)
        result["no_data_ratio_pct"] = no_data_pct
        if no_data_pct > no_data_threshold:
            result["pass"] = False
            result["failures"].append(f"No-data ratio {no_data_pct:.1f}% > threshold {no_data_threshold}%")
        
        # Alignment
        alignment_ok = check_alignment(image.shape[-2:], mask.shape)
        result["alignment_ok"] = alignment_ok
        if not alignment_ok and fail_on_shape_mismatch:
            result["pass"] = False
            result["failures"].append(f"Spatial mismatch: image {image.shape[-2:]} vs mask {mask.shape}")
        elif not alignment_ok:
            result["failures"].append(f"Spatial mismatch noted: image {image.shape[-2:]} vs mask {mask.shape}")
        
    except Exception as e:
        result["pass"] = False
        result["failures"].append(f"Load error: {str(e)}")
    
    return result


def batch_quality_check(
    baseline_dir: str,
    mask_dir: str,
    cloud_threshold: float = 20.0,
    no_data_threshold: float = 30.0,
    fail_on_shape_mismatch: bool = False,
) -> pd.DataFrame:
    """
    Check quality for all image-mask pairs in directories.
    
    Assumes filename pattern: {farm_id}_2020.tiff and {farm_id}_hybrid.tiff
    
    Args:
        baseline_dir: Directory with 2020 baseline images
        mask_dir: Directory with hybrid masks
        cloud_threshold: Max allowed cloud cover (%)
        no_data_threshold: Max allowed no-data ratio (%)
    
    Returns:
        DataFrame with quality check results for all pairs
    """
    results = []
    
    image_files = sorted(glob.glob(os.path.join(baseline_dir, "*_2020*.tiff")))
    image_files.extend(sorted(glob.glob(os.path.join(baseline_dir, "*_2020*.tif"))))
    print(f"Found {len(image_files)} baseline images to check")
    
    for i, image_path in enumerate(image_files, 1):
        base_name = os.path.basename(image_path)
        # Extract farm_id from patterns like:
        # relation_123_2020_2020-06-07.tiff -> relation_123
        # relation_123_2020.tif -> relation_123
        farm_id_match = re.match(r"(.*?)_2020(?:_\d{4}-\d{2}-\d{2})?\.tiff?$", base_name)
        if not farm_id_match:
            continue
        
        farm_id = farm_id_match.group(1)
        mask_path = os.path.join(mask_dir, f"{farm_id}_2020_hybrid.tif")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, f"{farm_id}_2020_hybrid.tiff")
        
        if not os.path.exists(mask_path):
            print(f"[{i}/{len(image_files)}] ⚠️  Missing mask for {farm_id}")
            continue
        
        result = quality_check(
            image_path=image_path,
            mask_path=mask_path,
            cloud_threshold=cloud_threshold,
            no_data_threshold=no_data_threshold,
            fail_on_shape_mismatch=fail_on_shape_mismatch,
        )
        results.append(result)
        
        if result["pass"]:
            status = "✅"
        else:
            status = "❌"
        print(f"[{i}/{len(image_files)}] {status} {farm_id}")
    
    df = pd.DataFrame(results)
    return df


def save_quality_report(df: pd.DataFrame, output_path: str) -> None:
    """
    Save quality check results to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Quality report saved to {output_path}")
    
    # Print summary
    num_pass = int(df["pass"].sum())
    num_fail = len(df) - num_pass
    print(f"\nSummary: {num_pass} passed, {num_fail} failed")
    if num_fail > 0:
        print("\nFailure reasons:")
        for failure in df[~df["pass"]]["failures"]:
            if isinstance(failure, list):
                reasons = failure
            elif isinstance(failure, str):
                try:
                    parsed = ast.literal_eval(failure)
                    reasons = parsed if isinstance(parsed, list) else [failure]
                except (ValueError, SyntaxError):
                    reasons = [failure]
            else:
                reasons = [str(failure)]

            for reason in reasons:
                print(f"  - {reason}")


def filter_low_quality_pairs(
    quality_df: pd.DataFrame,
    baseline_dir: str,
    mask_dir: str,
    output_baseline_dir: str,
    output_mask_dir: str,
) -> None:
    """
    Copy only high-quality pairs to output directories.
    
    Args:
        quality_df: DataFrame from batch_quality_check()
        baseline_dir: Original baseline image directory
        mask_dir: Original mask directory
        output_baseline_dir: Output baseline directory (high-quality only)
        output_mask_dir: Output mask directory (high-quality only)
    """
    os.makedirs(output_baseline_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    passed_pairs = quality_df[quality_df["pass"]]
    print(f"\nCopying {len(passed_pairs)} high-quality pairs...")
    
    for _, row in passed_pairs.iterrows():
        image_base = os.path.basename(row["image_path"])
        mask_base = os.path.basename(row["mask_path"])
        
        src_image = os.path.join(baseline_dir, image_base)
        dst_image = os.path.join(output_baseline_dir, image_base)
        
        src_mask = os.path.join(mask_dir, mask_base)
        dst_mask = os.path.join(output_mask_dir, mask_base)
        
        # Simple copy (could use shutil.copy2 for metadata)
        import shutil
        shutil.copy2(src_image, dst_image)
        shutil.copy2(src_mask, dst_mask)
    
    print(f"✅ Filtered pairs saved to {output_baseline_dir} and {output_mask_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality filter for image-mask training pairs.")
    parser.add_argument('--baseline-dir', required=True, help='Directory with 2020 baseline images')
    parser.add_argument('--mask-dir', required=True, help='Directory with hybrid masks')
    parser.add_argument('--output-report', required=True, help='Output CSV path for quality report')
    parser.add_argument('--cloud-threshold', type=float, default=20.0, help='Max cloud cover % (default 20)')
    parser.add_argument('--no-data-threshold', type=float, default=30.0, help='Max no-data % (default 30)')
    parser.add_argument('--fail-on-shape-mismatch', action='store_true', help='If set, shape mismatch fails the pair')
    parser.add_argument('--filter-output-baseline', default=None, help='Optional: save filtered images here')
    parser.add_argument('--filter-output-masks', default=None, help='Optional: save filtered masks here')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("🔍 Running quality checks on all image-mask pairs...")
    quality_df = batch_quality_check(
        baseline_dir=args.baseline_dir,
        mask_dir=args.mask_dir,
        cloud_threshold=args.cloud_threshold,
        no_data_threshold=args.no_data_threshold,
        fail_on_shape_mismatch=args.fail_on_shape_mismatch,
    )
    
    save_quality_report(quality_df, args.output_report)
    
    if args.filter_output_baseline and args.filter_output_masks:
        filter_low_quality_pairs(
            quality_df=quality_df,
            baseline_dir=args.baseline_dir,
            mask_dir=args.mask_dir,
            output_baseline_dir=args.filter_output_baseline,
            output_mask_dir=args.filter_output_masks,
        )


if __name__ == "__main__":
    main()
