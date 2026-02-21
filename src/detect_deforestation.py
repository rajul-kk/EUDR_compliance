
import numpy as np
import rasterio
import pandas as pd
import os
import glob
import re
from pathlib import Path
import json

def load_mask(mask_path):
    """
    Load a segmentation mask from disk.
    
    Args:
        mask_path: Path to .tif file
    
    Returns:
        mask: Numpy array (H, W)
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return mask

def compute_class_pixels(mask, class_id):
    """
    Count pixels belonging to a specific class.
    
    Args:
        mask: Numpy array (H, W)
        class_id: Integer class label
    
    Returns:
        count: Number of pixels with this class
    """
    return int(np.sum(mask == class_id))

def compute_deforestation_metrics(mask_2020, mask_2024, forest_class=1):
    """
    Compare two segmentation masks to detect forest loss.
    
    Args:
        mask_2020: Baseline mask (H, W)
        mask_2024: Current mask (H, W)
        forest_class: Class ID for forest (default: 1)
    
    Returns:
        metrics: Dictionary with deforestation statistics
    """
    # Count forest pixels in each year
    forest_2020 = compute_class_pixels(mask_2020, forest_class)
    forest_2024 = compute_class_pixels(mask_2024, forest_class)
    
    # Total pixels (for validation)
    total_pixels = mask_2020.size
    
    # Calculate metrics
    forest_lost = max(0, forest_2020 - forest_2024)  # Can't lose negative forest
    
    if forest_2020 == 0:
        # No forest in baseline, cannot compute deforestation
        deforestation_pct = 0.0
        alert_level = 'NO_BASELINE_FOREST'
    else:
        deforestation_pct = (forest_lost / forest_2020) * 100
        
        # EUDR compliance thresholds (example values)
        if deforestation_pct > 10:
            alert_level = 'VIOLATION'
        elif deforestation_pct > 5:
            alert_level = 'WARNING'
        else:
            alert_level = 'COMPLIANT'
    
    metrics = {
        'forest_pixels_2020': forest_2020,
        'forest_pixels_2024': forest_2024,
        'forest_pixels_lost': forest_lost,
        'deforestation_percent': round(deforestation_pct, 2),
        'forest_coverage_2020_pct': round((forest_2020 / total_pixels) * 100, 2),
        'forest_coverage_2024_pct': round((forest_2024 / total_pixels) * 100, 2),
        'alert_level': alert_level,
        'total_pixels': total_pixels
    }
    
    return metrics

def analyze_farm_pair(baseline_mask_path, predicted_mask_path, farm_id):
    """
    Analyze a single farm's deforestation.
    
    Args:
        baseline_mask_path: Path to 2020 GEE hybrid mask
        predicted_mask_path: Path to 2024 predicted mask
        farm_id: Identifier for this farm
    
    Returns:
        result: Dictionary with farm ID and metrics
    """
    try:
        mask_2020 = load_mask(baseline_mask_path)
        mask_2024 = load_mask(predicted_mask_path)
        
        # Validate dimensions match
        if mask_2020.shape != mask_2024.shape:
            raise ValueError(f"Mask dimensions don't match: {mask_2020.shape} vs {mask_2024.shape}")
        
        metrics = compute_deforestation_metrics(mask_2020, mask_2024)
        metrics['farm_id'] = farm_id
        
        return metrics
        
    except Exception as e:
        print(f"❌ Failed to analyze {farm_id}: {e}")
        return None

def batch_detect_deforestation(baseline_dir, prediction_dir, output_report_path):
    """
    Compare all 2020 baseline masks with 2024 predicted masks.
    
    Args:
        baseline_dir: Directory with 2020 GEE hybrid masks
        prediction_dir: Directory with 2024 predicted masks
        output_report_path: Path to save CSV report
    
    Returns:
        report_df: Pandas DataFrame with results
    """
    print("🔍 Starting Deforestation Detection...")
    
    # Find all predicted masks (2024)
    predicted_files = glob.glob(os.path.join(prediction_dir, "*_predicted.tif"))
    
    if not predicted_files:
        print(f"⚠️ No predicted masks found in {prediction_dir}")
        return None
    
    print(f"Found {len(predicted_files)} predicted masks for 2024.")
    
    results = []
    
    for pred_path in predicted_files:
        # Parse farm ID from filename
        # Expected: {type}_{id}_2024_{date}_predicted.tif
        filename = os.path.basename(pred_path)
        match = re.match(r"(relation|way)_(\d+)_(\d{4})_.*_predicted\.tif", filename)
        
        if not match:
            print(f"⚠️ Skipping unrecognized filename: {filename}")
            continue
        
        obj_type, obj_id, year = match.groups()
        farm_id = f"{obj_type}_{obj_id}"
        
        # Find corresponding 2020 baseline mask
        # Expected: {type}_{id}_2020_hybrid.tif
        baseline_name = f"{obj_type}_{obj_id}_2020_hybrid.tif"
        baseline_path = os.path.join(baseline_dir, baseline_name)
        
        if not os.path.exists(baseline_path):
            print(f"⚠️ No baseline mask found for {farm_id}, skipping...")
            continue
        
        # Analyze this farm
        result = analyze_farm_pair(baseline_path, pred_path, farm_id)
        if result:
            results.append(result)
    
    if not results:
        print("❌ No valid farm pairs found.")
        return None
    
    # Create DataFrame
    report_df = pd.DataFrame(results)
    
    # Sort by deforestation percentage (worst first)
    report_df = report_df.sort_values('deforestation_percent', ascending=False)
    
    # Save report
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    report_df.to_csv(output_report_path, index=False)
    
    # Print summary
    print(f"\n📊 Deforestation Analysis Summary:")
    print(f"   Total Farms Analyzed: {len(report_df)}")
    print(f"   🚨 VIOLATIONS: {len(report_df[report_df['alert_level'] == 'VIOLATION'])}")
    print(f"   ⚠️  WARNINGS: {len(report_df[report_df['alert_level'] == 'WARNING'])}")
    print(f"   ✅ COMPLIANT: {len(report_df[report_df['alert_level'] == 'COMPLIANT'])}")
    print(f"\n✅ Report saved to {output_report_path}")
    
    return report_df

def generate_summary_stats(report_df, output_json_path):
    """
    Generate summary statistics from the deforestation report.
    
    Args:
        report_df: DataFrame from batch_detect_deforestation
        output_json_path: Path to save JSON summary
    """
    summary = {
        'total_farms': len(report_df),
        'violations': int(len(report_df[report_df['alert_level'] == 'VIOLATION'])),
        'warnings': int(len(report_df[report_df['alert_level'] == 'WARNING'])),
        'compliant': int(len(report_df[report_df['alert_level'] == 'COMPLIANT'])),
        'avg_deforestation_pct': float(report_df['deforestation_percent'].mean()),
        'max_deforestation_pct': float(report_df['deforestation_percent'].max()),
        'worst_offender': report_df.iloc[0]['farm_id'] if len(report_df) > 0 else None
    }
    
    with open(output_json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary stats saved to {output_json_path}")
    
    return summary

if __name__ == "__main__":
    # Configuration
    BASELINE_DIR = r'd:\Work\EUDR-compliance\data\hybrid_masks'
    PREDICTION_DIR = r'd:\Work\EUDR-compliance\data\predictions_2024'
    REPORT_PATH = r'd:\Work\EUDR-compliance\reports\deforestation_report.csv'
    SUMMARY_PATH = r'd:\Work\EUDR-compliance\reports\summary_stats.json'
    
    # Run detection
    report_df = batch_detect_deforestation(BASELINE_DIR, PREDICTION_DIR, REPORT_PATH)
    
    if report_df is not None:
        # Generate summary
        generate_summary_stats(report_df, SUMMARY_PATH)
        
        # Display top 5 worst offenders
        print("\n🔥 Top 5 Farms with Highest Deforestation:")
        print(report_df[['farm_id', 'deforestation_percent', 'alert_level']].head())
