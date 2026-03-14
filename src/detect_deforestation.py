import numpy as np
import rasterio
import rasterio.features
import pandas as pd
import os
import glob
import re
from pathlib import Path
import json
from src.postprocessing.refiner import refine_mask

def load_mask(mask_path):
    """
    Load a segmentation mask from disk.
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
    return mask, transform, crs

def export_deforestation_vector(mask_2020, mask_2024, transform, crs, output_path, forest_class=1):
    """
    Export changed forest pixels as a vectorized GeoJSON.
    """
    # Detect loss: was forest (1) in 2020, not forest (not 1) in 2024
    loss_mask = ((mask_2020 == forest_class) & (mask_2024 != forest_class)).astype(np.uint8)
    
    if np.sum(loss_mask) == 0:
        return None

    # Vectorize
    shapes = rasterio.features.shapes(loss_mask, mask=loss_mask, transform=transform)
    
    features = []
    for geom, val in shapes:
        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {"class": "deforestation", "value": val}
        })
    
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {
            "type": "name",
            "properties": {"name": str(crs)}
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(geojson, f)
    
    return output_path

def compute_deforestation_metrics(mask_2020, mask_2024, forest_class=1):
    """
    Compare two segmentation masks to detect forest loss.
    """
    forest_2020 = int(np.sum(mask_2020 == forest_class))
    forest_2024 = int(np.sum(mask_2024 == forest_class))
    
    total_pixels = mask_2020.size
    forest_lost = max(0, forest_2020 - forest_2024)
    
    if forest_2020 == 0:
        deforestation_pct = 0.0
        alert_level = 'NO_BASELINE_FOREST'
    else:
        deforestation_pct = (forest_lost / forest_2020) * 100
        if deforestation_pct > 10: alert_level = 'VIOLATION'
        elif deforestation_pct > 5: alert_level = 'WARNING'
        else: alert_level = 'COMPLIANT'
    
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

def analyze_farm_pair(baseline_mask_path, predicted_mask_path, farm_id, output_vector_dir=None):
    """
    Analyze a single farm's deforestation with post-processing and vector output.
    """
    try:
        mask_2020, transform, crs = load_mask(baseline_mask_path)
        mask_2024, _, _ = load_mask(predicted_mask_path)
        
        # --- Post-processing ---
        mask_2024 = refine_mask(mask_2024)
        
        if mask_2020.shape != mask_2024.shape:
            raise ValueError(f"Dimensions mismatch: {mask_2020.shape} vs {mask_2024.shape}")
        
        metrics = compute_deforestation_metrics(mask_2020, mask_2024)
        metrics['farm_id'] = farm_id
        
        # --- Vector Export ---
        if output_vector_dir:
            vector_path = os.path.join(output_vector_dir, f"{farm_id}_deforestation.geojson")
            export_deforestation_vector(mask_2020, mask_2024, transform, crs, vector_path)
            metrics['vector_report'] = vector_path
        
        return metrics
    except Exception as e:
        print(f"❌ Analysis failed for {farm_id}: {e}")
        return None

def batch_detect_deforestation(baseline_dir, prediction_dir, output_report_path, vector_dir=None):
    """
    Compare all pairs and generate reports.
    """
    print("🔍 Starting Deforestation Detection with Refinement...")
    predicted_files = glob.glob(os.path.join(prediction_dir, "*_predicted.tif"))
    
    if not predicted_files:
        print(f"⚠️ No predicted masks found in {prediction_dir}")
        return None
    
    if vector_dir: os.makedirs(vector_dir, exist_ok=True)
    
    results = []
    for pred_path in predicted_files:
        filename = os.path.basename(pred_path)
        match = re.match(r"(relation|way)_(\d+)_(\d{4})_.*_predicted\.tif", filename)
        if not match: continue
        
        obj_type, obj_id, year = match.groups()
        farm_id = f"{obj_type}_{obj_id}"
        baseline_path = os.path.join(baseline_dir, f"{obj_type}_{obj_id}_2020_hybrid.tif")
        
        if not os.path.exists(baseline_path): continue
        
        result = analyze_farm_pair(baseline_path, pred_path, farm_id, vector_dir)
        if result: results.append(result)
    
    if not results: return None
    
    report_df = pd.DataFrame(results)
    report_df = report_df.sort_values('deforestation_percent', ascending=False)
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    report_df.to_csv(output_report_path, index=False)
    
    print(f"\n✅ Deforestation Report saved to {output_report_path}")
    return report_df

def generate_summary_stats(report_df, output_json_path):
    summary = {
        'total_farms': len(report_df),
        'violations': int(len(report_df[report_df['alert_level'] == 'VIOLATION'])),
        'warnings': int(len(report_df[report_df['alert_level'] == 'WARNING'])),
        'compliant': int(len(report_df[report_df['alert_level'] == 'COMPLIANT'])),
        'avg_deforestation_pct': float(report_df['deforestation_percent'].mean()),
        'worst_offender': report_df.iloc[0]['farm_id'] if len(report_df) > 0 else None
    }
    with open(output_json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    return summary

if __name__ == "__main__":
    BASE_DIR = r'd:\Work\Segmentation-logistics\EUDR-compliance'
    report_df = batch_detect_deforestation(
        os.path.join(BASE_DIR, 'data/hybrid_masks'),
        os.path.join(BASE_DIR, 'data/predictions_2024'),
        os.path.join(BASE_DIR, 'reports/deforestation_report.csv'),
        vector_dir=os.path.join(BASE_DIR, 'reports/vectors')
    )
    if report_df is not None:
        generate_summary_stats(report_df, os.path.join(BASE_DIR, 'reports/summary_stats.json'))
