import logging

import numpy as np
import rasterio

logger = logging.getLogger(__name__)
import glob
import json
import os
import re

import pandas as pd
import rasterio.features

from src.change_detector import FLAG_LEVELS, KNOWN_LEVELS, classify, decision_rule, pixel_area_m2
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

def compute_deforestation_metrics(mask_2020, mask_2024, forest_class=1, pixel_m2=None, rule=None):
    """
    Compare two segmentation masks to detect forest loss, and decide with the rule in force (`classify`).
    """
    forest_2020 = int(np.sum(mask_2020 == forest_class))
    forest_2024 = int(np.sum(mask_2024 == forest_class))

    total_pixels = mask_2020.size
    forest_lost = max(0, forest_2020 - forest_2024)

    deforestation_pct, level, loss_ha = classify(forest_2020, forest_lost, pixel_m2, rule)

    metrics = {
        'forest_pixels_2020': forest_2020,
        'forest_pixels_2024': forest_2024,
        'forest_pixels_lost': forest_lost,
        'forest_loss_ha': round(loss_ha, 4) if loss_ha == loss_ha else loss_ha,
        'deforestation_percent': round(deforestation_pct, 2),
        'forest_coverage_2020_pct': round((forest_2020 / total_pixels) * 100, 2),
        'forest_coverage_2024_pct': round((forest_2024 / total_pixels) * 100, 2),
        'alert_level': level.value,
        'decision_rule': decision_rule(rule),
        'total_pixels': total_pixels
    }
    if level.value == 'UNASSESSED':
        metrics['reason'] = 'pixel size unknown (no transform or CRS), so the 0.5 ha rule cannot be applied'
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

        metrics = compute_deforestation_metrics(mask_2020, mask_2024,
                                                pixel_m2=pixel_area_m2(transform, crs, mask_2020.shape[0]))
        metrics['farm_id'] = farm_id

        # --- Vector Export ---
        if output_vector_dir:
            vector_path = os.path.join(output_vector_dir, f"{farm_id}_deforestation.geojson")
            export_deforestation_vector(mask_2020, mask_2024, transform, crs, vector_path)
            metrics['vector_report'] = vector_path

        return metrics
    except Exception as e:
        logger.error("Analysis failed for %s: %s", farm_id, e)
        return None

def analyze_farm_change(change_mask_path: str, baseline_mask_path: str, farm_id: str, output_vector_dir=None,
                        forest_class: int = 1):
    """Analyze deforestation from a pre-computed binary change mask (1 = forest loss).

    For change-detection models (M3 Siamese, M5 embed-change). Loss is counted inside the farm's 2020 forest (from
    the baseline mask) and classified by the same rule as the two-mask path (`classify`), so a farm gets the same
    alert level whichever model produced it.

    Returns:
        Dict with change metrics, or None on failure.
    """
    try:
        with rasterio.open(change_mask_path) as src:
            change_mask = src.read(1)
            transform = src.transform
            crs = src.crs
        baseline, _, _ = load_mask(baseline_mask_path)
        if baseline.shape != change_mask.shape:
            raise ValueError(f"Dimensions mismatch: {baseline.shape} vs {change_mask.shape}")

        forest_2020 = baseline == forest_class
        lost = int(np.sum((change_mask == 1) & forest_2020))
        pixel_m2 = pixel_area_m2(transform, crs, change_mask.shape[0])
        deforestation_pct, level, loss_ha = classify(int(forest_2020.sum()), lost, pixel_m2)

        metrics = {
            'farm_id': farm_id,
            'forest_pixels_2020': int(forest_2020.sum()),
            'forest_pixels_lost': lost,
            'forest_loss_ha': round(loss_ha, 4) if loss_ha == loss_ha else loss_ha,
            'change_pixels_outside_forest': int(np.sum((change_mask == 1) & ~forest_2020)),
            'deforestation_percent': round(deforestation_pct, 2),
            'alert_level': level.value,
            'decision_rule': decision_rule(),
        }
        if level.value == 'UNASSESSED':
            metrics['reason'] = 'pixel size unknown (no transform or CRS), so the 0.5 ha rule cannot be applied'

        if output_vector_dir:
            os.makedirs(output_vector_dir, exist_ok=True)
            vector_path = os.path.join(output_vector_dir, f"{farm_id}_change.geojson")
            export_deforestation_vector(change_mask, np.zeros_like(change_mask), transform, crs, vector_path, forest_class=1)
            metrics['vector_report'] = vector_path

        return metrics
    except Exception as e:
        logger.error("Change analysis failed for %s: %s", farm_id, e)
        return None


UNASSESSED = 'UNASSESSED'


def _unassessed(farm_id, reason):
    return {'farm_id': farm_id, 'alert_level': UNASSESSED, 'deforestation_percent': float('nan'), 'reason': reason}


def batch_detect_deforestation(baseline_dir, prediction_dir, output_report_path, vector_dir=None):
    """
    Compare all pairs and generate reports.

    Every farm that has a 2024 prediction or a 2020 baseline mask gets a row. Farms that could not be assessed (no
    baseline, no prediction, or the analysis failed) are reported as UNASSESSED with a reason instead of being left
    out, so they cannot disappear from the summary and the due-diligence export.
    """
    logger.info("Starting deforestation detection with refinement")
    predicted_files = glob.glob(os.path.join(prediction_dir, "*_predicted.tif"))

    if not predicted_files:
        logger.warning("No predicted masks found in %s", prediction_dir)
        return None

    if vector_dir:
        os.makedirs(vector_dir, exist_ok=True)

    results = []
    predicted_farms = set()
    for pred_path in sorted(predicted_files):
        filename = os.path.basename(pred_path)
        match = re.match(r"(relation|way)_(\d+)_(\d{4})_.*_predicted\.tif", filename)
        if not match:
            continue

        obj_type, obj_id, year = match.groups()
        farm_id = f"{obj_type}_{obj_id}"
        predicted_farms.add(farm_id)
        baseline_path = os.path.join(baseline_dir, f"{obj_type}_{obj_id}_2020_hybrid.tif")

        if not os.path.exists(baseline_path):
            results.append(_unassessed(farm_id, "no 2020 baseline mask"))
            continue

        result = analyze_farm_pair(baseline_path, pred_path, farm_id, vector_dir)
        results.append(result if result else _unassessed(farm_id, "analysis failed (see log)"))

    for base_path in sorted(glob.glob(os.path.join(baseline_dir, "*_2020_hybrid.tif"))):
        m = re.match(r"(relation|way)_(\d+)_2020_hybrid\.tif", os.path.basename(base_path))
        if m and f"{m.group(1)}_{m.group(2)}" not in predicted_farms:
            results.append(_unassessed(f"{m.group(1)}_{m.group(2)}", "no 2024 prediction"))

    n_unassessed = sum(r['alert_level'] == UNASSESSED for r in results)
    if n_unassessed:
        logger.warning("%d of %d farms could not be assessed (alert_level UNASSESSED)", n_unassessed, len(results))
    if not results:
        return None

    report_df = pd.DataFrame(results)
    key = 'forest_loss_ha' if 'forest_loss_ha' in report_df and report_df['forest_loss_ha'].notna().any() \
        else 'deforestation_percent'
    report_df = report_df.sort_values(key, ascending=False, na_position='last')
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    report_df.to_csv(output_report_path, index=False)

    logger.info("Deforestation report saved to %s", output_report_path)
    return report_df


def generate_summary_stats(report_df, output_json_path):
    def n(level):
        return int((report_df['alert_level'] == level).sum())

    summary = {
        'total_farms': len(report_df),
        'decision_rule': decision_rule(),
        'flagged': n('FLAGGED'),
        'uncertain': n('UNCERTAIN'),
        'clear': n('CLEAR'),
        'violations': n('VIOLATION'),
        'warnings': n('WARNING'),
        'compliant': n('COMPLIANT'),
        'no_baseline_forest': n('NO_BASELINE_FOREST'),
        'unassessed': n(UNASSESSED),
        'other': int((~report_df['alert_level'].isin(KNOWN_LEVELS)).sum()),  # counts sum to total_farms
        'avg_deforestation_pct': float(report_df['deforestation_percent'].mean()),  # assessed farms only (NaN skipped)
        'worst_offender': report_df.iloc[0]['farm_id'] if len(report_df) > 0 else None
    }
    summary['flagged_or_violation'] = sum(n(level) for level in FLAG_LEVELS)
    with open(output_json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    return summary

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prediction_dir = os.getenv('EUDR_PRED_DIR', os.path.join(BASE_DIR, 'data/predictions_2024'))
    report_df = batch_detect_deforestation(
        os.path.join(BASE_DIR, 'data/hybrid_masks'),
        prediction_dir,
        os.path.join(BASE_DIR, 'reports/deforestation_report.csv'),
        vector_dir=os.path.join(BASE_DIR, 'reports/vectors')
    )
    if report_df is not None:
        generate_summary_stats(report_df, os.path.join(BASE_DIR, 'reports/summary_stats.json'))
