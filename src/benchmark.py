import argparse
import csv
import glob
import os
import time
from typing import Dict, List, Optional

import numpy as np
import rasterio


def load_mask(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def compute_segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4, ignore_index: int = 255) -> Dict[str, float]:
    valid = y_true != ignore_index
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    ious: List[float] = []
    forest_class = 1

    forest_tp = float(np.sum((y_true == forest_class) & (y_pred == forest_class)))
    forest_fp = float(np.sum((y_true != forest_class) & (y_pred == forest_class)))
    forest_fn = float(np.sum((y_true == forest_class) & (y_pred != forest_class)))

    forest_precision = forest_tp / (forest_tp + forest_fp + 1e-8)
    forest_recall = forest_tp / (forest_tp + forest_fn + 1e-8)
    forest_f1 = 2.0 * forest_precision * forest_recall / (forest_precision + forest_recall + 1e-8)

    for class_idx in range(num_classes):
        intersection = float(np.sum((y_true == class_idx) & (y_pred == class_idx)))
        union = float(np.sum((y_true == class_idx) | (y_pred == class_idx)))
        if union > 0:
            ious.append(intersection / union)

    miou = float(np.mean(ious)) if ious else 0.0

    return {
        "miou": miou,
        "forest_f1": forest_f1,
    }


def compute_change_f1(mask_2020: np.ndarray, mask_2024_true: np.ndarray, mask_2024_pred: np.ndarray, forest_class: int = 1) -> float:
    true_change = (mask_2020 == forest_class) & (mask_2024_true != forest_class)
    pred_change = (mask_2020 == forest_class) & (mask_2024_pred != forest_class)

    tp = float(np.sum(true_change & pred_change))
    fp = float(np.sum((~true_change) & pred_change))
    fn = float(np.sum(true_change & (~pred_change)))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2.0 * precision * recall / (precision + recall + 1e-8)


def infer_runtime_seconds(num_files: int, elapsed_seconds: float) -> float:
    if num_files <= 0:
        return 0.0
    return elapsed_seconds / num_files


def run_benchmark(
    baseline_2020_dir: str,
    masks_2024_dir: str,
    deeplab_pred_dir: str,
    tessera_pred_dir: str,
    output_csv: str,
    deeplab_elapsed_seconds: Optional[float] = None,
    tessera_elapsed_seconds: Optional[float] = None,
) -> None:
    rows: List[Dict[str, float]] = []

    deeplab_files = sorted(glob.glob(os.path.join(deeplab_pred_dir, "*_predicted.tif")))
    tessera_files = sorted(glob.glob(os.path.join(tessera_pred_dir, "*_predicted.tif")))

    for model_name, pred_files in [("deeplab", deeplab_files), ("tessera", tessera_files)]:
        per_image_metrics = []
        per_image_change_f1 = []

        for pred_path in pred_files:
            filename = os.path.basename(pred_path)
            base_key = filename.replace("_predicted.tif", "")

            mask_2024_path = os.path.join(masks_2024_dir, f"{base_key}_hybrid.tif")
            mask_2020_path = os.path.join(baseline_2020_dir, f"{base_key.replace('_2024_', '_2020_')}_hybrid.tif")

            if not os.path.exists(mask_2024_path) or not os.path.exists(mask_2020_path):
                continue

            y_pred = load_mask(pred_path)
            y_true = load_mask(mask_2024_path)
            y_2020 = load_mask(mask_2020_path)

            metrics = compute_segmentation_metrics(y_true, y_pred)
            change_f1 = compute_change_f1(y_2020, y_true, y_pred)
            per_image_metrics.append(metrics)
            per_image_change_f1.append(change_f1)

        if not per_image_metrics:
            continue

        avg_miou = float(np.mean([m["miou"] for m in per_image_metrics]))
        avg_forest_f1 = float(np.mean([m["forest_f1"] for m in per_image_metrics]))
        avg_change_f1 = float(np.mean(per_image_change_f1))

        elapsed = deeplab_elapsed_seconds if model_name == "deeplab" else tessera_elapsed_seconds
        sec_per_image = infer_runtime_seconds(len(pred_files), elapsed) if elapsed is not None else -1.0

        rows.append(
            {
                "model": model_name,
                "samples": float(len(per_image_metrics)),
                "miou": avg_miou,
                "forest_f1": avg_forest_f1,
                "change_f1": avg_change_f1,
                "sec_per_image": sec_per_image,
            }
        )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["model", "samples", "miou", "forest_f1", "change_f1", "sec_per_image"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Benchmark report saved to {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DeepLab vs TESSERA predictions.")
    parser.add_argument("--baseline-2020-dir", required=True)
    parser.add_argument("--masks-2024-dir", required=True)
    parser.add_argument("--deeplab-pred-dir", required=True)
    parser.add_argument("--tessera-pred-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--deeplab-elapsed-seconds", type=float, default=None)
    parser.add_argument("--tessera-elapsed-seconds", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        baseline_2020_dir=args.baseline_2020_dir,
        masks_2024_dir=args.masks_2024_dir,
        deeplab_pred_dir=args.deeplab_pred_dir,
        tessera_pred_dir=args.tessera_pred_dir,
        output_csv=args.output_csv,
        deeplab_elapsed_seconds=args.deeplab_elapsed_seconds,
        tessera_elapsed_seconds=args.tessera_elapsed_seconds,
    )


if __name__ == "__main__":
    main()
