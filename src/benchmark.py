import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio


def load_mask(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def find_mask(mask_dir: str, farm_key: str, year: str) -> Optional[str]:
    candidates = [
        os.path.join(mask_dir, f"{farm_key}_{year}_hybrid.tif"),
        os.path.join(mask_dir, f"{farm_key}_{year}_hybrid.tiff"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def find_tile_mask(mask_dir: str, stem: str) -> Optional[str]:
    candidates = [
        os.path.join(mask_dir, f"{stem}_mask.tif"),
        os.path.join(mask_dir, f"{stem}_mask.tiff"),
        os.path.join(mask_dir, f"{stem}.tif"),
        os.path.join(mask_dir, f"{stem}.tiff"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


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


def precision_recall_f1(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def extract_farm_key_from_prediction_name(filename: str) -> str:
    match = re.match(r"^(relation|way)_\d+", filename)
    if not match:
        return ""
    return match.group(0)


def extract_prediction_key(filename: str) -> str:
    base = filename.replace("_predicted.tif", "").replace("_predicted.tiff", "")
    farm_key = extract_farm_key_from_prediction_name(base)
    if farm_key:
        return farm_key
    if base.startswith("grid_"):
        return base
    return ""


def load_allowed_keys(split_manifest_path: Optional[str], split_name: str) -> Optional[set]:
    if not split_manifest_path:
        return None

    with open(split_manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    splits = payload.get("splits", {})
    keys = splits.get(split_name)
    if keys is None:
        raise ValueError(f"Split '{split_name}' not found in manifest: {split_manifest_path}")
    return set(keys)


def load_crop_map(farms_csv: str) -> Dict[str, str]:
    if not farms_csv or not os.path.exists(farms_csv):
        return {}

    farms_df = pd.read_csv(farms_csv)
    if "farm_id" not in farms_df.columns or "crop_type" not in farms_df.columns:
        return {}

    return {str(row["farm_id"]): str(row["crop_type"]) for _, row in farms_df.iterrows()}


def compute_forest_stats(y_true: np.ndarray, y_pred: np.ndarray, forest_class: int = 1, ignore_index: int = 255) -> Dict[str, float]:
    valid = y_true != ignore_index
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    tp = float(np.sum((y_true == forest_class) & (y_pred == forest_class)))
    fp = float(np.sum((y_true != forest_class) & (y_pred == forest_class)))
    fn = float(np.sum((y_true == forest_class) & (y_pred != forest_class)))
    return {"tp": tp, "fp": fp, "fn": fn}


def compute_change_stats(mask_2020: np.ndarray, mask_2024_true: np.ndarray, mask_2024_pred: np.ndarray, forest_class: int = 1) -> Dict[str, float]:
    true_change = (mask_2020 == forest_class) & (mask_2024_true != forest_class)
    pred_change = (mask_2020 == forest_class) & (mask_2024_pred != forest_class)

    tp = float(np.sum(true_change & pred_change))
    fp = float(np.sum((~true_change) & pred_change))
    fn = float(np.sum(true_change & (~pred_change)))
    return {"tp": tp, "fp": fp, "fn": fn}


def align_shapes(mask_a: np.ndarray, mask_b: np.ndarray, mask_c: Optional[np.ndarray] = None):
    h = min(mask_a.shape[0], mask_b.shape[0])
    w = min(mask_a.shape[1], mask_b.shape[1])
    if mask_c is not None:
        h = min(h, mask_c.shape[0])
        w = min(w, mask_c.shape[1])
        return mask_a[:h, :w], mask_b[:h, :w], mask_c[:h, :w]
    return mask_a[:h, :w], mask_b[:h, :w]


def run_baseline_metrics(
    prediction_dir: str,
    mask_dir: str,
    farms_csv: str,
    output_csv: str,
    model_name: str = "deeplab",
    train_seconds: Optional[float] = None,
    inference_seconds: Optional[float] = None,
    split_manifest_path: Optional[str] = None,
    split_name: str = "val",
) -> None:
    pred_files = sorted(glob.glob(os.path.join(prediction_dir, "*_predicted.tif")))
    if not pred_files:
        raise FileNotFoundError(f"No predictions found in {prediction_dir}")

    allowed_keys = load_allowed_keys(split_manifest_path, split_name)
    if allowed_keys is not None:
        pred_keys = set()
        for pred_path in pred_files:
            pred_key = extract_prediction_key(os.path.basename(pred_path))
            if pred_key:
                pred_keys.add(pred_key)

        matched_keys = allowed_keys.intersection(pred_keys)
        if not matched_keys:
            raise RuntimeError(
                f"Split-manifest consistency check failed: no predictions found for split '{split_name}' "
                f"from {split_manifest_path}"
            )

    crop_map = load_crop_map(farms_csv)
    overall = {
        "samples": 0,
        "miou_sum": 0.0,
        "forest_tp": 0.0,
        "forest_fp": 0.0,
        "forest_fn": 0.0,
        "change_tp": 0.0,
        "change_fp": 0.0,
        "change_fn": 0.0,
    }
    by_crop = defaultdict(lambda: {
        "samples": 0,
        "miou_sum": 0.0,
        "forest_tp": 0.0,
        "forest_fp": 0.0,
        "forest_fn": 0.0,
        "change_tp": 0.0,
        "change_fp": 0.0,
        "change_fn": 0.0,
    })

    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        pred_key = extract_prediction_key(filename)
        if not pred_key:
            continue

        if allowed_keys is not None and pred_key not in allowed_keys:
            continue

        farm_key = extract_farm_key_from_prediction_name(pred_key)
        is_grid = pred_key.startswith("grid_")

        if is_grid:
            mask_2024_path = find_tile_mask(mask_dir, pred_key)
            mask_2020_path = None
        else:
            mask_2024_path = find_mask(mask_dir, farm_key, "2024")
            mask_2020_path = find_mask(mask_dir, farm_key, "2020")

        if mask_2024_path is None:
            continue

        y_pred = load_mask(pred_path)
        y_true = load_mask(mask_2024_path)
        y_2020 = load_mask(mask_2020_path) if mask_2020_path is not None else None
        if y_2020 is not None:
            y_pred, y_true, y_2020 = align_shapes(y_pred, y_true, y_2020)
        else:
            y_pred, y_true = align_shapes(y_pred, y_true)

        seg_metrics = compute_segmentation_metrics(y_true, y_pred)
        forest = compute_forest_stats(y_true, y_pred)
        change = compute_change_stats(y_2020, y_true, y_pred) if y_2020 is not None else {"tp": 0.0, "fp": 0.0, "fn": 0.0}

        overall["samples"] += 1
        overall["miou_sum"] += seg_metrics["miou"]
        overall["forest_tp"] += forest["tp"]
        overall["forest_fp"] += forest["fp"]
        overall["forest_fn"] += forest["fn"]
        overall["change_tp"] += change["tp"]
        overall["change_fp"] += change["fp"]
        overall["change_fn"] += change["fn"]

        crop = crop_map.get(f"osm_{farm_key}", "UNKNOWN") if farm_key else "GRID"
        crop_acc = by_crop[crop]
        crop_acc["samples"] += 1
        crop_acc["miou_sum"] += seg_metrics["miou"]
        crop_acc["forest_tp"] += forest["tp"]
        crop_acc["forest_fp"] += forest["fp"]
        crop_acc["forest_fn"] += forest["fn"]
        crop_acc["change_tp"] += change["tp"]
        crop_acc["change_fp"] += change["fp"]
        crop_acc["change_fn"] += change["fn"]

    if overall["samples"] == 0:
        raise RuntimeError("No valid prediction/mask pairs found for baseline metrics.")

    rows: List[Dict[str, object]] = []
    forest_precision, forest_recall, forest_f1 = precision_recall_f1(overall["forest_tp"], overall["forest_fp"], overall["forest_fn"])
    change_precision, change_recall, change_f1 = precision_recall_f1(overall["change_tp"], overall["change_fp"], overall["change_fn"])
    rows.append(
        {
            "scope": "overall",
            "model": model_name,
            "crop_type": "ALL",
            "samples": overall["samples"],
            "miou": overall["miou_sum"] / overall["samples"],
            "forest_precision": forest_precision,
            "forest_recall": forest_recall,
            "forest_f1": forest_f1,
            "change_precision": change_precision,
            "change_recall": change_recall,
            "change_f1": change_f1,
            "train_seconds": train_seconds if train_seconds is not None else "",
            "inference_seconds": inference_seconds if inference_seconds is not None else "",
        }
    )

    for crop, acc in sorted(by_crop.items(), key=lambda x: x[0]):
        if acc["samples"] == 0:
            continue
        crop_forest_precision, crop_forest_recall, crop_forest_f1 = precision_recall_f1(acc["forest_tp"], acc["forest_fp"], acc["forest_fn"])
        crop_change_precision, crop_change_recall, crop_change_f1 = precision_recall_f1(acc["change_tp"], acc["change_fp"], acc["change_fn"])
        rows.append(
            {
                "scope": "crop",
                "model": model_name,
                "crop_type": crop,
                "samples": acc["samples"],
                "miou": acc["miou_sum"] / acc["samples"],
                "forest_precision": crop_forest_precision,
                "forest_recall": crop_forest_recall,
                "forest_f1": crop_forest_f1,
                "change_precision": crop_change_precision,
                "change_recall": crop_change_recall,
                "change_f1": crop_change_f1,
                "train_seconds": "",
                "inference_seconds": "",
            }
        )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "scope",
                "model",
                "crop_type",
                "samples",
                "miou",
                "forest_precision",
                "forest_recall",
                "forest_f1",
                "change_precision",
                "change_recall",
                "change_f1",
                "train_seconds",
                "inference_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Baseline metrics report saved to {output_csv}")


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


ALL_MODEL_TYPES = ["deeplab", "tessera", "tessera-embed", "siamese", "embed-change", "hybrid"]

# Models that output a change map directly (binary: 0=no-change, 1=forest-loss)
CHANGE_DETECTION_MODELS = {"siamese", "embed-change"}

# Models that still do two-stage segmentation but can be benchmarked via run_baseline_metrics
TWO_STAGE_MODELS = {"deeplab", "tessera", "tessera-embed", "hybrid"}


def run_change_detection_benchmark(
    change_pred_dir: str,
    mask_t1_dir: str,
    mask_t2_dir: str,
    farms_csv: str,
    output_csv: str,
    model_name: str,
    train_seconds: Optional[float] = None,
    inference_seconds: Optional[float] = None,
) -> None:
    """Benchmark models that directly output a binary change map.

    Args:
        change_pred_dir: Directory with *_change.tif predictions (1=forest-loss, 0=no-change).
        mask_t1_dir:     Directory with 2020 hybrid masks (ground truth for t1 forest cover).
        mask_t2_dir:     Directory with 2024 hybrid masks (ground truth for t2 forest cover).
    """
    pred_files = sorted(glob.glob(os.path.join(change_pred_dir, "*.tif")))
    if not pred_files:
        raise FileNotFoundError(f"No change prediction .tif files found in {change_pred_dir}")

    crop_map = load_crop_map(farms_csv)
    overall = {"samples": 0, "change_tp": 0.0, "change_fp": 0.0, "change_fn": 0.0}
    by_crop: Dict = defaultdict(lambda: {"samples": 0, "change_tp": 0.0, "change_fp": 0.0, "change_fn": 0.0})

    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        farm_key = extract_farm_key_from_prediction_name(filename)
        if not farm_key:
            continue

        mask_t1_path = find_mask(mask_t1_dir, farm_key, "2020")
        mask_t2_path = find_mask(mask_t2_dir, farm_key, "2024")
        if mask_t1_path is None or mask_t2_path is None:
            continue

        y_pred_change = load_mask(pred_path)  # 1 = predicted forest loss
        y_t1 = load_mask(mask_t1_path)
        y_t2 = load_mask(mask_t2_path)

        y_pred_change, y_t1, y_t2 = align_shapes(y_pred_change, y_t1, y_t2)

        # Ground truth change: was forest in 2020, not forest in 2024
        gt_change = (y_t1 == 1) & (y_t2 != 1) & (y_t2 != 255)
        pred_change = y_pred_change == 1

        tp = float(np.sum(gt_change & pred_change))
        fp = float(np.sum((~gt_change) & pred_change))
        fn = float(np.sum(gt_change & (~pred_change)))

        overall["samples"] += 1
        overall["change_tp"] += tp
        overall["change_fp"] += fp
        overall["change_fn"] += fn

        crop = crop_map.get(f"osm_{farm_key}", "UNKNOWN")
        by_crop[crop]["samples"] += 1
        by_crop[crop]["change_tp"] += tp
        by_crop[crop]["change_fp"] += fp
        by_crop[crop]["change_fn"] += fn

    if overall["samples"] == 0:
        raise RuntimeError("No valid change prediction / mask pairs found.")

    rows: List[Dict] = []
    cp, cr, cf1 = precision_recall_f1(overall["change_tp"], overall["change_fp"], overall["change_fn"])
    rows.append({
        "scope": "overall", "model": model_name, "crop_type": "ALL",
        "samples": overall["samples"], "miou": "",
        "forest_precision": "", "forest_recall": "", "forest_f1": "",
        "change_precision": cp, "change_recall": cr, "change_f1": cf1,
        "train_seconds": train_seconds if train_seconds is not None else "",
        "inference_seconds": inference_seconds if inference_seconds is not None else "",
    })

    for crop, acc in sorted(by_crop.items()):
        if acc["samples"] == 0:
            continue
        cp2, cr2, cf2 = precision_recall_f1(acc["change_tp"], acc["change_fp"], acc["change_fn"])
        rows.append({
            "scope": "crop", "model": model_name, "crop_type": crop,
            "samples": acc["samples"], "miou": "",
            "forest_precision": "", "forest_recall": "", "forest_f1": "",
            "change_precision": cp2, "change_recall": cr2, "change_f1": cf2,
            "train_seconds": "", "inference_seconds": "",
        })

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    file_exists = os.path.exists(output_csv)
    fieldnames = ["scope", "model", "crop_type", "samples", "miou",
                  "forest_precision", "forest_recall", "forest_f1",
                  "change_precision", "change_recall", "change_f1",
                  "train_seconds", "inference_seconds"]
    with open(output_csv, "a" if file_exists else "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Change detection benchmark saved to {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark utilities for segmentation outputs.")
    parser.add_argument("--mode", choices=["compare", "baseline", "change"], default="baseline",
                        help="compare: deeplab vs tessera; baseline: two-stage model; change: direct change map")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model-type", choices=ALL_MODEL_TYPES, default="deeplab",
                        help="Model architecture being evaluated")

    # compare mode
    parser.add_argument("--baseline-2020-dir", default=None)
    parser.add_argument("--masks-2024-dir", default=None)
    parser.add_argument("--deeplab-pred-dir", default=None)
    parser.add_argument("--tessera-pred-dir", default=None)
    parser.add_argument("--deeplab-elapsed-seconds", type=float, default=None)
    parser.add_argument("--tessera-elapsed-seconds", type=float, default=None)

    # baseline mode (two-stage models)
    parser.add_argument("--prediction-dir", default=None)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--farms-csv", default=None)
    parser.add_argument("--model-name", default=None, help="Model name label in output CSV (defaults to --model-type)")
    parser.add_argument("--train-seconds", type=float, default=None)
    parser.add_argument("--inference-seconds", type=float, default=None)
    parser.add_argument("--split-manifest-path", default=None)
    parser.add_argument("--split-name", default="val")

    # change mode (direct change map models: siamese, embed-change)
    parser.add_argument("--change-pred-dir", default=None, help="Directory with binary change map predictions")
    parser.add_argument("--mask-t1-dir", default=None, help="2020 hybrid mask directory")
    parser.add_argument("--mask-t2-dir", default=None, help="2024 hybrid mask directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_label = args.model_name or args.model_type

    if args.mode == "compare":
        missing = [n for n, v in [
            ("--baseline-2020-dir", args.baseline_2020_dir),
            ("--masks-2024-dir", args.masks_2024_dir),
            ("--deeplab-pred-dir", args.deeplab_pred_dir),
            ("--tessera-pred-dir", args.tessera_pred_dir),
        ] if not v]
        if missing:
            raise ValueError(f"Missing required arguments for compare mode: {', '.join(missing)}")
        run_benchmark(
            baseline_2020_dir=args.baseline_2020_dir,
            masks_2024_dir=args.masks_2024_dir,
            deeplab_pred_dir=args.deeplab_pred_dir,
            tessera_pred_dir=args.tessera_pred_dir,
            output_csv=args.output_csv,
            deeplab_elapsed_seconds=args.deeplab_elapsed_seconds,
            tessera_elapsed_seconds=args.tessera_elapsed_seconds,
        )
        return

    if args.mode == "change" or args.model_type in CHANGE_DETECTION_MODELS:
        missing = [n for n, v in [
            ("--change-pred-dir", args.change_pred_dir),
            ("--mask-t1-dir", args.mask_t1_dir),
            ("--mask-t2-dir", args.mask_t2_dir),
            ("--farms-csv", args.farms_csv),
        ] if not v]
        if missing:
            raise ValueError(f"Missing required arguments for change mode: {', '.join(missing)}")
        run_change_detection_benchmark(
            change_pred_dir=args.change_pred_dir,
            mask_t1_dir=args.mask_t1_dir,
            mask_t2_dir=args.mask_t2_dir,
            farms_csv=args.farms_csv,
            output_csv=args.output_csv,
            model_name=model_label,
            train_seconds=args.train_seconds,
            inference_seconds=args.inference_seconds,
        )
        return

    # baseline mode — all two-stage models (deeplab, tessera, tessera-embed, hybrid)
    missing = [n for n, v in [
        ("--prediction-dir", args.prediction_dir),
        ("--mask-dir", args.mask_dir),
        ("--farms-csv", args.farms_csv),
    ] if not v]
    if missing:
        raise ValueError(f"Missing required arguments for baseline mode: {', '.join(missing)}")
    run_baseline_metrics(
        prediction_dir=args.prediction_dir,
        mask_dir=args.mask_dir,
        farms_csv=args.farms_csv,
        output_csv=args.output_csv,
        model_name=model_label,
        train_seconds=args.train_seconds,
        inference_seconds=args.inference_seconds,
        split_manifest_path=args.split_manifest_path,
        split_name=args.split_name,
    )


if __name__ == "__main__":
    main()
