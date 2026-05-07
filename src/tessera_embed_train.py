"""
TESSERA Lightweight Segmentation Head Training

This module trains a lightweight segmentation head (2-4 Conv2d layers) on top of
precomputed 128-dimensional TESSERA embeddings for deforestation/segmentation tasks.

=== EMBEDDING SOURCE ===

PRECOMPUTED GEOTESSERA EMBEDDINGS (RECOMMENDED)
    Description:
        - Uses embeddings already extracted by the TESSERA team (via GeoTessera library)
        - Global coverage at 10m resolution for 2024 (progressively extending backwards)
        - Avoids expensive local embedding generation
    
    Setup:
        1. Install GeoTessera: pip install geotessera
        2. Download embeddings for your region:
           
           python src/tessera_embedding_generation.py \
               --min-lat 51.4 --max-lat 51.6 \
               --min-lon -0.2 --max-lon 0.0 \
               --year 2024 --output-dir data/embeddings

        3. Build tile-aligned masks from source hybrid masks:

           python src/geotessera_mask_tiler.py \
               --tile-tiff-dir data/embeddings/global_0.1_degree_tiff_all \
               --source-mask-dir data/hybrid_masks \
               --out-dir data/geotessera_tile_masks \
               --year 2024
        
        4. Train head on cached embeddings:
           
           python src/tessera_embed_train.py \
               --embeddings-dir data/embeddings/global_0.1_degree_representation \
               --mask-dir data/geotessera_tile_masks \
               --dataset-mode geotessera \
               --year 2024 \
               --learning-rate 0.001

=== EMBEDDING CACHE STRUCTURE ===

Expected cache structure:

    data/embeddings/
    ├── embeddings/
    │   ├── tile_001.npy      # shape: (H, W, 128) or (128, H, W)
    │   ├── tile_002.npy
    │   └── ...
    └── metadata.json          # Maps tile_id → filepath, shape, dtype

This module expects embeddings in {embeddings_dir}/embeddings/*.npy

"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import rasterio

from train_utils import compute_miou, extract_farm_key, load_embedding, seed_everything, split_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_year(name: str) -> Optional[str]:
    match = re.search(r"_(20\d{2})(?:_|\.)", name)
    if not match:
        return None
    return match.group(1)


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


class TesseraEmbeddingDataset(Dataset):
    """
    Load precomputed TESSERA embeddings + corresponding segmentation masks.
    
    This dataset assumes embeddings have already been extracted and cached to disk
    (either via GeoTessera download or tessera_embedding_generation.py).
    
    Expected directory structure:
        embeddings_dir/
        ├── tile_001.npy  # shape: (128, H, W) or (H, W, 128)
        ├── tile_002.npy
        └── ...
        
        mask_dir/
        ├── relation_12345_2020_hybrid.tif
        ├── way_67890_2020_hybrid.tif
        └── ...
    
    Pairs embeddings to masks by extracting farm_key and year from filename using regex.
    
    Args:
        embeddings_dir (str): Directory containing .npy embedding files
        mask_dir (str): Directory containing .tif segmentation masks
        year (str): Filter by year (e.g., "2020")
    
    Raises:
        RuntimeError: If no embedding-mask pairs found after filtering
    """
    
    def __init__(self, embeddings_dir: str, mask_dir: str, year: str = "2020", dataset_mode: str = "auto"):
        self.pairs: List[Tuple[str, Optional[str], str]] = []
        self.missing_scales_count = 0

        if dataset_mode not in {"auto", "legacy", "geotessera"}:
            raise ValueError("dataset_mode must be one of: auto, legacy, geotessera")

        emb_root = Path(embeddings_dir)
        all_npy_paths = sorted(emb_root.rglob("*.npy"))
        geotessera_candidates = [p for p in all_npy_paths if p.stem.startswith("grid_") and not p.stem.endswith("_scales")]

        resolved_mode = dataset_mode
        if dataset_mode == "auto":
            resolved_mode = "geotessera" if geotessera_candidates else "legacy"

        if resolved_mode == "legacy":
            emb_files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith(".npy")])
            for emb_file in emb_files:
                farm_key = extract_farm_key(emb_file)
                if not farm_key:
                    continue

                emb_year = extract_year(emb_file)
                if emb_year is not None and emb_year != year:
                    continue

                mask_path = find_mask(mask_dir, farm_key, year)
                if mask_path is None:
                    continue

                self.pairs.append((os.path.join(embeddings_dir, emb_file), None, mask_path))
        else:
            for emb_path in geotessera_candidates:
                stem = emb_path.stem
                scales_path = emb_path.with_name(f"{stem}_scales.npy")
                mask_path = find_tile_mask(mask_dir, stem)
                if mask_path is None:
                    continue
                if not scales_path.exists():
                    self.missing_scales_count += 1
                self.pairs.append((str(emb_path), str(scales_path) if scales_path.exists() else None, mask_path))

        if not self.pairs:
            raise RuntimeError(
                f"No embedding-mask pairs found in embeddings_dir={embeddings_dir}, "
                f"mask_dir={mask_dir}, year={year}, dataset_mode={resolved_mode}."
            )

        self.dataset_mode = resolved_mode

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        emb_path, scales_path, mask_path = self.pairs[idx]

        embedding = load_embedding(emb_path, scales_path=scales_path)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)

        h = min(embedding.shape[1], mask.shape[0])
        w = min(embedding.shape[2], mask.shape[1])

        embedding = embedding[:, :h, :w]
        mask = mask[:h, :w]

        return torch.from_numpy(embedding), torch.from_numpy(mask)


class TesseraEmbeddingSegHead(nn.Module):
    def __init__(self, in_channels: int = 128, num_classes: int = 4, hidden_channels: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"out": self.head(x)}


def pair_key_from_embedding_path(emb_path: str) -> str:
    return Path(emb_path).stem


def run_preflight_checks(pairs: List[Tuple[str, Optional[str], str]], expected_in_channels: int) -> None:
    if not pairs:
        raise RuntimeError("No pairs available for preflight checks")

    orphan_scales = 0
    for emb_path, scales_path, mask_path in pairs:
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing mask path in pair: {mask_path}")
        if scales_path is not None and not os.path.exists(scales_path):
            orphan_scales += 1

    # Validate channels and shape alignment on one deterministic sample.
    emb_path, scales_path, mask_path = pairs[0]
    embedding = load_embedding(emb_path, scales_path=scales_path)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    if embedding.shape[0] != expected_in_channels:
        raise ValueError(
            f"Embedding channels mismatch: got {embedding.shape[0]}, expected {expected_in_channels}"
        )

    if embedding.shape[1] == 0 or embedding.shape[2] == 0:
        raise ValueError("Embedding has empty spatial dimensions")
    if mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError("Mask has empty spatial dimensions")

    if orphan_scales > 0:
        logger.warning("[preflight] %d pairs reference missing scales files", orphan_scales)

    logger.info(
        "[preflight] OK | pairs=%d sample_key=%s embedding_shape=%s mask_shape=%s",
        len(pairs), pair_key_from_embedding_path(emb_path), embedding.shape, mask.shape,
    )


def run_dataset_wide_shape_audit(
    pairs: List[Tuple[str, Optional[str], str]],
    expected_in_channels: int,
) -> None:
    channel_errors = 0
    empty_dim_errors = 0
    read_errors = 0

    for emb_path, scales_path, mask_path in pairs:
        try:
            embedding = load_embedding(emb_path, scales_path=scales_path)
            with rasterio.open(mask_path) as src:
                mask = src.read(1)

            if embedding.shape[0] != expected_in_channels:
                channel_errors += 1
            if embedding.shape[1] == 0 or embedding.shape[2] == 0 or mask.shape[0] == 0 or mask.shape[1] == 0:
                empty_dim_errors += 1
        except Exception:
            read_errors += 1

    if channel_errors > 0 or empty_dim_errors > 0 or read_errors > 0:
        raise RuntimeError(
            "Dataset-wide shape audit failed: "
            f"channel_errors={channel_errors}, empty_dim_errors={empty_dim_errors}, read_errors={read_errors}"
        )

    logger.info("[preflight] Dataset-wide shape audit passed for %d pairs", len(pairs))


def compute_subset_class_distribution(
    dataset: "TesseraEmbeddingDataset",
    subset,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    counts = np.zeros((num_classes,), dtype=np.float64)

    for idx in subset.indices:
        _, _, mask_path = dataset.pairs[idx]
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)

        valid = mask != ignore_index
        mask_valid = mask[valid]
        if mask_valid.size == 0:
            continue

        binc = np.bincount(mask_valid, minlength=num_classes)
        counts += binc[:num_classes]

    total = counts.sum()
    if total == 0:
        raise RuntimeError("Class distribution check failed: no valid labeled pixels found in subset")
    return counts / total


def run_split_distribution_check(
    dataset: "TesseraEmbeddingDataset",
    train_set,
    val_set,
    num_classes: int,
    max_class_ratio_delta: float,
    ignore_index: int = 255,
) -> None:
    train_dist = compute_subset_class_distribution(dataset, train_set, num_classes, ignore_index=ignore_index)
    val_dist = compute_subset_class_distribution(dataset, val_set, num_classes, ignore_index=ignore_index)
    deltas = np.abs(train_dist - val_dist)
    max_delta = float(np.max(deltas))

    logger.info(
        "[preflight] Class distribution delta (train vs val): %s",
        ", ".join([f"c{i}={d:.4f}" for i, d in enumerate(deltas)]),
    )

    if max_delta > max_class_ratio_delta:
        raise RuntimeError(
            f"Class distribution check failed: max_delta={max_delta:.4f} exceeds threshold={max_class_ratio_delta:.4f}"
        )


def write_split_manifest(
    output_path: str,
    dataset: "TesseraEmbeddingDataset",
    train_set,
    val_set,
    seed: int,
    val_ratio: float,
    dataset_mode: str,
) -> None:
    if not output_path:
        return

    train_keys = [pair_key_from_embedding_path(dataset.pairs[i][0]) for i in train_set.indices]
    val_keys = [pair_key_from_embedding_path(dataset.pairs[i][0]) for i in val_set.indices]

    payload = {
        "dataset_mode": dataset_mode,
        "seed": seed,
        "val_ratio": val_ratio,
        "counts": {
            "train": len(train_keys),
            "val": len(val_keys),
        },
        "splits": {
            "train": train_keys,
            "val": val_keys,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote split manifest: %s", output_path)


def save_checkpoint(model: nn.Module, output_path: str, config: Dict[str, object], extra: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(
        {
            "model_type": "tessera-embed",
            "config": config,
            "state_dict": model.state_dict(),
            "extra": extra,
        },
        output_path,
    )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    dataset = TesseraEmbeddingDataset(
        args.embeddings_dir,
        args.mask_dir,
        year=args.year,
        dataset_mode=args.dataset_mode,
    )
    logger.info("Resolved dataset mode: %s | pairs: %d", dataset.dataset_mode, len(dataset))
    if not args.skip_preflight:
        if dataset.missing_scales_count > args.max_missing_scales:
            raise RuntimeError(
                f"Missing scales threshold exceeded: {dataset.missing_scales_count} > {args.max_missing_scales}"
            )
        run_preflight_checks(dataset.pairs, expected_in_channels=args.in_channels)
        run_dataset_wide_shape_audit(dataset.pairs, expected_in_channels=args.in_channels)

    train_set, val_set = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)
    if not args.skip_preflight:
        run_split_distribution_check(
            dataset,
            train_set,
            val_set,
            num_classes=args.num_classes,
            max_class_ratio_delta=args.max_class_ratio_delta,
            ignore_index=255,
        )

    write_split_manifest(
        output_path=args.split_manifest_path,
        dataset=dataset,
        train_set=train_set,
        val_set=val_set,
        seed=args.seed,
        val_ratio=args.val_ratio,
        dataset_mode=dataset.dataset_mode,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TesseraEmbeddingSegHead(in_channels=args.in_channels, num_classes=args.num_classes, hidden_channels=args.hidden_channels).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_miou = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    config = {
        "in_channels": args.in_channels,
        "num_classes": args.num_classes,
        "hidden_channels": args.hidden_channels,
    }

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for embeddings, masks in train_loader:
            embeddings = embeddings.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(embeddings)["out"]
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_miou_sum = 0.0
        with torch.no_grad():
            for embeddings, masks in val_loader:
                embeddings = embeddings.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = model(embeddings)["out"]
                loss = criterion(logits, masks)

                val_loss += loss.item()
                val_miou_sum += compute_miou(logits, masks, num_classes=args.num_classes)

        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_val_miou = val_miou_sum / max(1, len(val_loader))

        logger.info(
            "Epoch [%d/%d] train_loss=%.4f val_loss=%.4f val_mIoU=%.4f",
            epoch + 1, args.epochs, avg_train_loss, avg_val_loss, avg_val_miou,
        )

        if avg_val_miou > best_miou:
            best_miou = avg_val_miou
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            save_checkpoint(
                model,
                args.output_model_path,
                config=config,
                extra={
                    "best_epoch": best_epoch,
                    "best_val_miou": best_miou,
                    "year": args.year,
                    "val_ratio": args.val_ratio,
                    "seed": args.seed,
                    "dataset_mode": dataset.dataset_mode,
                },
            )
            logger.info("Saved best checkpoint: %s", args.output_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight segmentation head on precomputed TESSERA embeddings.")
    parser.add_argument("--embeddings-dir", required=True, help="Directory containing embedding .npy files")
    parser.add_argument("--mask-dir", required=True, help="Directory containing yearly hybrid masks")
    parser.add_argument("--output-model-path", required=True, help="Output path for trained head checkpoint")

    parser.add_argument("--year", default="2020", help="Year of embeddings/masks to use for training")
    parser.add_argument(
        "--dataset-mode",
        choices=["auto", "legacy", "geotessera"],
        default="auto",
        help="Dataset pairing mode. auto detects geotessera tile layout.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--in-channels", type=int, default=128)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument(
        "--split-manifest-path",
        default="",
        help="Optional JSON path to store deterministic train/val split keys",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip dataset preflight checks",
    )
    parser.add_argument(
        "--max-missing-scales",
        type=int,
        default=0,
        help="Maximum allowed missing *_scales.npy files in geotessera mode before failing",
    )
    parser.add_argument(
        "--max-class-ratio-delta",
        type=float,
        default=0.10,
        help="Maximum allowed absolute train-vs-val class distribution delta",
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
