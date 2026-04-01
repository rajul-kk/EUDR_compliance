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
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import rasterio


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_farm_key(name: str) -> Optional[str]:
    match = re.match(r"^(relation|way)_\d+", name)
    if not match:
        return None
    return match.group(0)


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


def load_embedding(path: str, scales_path: Optional[str] = None) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D embedding array, got shape {arr.shape} for {path}")

    # GeoTessera arrays are commonly int8 and require per-tile scales.
    if scales_path is not None:
        scales = np.load(scales_path)
        if scales.ndim == 2:
            scales = np.expand_dims(scales, axis=-1)
        if arr.shape[:2] != scales.shape[:2]:
            raise ValueError(
                f"Embedding/scales spatial shape mismatch: {arr.shape[:2]} vs {scales.shape[:2]} "
                f"for {path} and {scales_path}"
            )
        arr = arr.astype(np.float32) * scales.astype(np.float32)

    # Accept either HWC (H, W, C) or CHW (C, H, W).
    if arr.shape[-1] == 128:
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.shape[0] == 128:
        pass
    else:
        raise ValueError(f"Cannot infer embedding channel axis for shape {arr.shape} ({path})")

    return arr.astype(np.float32)


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


@torch.no_grad()
def compute_miou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4, ignore_index: int = 255) -> float:
    preds = torch.argmax(logits, dim=1)
    valid = targets != ignore_index

    ious = []
    for c in range(num_classes):
        pred_mask = (preds == c) & valid
        true_mask = (targets == c) & valid

        inter = torch.logical_and(pred_mask, true_mask).sum().float()
        union = torch.logical_or(pred_mask, true_mask).sum().float()
        if union > 0:
            ious.append((inter / union).item())

    return float(np.mean(ious)) if ious else 0.0


def split_dataset(dataset: Dataset, val_ratio: float, seed: int):
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split too large for dataset size")

    gen = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=gen)


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
    print(f"Resolved dataset mode: {dataset.dataset_mode} | pairs: {len(dataset)}")
    train_set, val_set = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)

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

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} val_mIoU={avg_val_miou:.4f}"
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
                },
            )
            print(f"Saved best checkpoint: {args.output_model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
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

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
