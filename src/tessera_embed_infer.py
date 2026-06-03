import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

import numpy as np
import rasterio
import torch
import torch.nn as nn
from affine import Affine

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from train_utils import extract_farm_key, load_embedding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_reference_profile(reference_dir: Optional[str], farm_key: str, year: str) -> Optional[dict]:
    if not reference_dir:
        return None

    patterns = [
        os.path.join(reference_dir, f"{farm_key}_{year}_*.tiff"),
        os.path.join(reference_dir, f"{farm_key}_{year}_*.tif"),
    ]
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            with rasterio.open(files[0]) as src:
                return src.profile
    return None


def find_tile_reference_profile(reference_dir: Optional[str], stem: str) -> Optional[dict]:
    if not reference_dir:
        return None

    patterns = [
        os.path.join(reference_dir, f"{stem}.tif"),
        os.path.join(reference_dir, f"{stem}.tiff"),
    ]
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            with rasterio.open(files[0]) as src:
                return src.profile
    return None


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


def load_model(checkpoint_path: str) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    model = TesseraEmbeddingSegHead(
        in_channels=int(config.get("in_channels", 128)),
        num_classes=int(config.get("num_classes", 4)),
        hidden_channels=int(config.get("hidden_channels", 128)),
    )

    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def write_prediction(output_path: str, prediction: np.ndarray, profile: Optional[dict]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if profile is None:
        h, w = prediction.shape
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 1,
            "dtype": rasterio.uint8,
            "crs": None,
            "transform": Affine.identity(),
            "compress": "lzw",
        }
    else:
        profile = profile.copy()
        profile.update({"count": 1, "dtype": rasterio.uint8, "compress": "lzw"})

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prediction.astype(rasterio.uint8), 1)


def run_inference(
    model_path: str,
    embeddings_dir: str,
    output_dir: str,
    year: str = "2024",
    reference_image_dir: Optional[str] = None,
) -> None:
    model = load_model(model_path)

    emb_root = Path(embeddings_dir)
    emb_files = sorted(
        [
            str(p)
            for p in emb_root.rglob("*.npy")
            if p.stem.startswith("grid_") and not p.stem.endswith("_scales")
        ]
    )
    if not emb_files:
        emb_files = sorted(glob.glob(os.path.join(embeddings_dir, "*.npy")))

    if not emb_files:
        logger.warning("No embedding files found in %s", embeddings_dir)
        return

    total = len(emb_files)
    processed = 0
    for idx, emb_path in enumerate(emb_files, 1):
        base_name = os.path.splitext(os.path.basename(emb_path))[0]
        farm_key = extract_farm_key(base_name)
        if not farm_key:
            if not base_name.startswith("grid_"):
                logger.debug("[%d/%d] Skipping (unrecognized key): %s", idx, total, base_name)
                continue

        try:
            scales_path = os.path.join(os.path.dirname(emb_path), f"{base_name}_scales.npy")
            if not os.path.exists(scales_path):
                scales_path = None

            embedding = load_embedding(emb_path, scales_path=scales_path)
            tensor = torch.from_numpy(embedding).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(tensor)["out"]
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

            output_path = os.path.join(output_dir, f"{base_name}_predicted.tif")
            profile = find_tile_reference_profile(reference_image_dir, base_name)
            if profile is None and farm_key:
                profile = find_reference_profile(reference_image_dir, farm_key, year)
            write_prediction(output_path, pred, profile)

            processed += 1
            logger.info("[%d/%d] Saved: %s", idx, total, output_path)
        except Exception as e:
            logger.error("[%d/%d] Failed %s: %s", idx, total, base_name, e)

    logger.info("Done: processed %d/%d embeddings", processed, total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with TESSERA embedding segmentation head.")
    parser.add_argument("--model-path", required=True, help="Path to trained tessera-embed checkpoint")
    parser.add_argument("--embeddings-dir", required=True, help="Directory with embedding .npy files")
    parser.add_argument("--output-dir", required=True, help="Where predicted masks are written")
    parser.add_argument("--year", default="2024", help="Year in filename lookup for reference images")
    parser.add_argument(
        "--reference-image-dir",
        default=None,
        help="Optional directory with georeferenced raw images for profile transfer",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        model_path=args.model_path,
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        year=args.year,
        reference_image_dir=args.reference_image_dir,
    )
