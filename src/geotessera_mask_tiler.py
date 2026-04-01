import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def read_bounds(path: Path) -> Tuple[float, float, float, float]:
    with rasterio.open(path) as src:
        b = src.bounds
    return (b.left, b.bottom, b.right, b.top)


def intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def collect_tile_tiffs(tile_tiff_dir: Path) -> List[Path]:
    return sorted(tile_tiff_dir.glob("grid_*.tif*"))


def collect_source_masks(mask_dir: Path, year: str) -> List[Path]:
    return sorted(mask_dir.glob(f"*_{year}_hybrid.tif*"))


def build_tile_masks(tile_tiff_dir: Path, source_mask_dir: Path, out_dir: Path, year: str) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_paths = collect_tile_tiffs(tile_tiff_dir)
    source_masks = collect_source_masks(source_mask_dir, year)

    if not tile_paths:
        raise RuntimeError(f"No tile tiff files found in {tile_tiff_dir}")
    if not source_masks:
        raise RuntimeError(f"No source masks found for year={year} in {source_mask_dir}")

    source_bounds = {p: read_bounds(p) for p in source_masks}

    summary = {
        "year": year,
        "tile_count": len(tile_paths),
        "source_mask_count": len(source_masks),
        "outputs": [],
    }

    for tile_path in tile_paths:
        stem = tile_path.stem

        with rasterio.open(tile_path) as tile_src:
            profile = tile_src.profile.copy()
            profile.update(count=1, dtype=rasterio.uint8, nodata=0, compress="lzw")
            tile_bounds = (tile_src.bounds.left, tile_src.bounds.bottom, tile_src.bounds.right, tile_src.bounds.top)

            merged = np.zeros((tile_src.height, tile_src.width), dtype=np.uint8)
            overlaps = 0

            for mask_path in source_masks:
                if not intersects(tile_bounds, source_bounds[mask_path]):
                    continue

                overlaps += 1
                with rasterio.open(mask_path) as mask_src:
                    src_arr = mask_src.read(1)
                    dst_arr = np.zeros((tile_src.height, tile_src.width), dtype=np.float32)

                    reproject(
                        source=src_arr,
                        destination=dst_arr,
                        src_transform=mask_src.transform,
                        src_crs=mask_src.crs,
                        dst_transform=tile_src.transform,
                        dst_crs=tile_src.crs,
                        resampling=Resampling.nearest,
                    )

                    dst_arr = np.nan_to_num(dst_arr, nan=0.0)
                    dst_arr = np.clip(dst_arr, 0, 255).astype(np.uint8)
                    merged = np.maximum(merged, dst_arr)

        out_path = out_dir / f"{stem}_mask.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(merged, 1)

        summary["outputs"].append(
            {
                "tile": stem,
                "path": str(out_path),
                "overlap_source_masks": overlaps,
                "non_zero_pixels": int((merged > 0).sum()),
                "height": int(merged.shape[0]),
                "width": int(merged.shape[1]),
            }
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GeoTessera-aligned mask tiles from source farm masks.")
    parser.add_argument("--tile-tiff-dir", required=True, help="Directory containing grid_*.tif tile files")
    parser.add_argument("--source-mask-dir", required=True, help="Directory containing *_<year>_hybrid.tif masks")
    parser.add_argument("--out-dir", required=True, help="Output directory for grid_*_mask.tif files")
    parser.add_argument("--year", default="2024", help="Mask year to use, e.g. 2024")
    parser.add_argument("--summary-json", default="", help="Optional path to write summary JSON")
    args = parser.parse_args()

    summary = build_tile_masks(
        tile_tiff_dir=Path(args.tile_tiff_dir),
        source_mask_dir=Path(args.source_mask_dir),
        out_dir=Path(args.out_dir),
        year=str(args.year),
    )

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary: {summary_path}")

    print(
        f"Generated {len(summary['outputs'])} tile masks from {summary['source_mask_count']} source masks "
        f"for year {summary['year']}"
    )


if __name__ == "__main__":
    main()
