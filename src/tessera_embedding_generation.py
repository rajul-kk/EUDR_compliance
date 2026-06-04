"""
TESSERA precomputed embedding downloader.

Downloads precomputed TESSERA embeddings via the GeoTessera library.
Bounding box can be supplied explicitly or derived automatically from a farms CSV.
"""

import argparse
import logging
import os
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def bbox_from_farms_csv(csv_path: str, padding: float = 0.5) -> Tuple[float, float, float, float]:
    """
    Compute a bounding box from the lat/lon columns in farms_osm.csv.

    Args:
        csv_path: Path to CSV with 'lat' and 'lon' columns.
        padding:  Degrees of extra margin added on each side (default 0.5°).

    Returns:
        (min_lat, min_lon, max_lat, max_lon)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Farms CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in ("lat", "lon") if c not in df.columns]
    if missing:
        raise ValueError(f"Farms CSV is missing columns: {missing}")

    df = df.dropna(subset=["lat", "lon"])
    if df.empty:
        raise ValueError("Farms CSV has no rows with valid lat/lon after dropping NaN")

    min_lat = float(df["lat"].min()) - padding
    max_lat = float(df["lat"].max()) + padding
    min_lon = float(df["lon"].min()) - padding
    max_lon = float(df["lon"].max()) + padding

    # Clamp to valid geographic range
    min_lat = max(min_lat, -90.0)
    max_lat = min(max_lat, 90.0)
    min_lon = max(min_lon, -180.0)
    max_lon = min(max_lon, 180.0)

    logger.info(
        "Derived bbox from %d farms (padding=%.1f°): lat=[%.4f, %.4f] lon=[%.4f, %.4f]",
        len(df), padding, min_lat, max_lat, min_lon, max_lon,
    )
    return min_lat, min_lon, max_lat, max_lon


def download_geotessera_embeddings(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    year: int,
    output_dir: str,
    grid_resolution: Optional[float] = None,
) -> None:
    """Download precomputed TESSERA embeddings for a bounding box.

    Args:
        grid_resolution: Grid cell size in degrees. Default (None) uses the GeoTessera
            library default (~0.1° ≈ 11 km). Pass 0.01 for ~1.1 km resolution if the
            library supports it — check geotessera API for availability.
    """
    try:
        from geotessera import GeoTessera
    except ImportError as exc:
        raise RuntimeError("GeoTessera is required. Install with: pip install geotessera") from exc

    bounds = (min_lat, min_lon, max_lat, max_lon)
    logger.info("Downloading precomputed embeddings for bounds=%s year=%s resolution=%s", bounds, year, grid_resolution)

    geo = GeoTessera()

    # Pass resolution kwarg only if the installed version supports it; fall back gracefully.
    import inspect
    download_sig = inspect.signature(geo.download)
    if grid_resolution is not None and "resolution" in download_sig.parameters:
        geo.download(bounds=bounds, year=year, output_dir=output_dir, resolution=grid_resolution)
    elif grid_resolution is not None:
        logger.warning(
            "Installed geotessera version does not support resolution= kwarg; "
            "downloading at default resolution. Upgrade with: pip install --upgrade geotessera"
        )
        geo.download(bounds=bounds, year=year, output_dir=output_dir)
    else:
        geo.download(bounds=bounds, year=year, output_dir=output_dir)

    logger.info("Download complete. Embeddings saved under %s", output_dir)


def resolve_bbox(
    farms_csv: Optional[str],
    min_lat: Optional[float],
    min_lon: Optional[float],
    max_lat: Optional[float],
    max_lon: Optional[float],
    padding: float,
) -> Tuple[float, float, float, float]:
    """Return (min_lat, min_lon, max_lat, max_lon) from explicit args or farms CSV."""
    explicit = [x for x in (min_lat, min_lon, max_lat, max_lon) if x is not None]

    if len(explicit) == 4:
        return min_lat, min_lon, max_lat, max_lon  # type: ignore[return-value]

    if len(explicit) > 0:
        raise ValueError("Provide all four of --min-lat/--min-lon/--max-lat/--max-lon, or none (use --farms-csv).")

    if not farms_csv:
        raise ValueError("Either provide explicit bbox arguments or --farms-csv to auto-derive the bounding box.")

    return bbox_from_farms_csv(farms_csv, padding=padding)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download precomputed TESSERA embeddings from GeoTessera.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # auto-derive bbox from your farm locations:\n"
            "  python src/tessera_embedding_generation.py --farms-csv inputs/farms_osm.csv --year 2024\n\n"
            "  # explicit bbox:\n"
            "  python src/tessera_embedding_generation.py \\\n"
            "      --min-lat -10 --max-lat 5 --min-lon -80 --max-lon -50 --year 2024"
        ),
    )

    bbox_group = parser.add_argument_group("bounding box (explicit or auto-derived from farms CSV)")
    bbox_group.add_argument("--min-lat", type=float, default=None, help="Minimum latitude")
    bbox_group.add_argument("--min-lon", type=float, default=None, help="Minimum longitude")
    bbox_group.add_argument("--max-lat", type=float, default=None, help="Maximum latitude")
    bbox_group.add_argument("--max-lon", type=float, default=None, help="Maximum longitude")
    bbox_group.add_argument(
        "--farms-csv",
        type=str,
        default=None,
        help="Path to farms CSV with lat/lon columns; bbox is derived automatically when explicit args are omitted",
    )
    bbox_group.add_argument(
        "--bbox-padding",
        type=float,
        default=0.5,
        help="Degrees of margin added around the farm bbox (default: 0.5)",
    )

    parser.add_argument("--year", type=int, default=2024, help="Coverage year")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/embeddings",
        help="Output directory for downloaded embeddings",
    )
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=None,
        help=(
            "Grid cell size in degrees (e.g. 0.01 ≈ 1.1 km). "
            "Default uses the GeoTessera library default (~0.1° ≈ 11 km). "
            "Only effective if the installed geotessera version supports the resolution= kwarg."
        ),
    )

    args = parser.parse_args()

    min_lat, min_lon, max_lat, max_lon = resolve_bbox(
        farms_csv=args.farms_csv,
        min_lat=args.min_lat,
        min_lon=args.min_lon,
        max_lat=args.max_lat,
        max_lon=args.max_lon,
        padding=args.bbox_padding,
    )

    download_geotessera_embeddings(
        min_lat=min_lat,
        min_lon=min_lon,
        max_lat=max_lat,
        max_lon=max_lon,
        year=args.year,
        output_dir=args.output_dir,
        grid_resolution=args.grid_resolution,
    )


if __name__ == "__main__":
    main()
