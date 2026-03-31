"""
TESSERA precomputed embedding downloader.

This script keeps only the low-cost pathway:
- Download precomputed TESSERA embeddings from GeoTessera.

It does not support local embedding generation from raw Sentinel imagery.
"""

import argparse
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_geotessera_embeddings(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    year: int,
    output_dir: str,
):
    """Download precomputed TESSERA embeddings for a bounding box."""
    try:
        from geotessera import GeoTessera
    except ImportError as exc:
        raise RuntimeError("GeoTessera is required. Install with: pip install geotessera") from exc

    bounds = (min_lat, min_lon, max_lat, max_lon)
    logger.info("Downloading precomputed embeddings for bounds=%s year=%s", bounds, year)

    geo = GeoTessera()
    geo.download(bounds=bounds, year=year, output_dir=output_dir)

    logger.info("Download complete. Embeddings saved under %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download precomputed TESSERA embeddings from GeoTessera."
    )
    parser.add_argument("--min-lat", type=float, required=True, help="Minimum latitude")
    parser.add_argument("--min-lon", type=float, required=True, help="Minimum longitude")
    parser.add_argument("--max-lat", type=float, required=True, help="Maximum latitude")
    parser.add_argument("--max-lon", type=float, required=True, help="Maximum longitude")
    parser.add_argument("--year", type=int, default=2024, help="Coverage year")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/embeddings",
        help="Output directory for downloaded embeddings",
    )

    args = parser.parse_args()

    download_geotessera_embeddings(
        min_lat=args.min_lat,
        min_lon=args.min_lon,
        max_lat=args.max_lat,
        max_lon=args.max_lon,
        year=args.year,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
