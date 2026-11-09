"""Read-only access to the Kaggle-style data folder.

File naming (verified on the data, see docs/data_inventory.md):
  2020_baseline/{kind}_{osmid}_2020_{YYYY-MM-DD}.tiff
  2024_current/{kind}_{osmid}_2024_{YYYY-MM-DD}.tiff
  hansen_labels/{kind}_{osmid}_hansen_label.tif
  farms_osm.csv: farm_id = "osm_{kind}_{osmid}"
kind is "relation" or "way". A farm can have several image dates per year.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

IMG_RE = re.compile(r"^(?P<fid>(?:relation|way)_\d+)_(?P<year>20\d\d)_(?P<date>\d{4}-\d{2}-\d{2})\.tiff?$")
LAB_RE = re.compile(r"^(?P<fid>(?:relation|way)_\d+)_hansen_label\.tif$")


def farm_key(farm_id: str) -> str:
    """'osm_relation_123' -> 'relation_123'."""
    return farm_id[4:] if farm_id.startswith("osm_") else farm_id


def index_images(folder: Path) -> dict[str, list[tuple[str, Path]]]:
    """farm key -> sorted [(date, path)]; unparsable names are skipped."""
    out: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for p in sorted(Path(folder).iterdir()):
        m = IMG_RE.match(p.name)
        if m:
            out[m["fid"]].append((m["date"], p))
    return {k: sorted(v) for k, v in out.items()}


def index_labels(folder: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(Path(folder).iterdir()):
        m = LAB_RE.match(p.name)
        if m:
            out[m["fid"]] = p
    return out


def load_farms(root: Path) -> pd.DataFrame:
    df = pd.read_csv(Path(root) / "farms_osm.csv")
    df["key"] = df["farm_id"].map(farm_key)
    return df
