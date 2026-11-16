"""Read-only access to the Kaggle-style data folder.

File naming (verified on the data, see docs/data_inventory.md):
  2020_baseline/{kind}_{osmid}_2020_{YYYY-MM-DD}.tiff
  2024_current/{kind}_{osmid}_2024_{YYYY-MM-DD}.tiff
  hansen_labels/{kind}_{osmid}_hansen_label.tif
  2020_baseline/{kind}_{osmid}_2020.tiff, 2024_current/{kind}_{osmid}_2024.tiff: median composites of up to 3 scenes written by
    src/sentinel_client.py (no scene date; found 14 Nov, E6). Read only when asked (composites=True).
  farms_osm.csv: farm_id = "osm_{kind}_{osmid}"
kind is "relation" or "way". A farm can have several image dates per year.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

IMG_RE = re.compile(r"^(?P<fid>(?:relation|way)_\d+)_(?P<year>20\d\d)_(?P<date>\d{4}-\d{2}-\d{2})\.tiff?$")
COMP_RE = re.compile(r"^(?P<fid>(?:relation|way)_\d+)_(?P<year>20\d\d)\.tiff?$")
COMPOSITE = "composite"
LAB_RE = re.compile(r"^(?P<fid>(?:relation|way)_\d+)_hansen_label\.tif$")


def farm_key(farm_id: str) -> str:
    """'osm_relation_123' -> 'relation_123'."""
    return farm_id[4:] if farm_id.startswith("osm_") else farm_id


def index_images(folder: Path, composites: bool = False) -> dict[str, list[tuple[str, Path]]]:
    """farm key -> sorted [(date, path)]; unparsable names are skipped. With composites=True, undated composites are
    included with the date COMPOSITE, which sorts after every scene date, so a dated scene is still taken first."""
    out: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for p in sorted(Path(folder).iterdir()):
        m = IMG_RE.match(p.name)
        if m:
            out[m["fid"]].append((m["date"], p))
        elif composites and (c := COMP_RE.match(p.name)):
            out[c["fid"]].append((COMPOSITE, p))
    return {k: sorted(v) for k, v in out.items()}


def labels_root(data_root, labels_dir=None) -> Path:
    """The label rasters to read: `labels_dir` if given (the PROTOCOL_E10 rebuild, hansen_labels_v2), else
    <data_root>/hansen_labels, the files every result before 15 Nov 2026 used."""
    return Path(labels_dir) if labels_dir else Path(data_root) / "hansen_labels"


def read_json(path):
    """Load a JSON file as UTF-8 (Windows would otherwise use cp1252), closing it."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_csv(path) -> list[dict]:
    """All rows of a CSV file as dicts (UTF-8), closing it."""
    import csv

    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_json(obj, path, **kw):
    """Write `obj` as JSON in UTF-8, closing the file; keyword arguments go to json.dump."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, **kw)


def farm_groups(features) -> dict[str, str]:
    """{farm key: "country|crop"} from a loaded features.npz."""
    return {str(k): f"{c}|{p}" for k, c, p in zip(features["keys"], features["country"], features["crop"])}


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
