"""PROTOCOL_E13: Hansen loss against oil palm planting year (Descals et al. 2024, Zenodo 13379129, CC BY 4.0).

usage:
  python -I descals.py download <out_dir> [--cap-mb 304]
  python -I descals.py encoding <out_dir> <results_dir>
  python -I descals.py grids <out_dir> <labels_dir> <grids_dir> [--windows <windows.json> ...]
  python -I descals.py run <labels_dir> <grids_dir> <gfc_dir> <features.npz> <results_dir> [--outlines <dir>]
                       [--boxes <windows.json> <farm_preds.npz> ...]
download: the planting-year zip, the tile grid and the validation points, checked against Zenodo's MD5s, streamed under a cap
          on the job's own bytes (refused up front if the declared sizes pass it) and the machine-wide download lock.
encoding: reads every tile's header and values from inside the zip (nothing unpacked) and writes encoding.json; stops if
          the values are not planting years plus one non-palm value.
grids:    per label raster (and per E4/E8 window), the area-weighted share of each cell that is palm planted by 2020 and
          palm planted in 2021, from the tiles that intersect it; one .npz per grid.
run:      the PROTOCOL_E13 analyses; writes <results_dir>/descals.json.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from hansen_audit import descals as D  # noqa: E402
from hansen_audit.budget import OverBudget, capped_download, exclusive_download  # noqa: E402
from hansen_audit.io import index_labels  # noqa: E402

RECORD = "https://zenodo.org/api/records/13379129"
FILES = ("GlobalOilPalm_OP-YoP.zip", "Grid_OilPalm2016-2021.zip", "Validation_points_GlobalOP2016-2021.zip")
YOP_ZIP = FILES[0]
THRESHOLDS = (0.3, 0.5, 0.7)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(a):
    import requests

    os.makedirs(a.out_dir, exist_ok=True)
    rec = requests.get(RECORD, timeout=60).json()
    meta = {f["key"]: f for f in rec["files"] if f["key"] in FILES}
    missing = sorted(set(FILES) - set(meta))
    if missing:
        sys.exit(f"not in the record: {missing}")
    cap = int(a.cap_mb * 1e6)
    declared = sum(int(meta[k]["size"]) for k in FILES)
    if declared > cap:
        raise OverBudget(f"declared sizes {declared} pass the cap {cap}")
    man = {"record": RECORD, "version": rec["metadata"].get("version"), "licence": rec["metadata"].get("license"),
           "cap_bytes": cap, "declared_bytes": declared, "files": {}}
    used = 0
    with exclusive_download():
        for k in FILES:
            path = os.path.join(a.out_dir, k)
            want = meta[k]["checksum"].split(":", 1)[1]
            if os.path.exists(path) and md5(path) == want:
                man["files"][k] = {"bytes": os.path.getsize(path), "md5": want, "status": "present"}
                continue
            t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with requests.get(meta[k]["links"]["self"], stream=True, timeout=120,
                              headers={"User-Agent": "hansen-audit-research/1.0"}) as r:
                r.raise_for_status()
                n = capped_download(r, path, cap - used)
            used += n
            got = md5(path)
            man["files"][k] = {"bytes": n, "md5": got, "md5_ok": got == want, "fetched_utc": t}
            if got != want:
                os.remove(path)
                sys.exit(f"{k}: MD5 mismatch; file removed")
    man["bytes_received"] = used
    with open(os.path.join(a.out_dir, "download_manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    print(json.dumps(man, indent=1))


def tile_paths(out_dir):
    import zipfile

    z = os.path.join(out_dir, YOP_ZIP)
    with zipfile.ZipFile(z) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff")))
    return [f"/vsizip/{Path(z).as_posix()}/{n}" for n in names]


def tile_index(out_dir):
    """[(path, (west, south, east, north) in EPSG:4326)] for every planting-year tile, from the headers."""
    import rasterio
    from rasterio.warp import transform_bounds

    out = []
    for p in tile_paths(out_dir):
        with rasterio.open(p) as s:
            out.append((p, transform_bounds(s.crs, "EPSG:4326", *s.bounds)))
    return out


def encoding(a):
    import rasterio

    paths = tile_paths(a.out_dir)
    seen, info = set(), {}
    for p in paths:
        with rasterio.open(p) as s:
            key = (s.dtypes[0], s.nodata, s.crs.to_string(), round(s.res[0], 9), round(s.res[1], 9))
            info[str(key)] = info.get(str(key), 0) + 1
            seen |= set(np.unique(s.read(1)).tolist())
            nodata = s.nodata
    ok, why = D.encoding_ok(seen, nodata)
    R = {"tiles": len(paths), "dtype_nodata_crs_res": info, "distinct_values": sorted(seen)[:80], "n_distinct": len(seen),
         "encoding_ok": ok, "reading": why}
    os.makedirs(a.results_dir, exist_ok=True)
    with open(os.path.join(a.results_dir, "encoding.json"), "w") as fh:
        json.dump(R, fh, indent=1)
    print(json.dumps(R, indent=1))
    if not ok:
        sys.exit("encoding does not match PROTOCOL_E13; nothing run")


def shares_on(index, transform, shape):
    """(share palm by 2020, share palm 2021) on a north-up EPSG:4326 grid, from the intersecting tiles."""
    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling, reproject

    a, _, c, _, e, f = transform
    west, north, east, south = c, f, c + a * shape[1], f + e * shape[0]
    out20, out21 = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    for p, (w, s_, e_, n) in index:
        if w >= east or e_ <= west or s_ >= north or n <= south:
            continue
        with rasterio.open(p) as src:
            win = rasterio.windows.from_bounds(*rasterio.warp.transform_bounds("EPSG:4326", src.crs, west, south, east, north),
                                               transform=src.transform).round_offsets().round_lengths()
            yop = src.read(1, window=win, boundless=True, fill_value=0)
            src_tr = src.window_transform(win)
            for ind, acc in zip(D.indicators(yop), (out20, out21)):
                dst = np.zeros(shape, np.float32)
                reproject(ind, dst, src_transform=src_tr, src_crs=src.crs, dst_transform=Affine(*transform),
                          dst_crs="EPSG:4326", resampling=Resampling.average)
                acc += dst
    return np.clip(out20, 0, 1), np.clip(out21, 0, 1)


def grids(a):
    import rasterio

    from hansen_audit.s2 import HANSEN_RES

    index = tile_index(a.out_dir)
    os.makedirs(a.grids_dir, exist_ok=True)
    todo = {}
    for k, p in index_labels(Path(a.labels_dir)).items():
        with rasterio.open(p) as s:
            assert s.crs.to_string() == "EPSG:4326", (k, s.crs)
            todo[f"farm_{k}"] = (tuple(s.transform)[:6], s.shape)
    for wj in a.windows or []:
        with open(wj) as fh:
            W = json.load(fh)
        tag = Path(wj).parent.name
        for w in W["windows"]:
            todo[f"{tag}_win_{w['id']}"] = ((HANSEN_RES, 0.0, w["origin"][0], 0.0, -HANSEN_RES, w["origin"][1]),
                                            tuple(w["shape"]))
    t0, n = time.time(), 0
    for name, (tr, shape) in todo.items():
        out = os.path.join(a.grids_dir, f"{name}.npz")
        if os.path.exists(out):
            continue
        s20, s21 = shares_on(index, tr, shape)
        np.savez_compressed(out, palm2020=(s20 * 100).round().astype(np.uint8), palm2021=(s21 * 100).round().astype(np.uint8))
        n += 1
        if n % 100 == 0:
            print(f"{n} grids, {time.time() - t0:.0f} s", flush=True)
    print(f"{n} grids written, {len(todo)} in all", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("out_dir")
    d.add_argument("--cap-mb", type=float, default=304)
    e = sub.add_parser("encoding")
    for n in ("out_dir", "results_dir"):
        e.add_argument(n)
    g = sub.add_parser("grids")
    for n in ("out_dir", "labels_dir", "grids_dir"):
        g.add_argument(n)
    g.add_argument("--windows", action="append")
    a = ap.parse_args()
    {"download": download, "encoding": encoding, "grids": grids}[a.cmd](a)


if __name__ == "__main__":
    main()
