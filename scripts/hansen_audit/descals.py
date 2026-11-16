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
    """({category: percentage share 0-100}, nearest planting year) on a north-up EPSG:4326 grid, from the tiles that
    intersect it (the tiles are EPSG:4326 too, so the bounds need no transform)."""
    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling, reproject

    a, _, c, _, e, f = transform
    west, north, east, south = c, f, c + a * shape[1], f + e * shape[0]
    acc = {k: np.zeros(shape, np.float32) for k in D.CATEGORIES}
    yop_nn = np.zeros(shape, np.uint16)
    dst_tr = Affine(*transform)
    for p, (w, s_, e_, n) in index:
        if w >= east or e_ <= west or s_ >= north or n <= south:
            continue
        with rasterio.open(p) as src:
            assert src.crs.to_string() == "EPSG:4326", (p, src.crs)
            win = rasterio.windows.from_bounds(west, south, east, north, transform=src.transform)
            win = win.round_offsets().round_lengths()
            yop = src.read(1, window=win, boundless=True, fill_value=0)
            src_tr = src.window_transform(win)
        for k, ind in D.indicators(yop).items():
            dst = np.zeros(shape, np.float32)
            reproject(ind, dst, src_transform=src_tr, src_crs="EPSG:4326", dst_transform=dst_tr, dst_crs="EPSG:4326",
                      resampling=Resampling.average)
            acc[k] += dst
        nn = np.zeros(shape, np.uint16)
        reproject(yop, nn, src_transform=src_tr, src_crs="EPSG:4326", dst_transform=dst_tr, dst_crs="EPSG:4326",
                  resampling=Resampling.nearest)
        yop_nn = np.maximum(yop_nn, nn)
    return {k: (np.clip(v, 0, 1) * 100).round().astype(np.uint8) for k, v in acc.items()}, yop_nn


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
        shares, yop = shares_on(index, tr, shape)
        np.savez_compressed(out, yop=yop, **shares)
        n += 1
        if n % 100 == 0:
            print(f"{n} grids, {time.time() - t0:.0f} s", flush=True)
    print(f"{n} grids written, {len(todo)} in all, {time.time() - t0:.0f} s", flush=True)


def load_grid(grids_dir, name):
    p = os.path.join(grids_dir, f"{name}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def masks(g, threshold=D.PALM_SHARE, count_1989=True):
    """(palm by 2020, palm 2021, value 2022) bool arrays from a stored grid."""
    return (D.palm_by_2020(g, threshold, count_1989), D.palm_share_at_least(g["y2021"], threshold),
            D.palm_share_at_least(g["y2022"], threshold))


def farm_row(lab, g, forest, threshold=D.PALM_SHARE, count_1989=True):
    p20, p21, p22 = masks(g, threshold, count_1989)
    row = D.farm_counts(lab, p20, p21, p22)
    loss = lab == 2
    row["class2_palm2020_gfc_nonforest"] = int((loss & p20 & ~forest).sum())
    row["class2_gfc_forest_not_palm2020"] = int((loss & ~p20 & forest).sum())
    return row


def summary_block(rows, keys):
    from hansen_audit import gfc

    col = {c: np.array([rows[k][c] for k in keys]) for c in rows[keys[0]]}
    hflag, gflag = D.gated_flags(col["class2"], col["class2_palm2020"])
    return {"n_farms": len(keys), "loss_cells": int(col["class2"].sum()),
            "share_loss_palm_by_2020": gfc.ratio_ci(col["class2_palm2020"], col["class2"]),
            "share_loss_palm_planted_2021": gfc.ratio_ci(col["class2_palm2021"], col["class2"]),
            "share_loss_value_2022": gfc.ratio_ci(col["class2_palm2022"], col["class2"]),
            "hansen_flagged": int(hflag.sum()), "gated_flagged": int(gflag.sum()),
            "share_flags_not_surviving": gfc.ratio_ci((hflag & ~gflag).astype(float), hflag.astype(float)),
            "palm2020_loss_cells": int(col["class2_palm2020"].sum()),
            "palm2020_loss_cells_gfc_nonforest": int(col["class2_palm2020_gfc_nonforest"].sum()),
            "gfc_forest_loss_cells_not_palm2020": int(col["class2_gfc_forest_not_palm2020"].sum())}


def summary(rows, group):
    """Analyses 1 and 2 over the farms in `rows`, pooled and by group."""
    keys = sorted(rows)
    if not keys:
        return None
    out = summary_block(rows, keys)
    out["by_group"] = {g: summary_block(rows, [k for k in keys if group[k] == g]) for g in sorted({group[k] for k in keys})}
    return out


def s2c_check(labels, grids_dir, boxes_pairs):
    """Analysis 5 on the E4/E8 windows: S2C 2020 class of loss cells in the farm boxes against palm by 2020."""
    from e4_farms import label_on_window

    from hansen_audit import farms_e4 as E
    from hansen_audit import gfc

    t = np.zeros((3, 2), int)   # S2C class x (not palm by 2020, palm by 2020)
    used, missing = 0, []
    for wj, pj in boxes_pairs:
        with open(wj) as fh:
            W = json.load(fh)
        tag = Path(wj).parent.name
        boxes = dict(zip(W["farms"], np.array(W["boxes"])))
        preds = np.load(pj)
        for w in W["windows"]:
            name = f"win_{w['id']}"
            g = load_grid(grids_dir, f"{tag}_{name}")
            if name not in preds.files or g is None:
                missing.append(f"{tag}_{name}")
                continue
            origin, shape = np.array(w["origin"]), tuple(w["shape"])
            p = preds[name]
            p20 = masks(g)[0]
            assert p20.shape == p.shape == shape, (tag, name)
            used += 1
            for k in w["farms"]:
                lab = label_on_window(labels[k], origin, shape)
                cells = E.farm_cells(origin, shape, boxes[k]) & (p >= 0) & (lab == 2)
                for c in range(3):
                    m = cells & (p == c)
                    t[c] += [int((m & ~p20).sum()), int((m & p20).sum())]
    plant_palm = float(t[1, 1] / t[1].sum()) if t[1].sum() else None
    forest_not = float(t[0, 0] / t[0].sum()) if t[0].sum() else None
    return {"windows_used": used, "windows_missing": missing,
            "loss_cells_s2c_by_palm2020": {n: {"not_palm": int(t[i, 0]), "palm_by_2020": int(t[i, 1])}
                                           for i, n in enumerate(("FOREST", "PLANT", "OTHER"))},
            "share_s2c_plant_palm_by_2020": plant_palm, "share_s2c_forest_not_palm": forest_not,
            "reading": gfc.s2c_reading(plant_palm, forest_not).replace("GFC2020", "Descals"),
            "note": "farm boxes overlap within a window, so a cell can count for more than one farm (as in E4)"}


HIST_FIRST, HIST_LAST = 1989, 2022


def yop_hist(data, group):
    """Analysis 6: planting year of the palm-by-2020 loss cells on ID Oil Palm farms (nearest value at the cell centre)."""
    counts = np.zeros(HIST_LAST - HIST_FIRST + 1, np.int64)
    for k, (lab, g, _, _) in data.items():
        if group[k] == "ID|Oil Palm":
            y = g["yop"][(lab == 2) & masks(g)[0]].astype(int)
            y = y[(y >= HIST_FIRST) & (y <= HIST_LAST)]
            counts += np.bincount(y - HIST_FIRST, minlength=len(counts))
    return {str(HIST_FIRST + i): int(v) for i, v in enumerate(counts) if v}


def read_farms(labels, grids_dir, gfc_dir, els):
    """{farm: (label, stored Descals grid, GFC forest, outline cells or None)}, and the farms missing a grid."""
    import rasterio

    from hansen_audit import gfc
    from hansen_audit import outlines as O

    data, missing = {}, []
    for k in sorted(labels):
        g = load_grid(grids_dir, f"farm_{k}")
        fp = os.path.join(gfc_dir, "V3", f"farm_{k}.npy")
        if g is None or not os.path.exists(fp):
            missing.append(k)
            continue
        with rasterio.open(labels[k]) as s:
            lab, tr = s.read(1), tuple(s.transform)[:6]
        frac = np.load(fp)
        assert g["y1989"].shape == lab.shape == frac.shape, k
        poly = O.element_polygon(els[k]) if k in els else None
        data[k] = (lab, g, gfc.gfc_forest(frac), O.outline_cells_on(tr, lab.shape, poly) if poly is not None else None)
    return data, missing


def farm_rows(data, threshold=D.PALM_SHARE, count_1989=True, outline=False):
    out = {}
    for k, (lab, g, forest, inside) in data.items():
        if outline:
            if inside is None or not inside.any():
                continue
            lab = np.where(inside, lab, 255)
        out[k] = farm_row(lab, g, forest, threshold, count_1989)
    return out


def ratio(num, den):
    return num / den if den else None


def run(a):
    from hansen_audit import outlines as O

    t0 = time.time()
    z = np.load(a.features, allow_pickle=True)
    group = {str(k): f"{c}|{p}" for k, c, p in zip(z["keys"], z["country"], z["crop"])}
    labels = index_labels(Path(a.labels_dir))
    els = {}
    if a.outlines:
        with open(os.path.join(a.outlines, "overpass.json"), encoding="utf-8") as fh:
            els = {O.farm_key(e): e for e in json.load(fh)["elements"]}
    data, missing = read_farms(labels, a.grids_dir, a.gfc_dir, els)
    primary = summary(farm_rows(data), group)
    idp = primary["by_group"]["ID|Oil Palm"]
    share3 = ratio(idp["palm2020_loss_cells_gfc_nonforest"], idp["palm2020_loss_cells"])
    R = {"protocol": "PROTOCOL_E13.md", "labels": Path(a.labels_dir).name,
         "reading_1989_2022": "1989 = planted 1989 or earlier (palm by 2020); 2022 own category (owner, 15 Nov)",
         "missing_farms": missing, "analyses_1_2": primary,
         "analysis_3_vs_gfc2020": {"id_oil_palm_share_palm2020_loss_gfc_nonforest": share3, "reading": D.map_reading(share3),
                                   "all_groups_share": ratio(primary["palm2020_loss_cells_gfc_nonforest"],
                                                             primary["palm2020_loss_cells"])},
         "analysis_4_outlines": summary(farm_rows(data, outline=True), group) if a.outlines else None,
         "analysis_5_s2c": s2c_check(labels, a.grids_dir, list(zip(a.boxes[::2], a.boxes[1::2]))) if a.boxes else None,
         "analysis_6_yop_hist_id_oil_palm": yop_hist(data, group),
         "analysis_7_sensitivity": {}}
    fields = ("share_loss_palm_by_2020", "hansen_flagged", "gated_flagged", "share_flags_not_surviving")
    for name, kw in {"share_0.3": {"threshold": 0.3}, "share_0.7": {"threshold": 0.7},
                     "1989_not_palm": {"count_1989": False}}.items():
        s = summary(farm_rows(data, **kw), group)
        R["analysis_7_sensitivity"][name] = {f: s[f] for f in fields} | {
            "id_oil_palm": {f: s["by_group"]["ID|Oil Palm"][f] for f in fields}}
    R["seconds"] = round(time.time() - t0, 1)
    os.makedirs(a.results_dir, exist_ok=True)
    with open(os.path.join(a.results_dir, "descals.json"), "w") as fh:
        json.dump(R, fh, indent=1)
    print(json.dumps({k: v for k, v in primary.items() if k != "by_group"}, indent=1))
    print(json.dumps({k: R[k] for k in ("analysis_3_vs_gfc2020", "analysis_5_s2c", "analysis_6_yop_hist_id_oil_palm",
                                        "analysis_7_sensitivity")}, indent=1))


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
    r = sub.add_parser("run")
    for n in ("labels_dir", "grids_dir", "gfc_dir", "features", "results_dir"):
        r.add_argument(n)
    r.add_argument("--outlines", help="E12 outlines dir (analysis 4)")
    r.add_argument("--boxes", nargs="+", default=[], help="pairs: windows.json farm_preds.npz (analysis 5)")
    a = ap.parse_args()
    if a.cmd == "run" and len(a.boxes) % 2:
        sys.exit("--boxes takes pairs: windows.json farm_preds.npz")
    {"download": download, "encoding": encoding, "grids": grids, "run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
