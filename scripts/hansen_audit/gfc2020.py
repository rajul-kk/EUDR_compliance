"""PROTOCOL_E9: Hansen loss against the JRC GFC2020 forest map.

usage:
  python -I gfc2020.py download <data_root> <env_file> <out_dir> <windows.json>... [--budget-gb 1]
  python -I gfc2020.py run <data_root> <features.npz> <gfc_dir> <results_dir> --boxes <windows.json> <farm_preds.npz> [...]
download: for GFC2020 V3 and V4, the per-cell forest percentage on every farm's Hansen label grid and on each window grid
          in the given windows.json files (E4, E8), computed on Earth Engine (area-weighted mean of Map == 1). One .npy per
          grid in <out_dir>/<version>/, resumable, under hansen_audit.budget.DownloadBudget. Credentials: GEE_PROJECT_ID
          from <env_file> (never printed).
run:      the PROTOCOL_E9 analyses; writes <results_dir>/gfc2020.json. --boxes pairs a windows.json with the farm_preds.npz
          written by `e4_farms.py run` for those windows (S2C check, analysis 4).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from hansen_audit import farms_e4 as E  # noqa: E402
from hansen_audit import gfc  # noqa: E402
from hansen_audit.budget import exclusive_download  # noqa: E402
from hansen_audit.io import index_labels  # noqa: E402

ASSETS = {"V3": "JRC/GFC2020/V3", "V4": "JRC/GFC2020/V4"}
PRIMARY = "V3"
THRESHOLDS = (30, 50, 70)
WORKERS = 8


def label_grids(data_root):
    """{farm key: (transform tuple a, b, c, d, e, f, shape)} for every Hansen label raster."""
    import rasterio

    out = {}
    for k, p in index_labels(Path(data_root) / "hansen_labels").items():
        with rasterio.open(p) as s:
            out[k] = (tuple(s.transform)[:6], s.shape)
    return out


def window_grids(windows_json):
    """{window id: (transform, shape)} on the Hansen grid used by e4_farms.py."""
    from hansen_audit.s1 import hansen_transform

    W = json.load(open(windows_json))
    tag = Path(windows_json).parent.name
    return {f"{tag}_win_{w['id']}": (tuple(hansen_transform(np.array(w["origin"])))[:6], tuple(w["shape"]))
            for w in W["windows"]}


def fetch_grid(asset, transform, shape):
    """uint8 (H, W) GFC forest percentage on the grid, computed server-side."""
    import ee

    a, b, c, d, e, f = transform
    img = (ee.Image(asset).select("Map").eq(1).unmask(0)
           .reduceResolution(ee.Reducer.mean(), maxPixels=1024).multiply(100).round().toUint8())
    arr = ee.data.computePixels({
        "expression": img, "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": int(shape[1]), "height": int(shape[0])},
                 "affineTransform": {"scaleX": a, "shearX": b, "translateX": c, "shearY": d, "scaleY": e, "translateY": f},
                 "crsCode": "EPSG:4326"}})
    return np.asarray(arr["Map"], np.uint8)


def download(a):
    import ee
    from dotenv import dotenv_values

    from hansen_audit.budget import DownloadBudget

    project = dotenv_values(a.env_file).get("GEE_PROJECT_ID")
    if not project:
        sys.exit("GEE_PROJECT_ID missing from the env file")
    ee.Initialize(project=project)
    grids = {f"farm_{k}": v for k, v in label_grids(a.data_root).items()}
    for w in a.windows:
        grids.update(window_grids(w))
    os.makedirs(a.out_dir, exist_ok=True)
    budget = DownloadBudget(a.out_dir, a.budget_gb, "grids", retry_failed=a.retry_failed)
    todo = [(v, g) for v in ASSETS for g in grids
            if not budget.done(f"{v}/{g}", os.path.join(a.out_dir, v, f"{g}.npy"))]
    print(f"{len(grids)} grids x {len(ASSETS)} versions, {len(todo)} to fetch", flush=True)

    def one(item):
        v, g = item
        out = os.path.join(a.out_dir, v, f"{g}.npy")
        t0 = time.time()
        try:
            arr = fetch_grid(ASSETS[v], *grids[g])
        except Exception as ex:  # noqa: BLE001 - recorded per item, retried with --retry-failed
            return item, {"error": f"{type(ex).__name__}: {str(ex).replace(project, '<project>')}"[:300]}, True
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out[:-4] + ".part.npy"
        np.save(tmp, arr)
        os.replace(tmp, out)
        # bytes_recv: the guard's per-item estimate (next_item_estimate); the array is sent uncompressed
        return item, {"shape": list(arr.shape), "seconds": round(time.time() - t0, 2), "bytes_recv": int(arr.nbytes)}, False

    n_fail = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        for i in range(0, len(todo), WORKERS):
            if budget.exhausted():
                print(f"budget reached ({budget.used() / 1e9:.3f} GB), stopping", flush=True)
                break
            for (v, g), meta, failed in pool.map(one, todo[i:i + WORKERS]):
                budget.record(f"{v}/{g}", meta, failed=failed)
                n_fail += failed
            if (i // WORKERS) % 25 == 0:
                print(f"{min(i + WORKERS, len(todo))}/{len(todo)} done, {n_fail} failed, {budget.used() / 1e9:.3f} GB", flush=True)
    budget.close()
    print("all done", flush=True)


def load(gfc_dir, version, name):
    p = os.path.join(gfc_dir, version, f"{name}.npy")
    return np.load(p) if os.path.exists(p) else None


def read_labels(data_root):
    """{farm: Hansen label array}, read once."""
    import rasterio

    out = {}
    for k, p in index_labels(Path(data_root) / "hansen_labels").items():
        with rasterio.open(p) as s:
            out[k] = s.read(1)
    return out


def read_fracs(labels, gfc_dir, version):
    """{farm: GFC forest percentage} for farms whose grid was fetched; the grid must match the label raster."""
    out = {}
    for k, lab in labels.items():
        frac = load(gfc_dir, version, f"farm_{k}")
        if frac is not None:
            assert lab.shape == frac.shape, (k, lab.shape, frac.shape)
            out[k] = frac
    return out


def farm_table(labels, fracs, threshold):
    return {k: gfc.farm_counts(labels[k], gfc.gfc_forest(f, threshold)) for k, f in fracs.items()}


def flag_block(col, m=None):
    """Flag counts and the share of Hansen flags that do not survive GFC gating, over farms m (all if None)."""
    hflag, gflag = gfc.gated_flags(col["class2"], col["class2_gfc"])
    m = np.ones(len(hflag), bool) if m is None else m
    return {"hansen_flagged": int(hflag[m].sum()), "gated_flagged": int(gflag[m].sum()),
            "share_flags_not_surviving": gfc.ratio_ci((hflag & ~gflag)[m].astype(float), hflag[m].astype(float))}


def summarise(table, group):
    keys = sorted(table)
    col = {c: np.array([table[k][c] for k in keys]) for c in table[keys[0]]}
    by_group = {}
    for g in sorted({group[k] for k in keys}):
        m = np.array([group[k] == g for k in keys])
        by_group[g] = {"n_farms": int(m.sum()), "loss_cells": int(col["class2"][m].sum()),
                       "share_loss_on_gfc_forest": gfc.ratio_ci(col["class2_gfc"][m], col["class2"][m])} | flag_block(col, m)
    return {"n_farms": len(keys), "loss_cells": int(col["class2"].sum()),
            "share_loss_on_gfc_forest": gfc.ratio_ci(col["class2_gfc"], col["class2"]),
            "share_class1_on_gfc_forest": gfc.ratio_ci(col["class1_gfc"], col["class1"]),
            "share_class0_on_gfc_forest": gfc.ratio_ci(col["class0_gfc"], col["class0"]),
            **flag_block(col), "agreement_forest_2020": gfc.agreement_from_counts(col), "by_group": by_group}


def s2c_check(data_root, gfc_dir, boxes_pairs):
    """PROTOCOL_E9 analysis 4 on the E4/E8 windows, primary version and threshold."""
    from e4_farms import label_on_window

    labels = index_labels(Path(data_root) / "hansen_labels")
    loss = np.zeros((3, 2), int)   # S2C class x GFC (non-forest, forest)
    stable = np.zeros(2, int)
    used, missing = [], []
    for wj, pj in boxes_pairs:
        W = json.load(open(wj))
        tag = Path(wj).parent.name
        boxes = dict(zip(W["farms"], np.array(W["boxes"])))
        preds = np.load(pj)
        for w in W["windows"]:
            name = f"win_{w['id']}"
            frac = load(gfc_dir, PRIMARY, f"{tag}_{name}")
            if name not in preds.files or frac is None:
                missing.append(f"{tag}_{name}")
                continue
            origin, shape = np.array(w["origin"]), tuple(w["shape"])
            p = preds[name]
            assert frac.shape == p.shape == shape, (tag, name, frac.shape, p.shape, shape)
            forest = gfc.gfc_forest(frac)
            used.append(f"{tag}_{name}")
            for k in w["farms"]:
                cells = E.farm_cells(origin, shape, boxes[k]) & (p >= 0)
                lab = label_on_window(labels[k], origin, shape)
                for c in range(3):
                    m = cells & (lab == 2) & (p == c)
                    loss[c] += [int((m & ~forest).sum()), int((m & forest).sum())]
                m = cells & (lab == 1)
                stable += [int((m & ~forest).sum()), int((m & forest).sum())]
    plant_nf = float(loss[1, 0] / loss[1].sum()) if loss[1].sum() else None
    forest_f = float(loss[0, 1] / loss[0].sum()) if loss[0].sum() else None
    return {"windows_used": len(used), "windows_missing": missing,
            "loss_cells_s2c_by_gfc": {n: {"gfc_nonforest": int(loss[i, 0]), "gfc_forest": int(loss[i, 1])}
                                      for i, n in enumerate(("FOREST", "PLANT", "OTHER"))},
            "share_s2c_plant_gfc_nonforest": plant_nf, "share_s2c_forest_gfc_forest": forest_f,
            "reading": gfc.s2c_reading(plant_nf, forest_f),
            "stable_cells_gfc_forest_share": float(stable[1] / stable.sum()) if stable.sum() else None,
            "note": "farm boxes overlap within a window, so a cell can be counted for more than one farm (as in E4)"}


def run(a):
    t0 = time.time()
    if len(a.boxes) % 2:
        sys.exit("--boxes takes pairs: windows.json farm_preds.npz")
    z = np.load(a.features, allow_pickle=True)
    group = {str(k): f"{c}|{p}" for k, c, p in zip(z["keys"], z["country"], z["crop"])}
    labels = read_labels(a.data_root)
    fracs = {v: read_fracs(labels, a.gfc_dir, v) for v in ASSETS}
    if not fracs[PRIMARY]:
        sys.exit(f"no {PRIMARY} grids in {a.gfc_dir}")
    R = {"protocol": "PROTOCOL_E9.md", "primary": {"version": PRIMARY, "threshold_pct": gfc.FOREST_PCT},
         "missing_farms": sorted(set(labels) - set(fracs[PRIMARY]))}
    R["analyses_1_to_3"] = summarise(farm_table(labels, fracs[PRIMARY], gfc.FOREST_PCT), group)
    print(json.dumps({k: v for k, v in R["analyses_1_to_3"].items() if k != "by_group"}, indent=1), flush=True)
    R["analysis_4_s2c"] = s2c_check(a.data_root, a.gfc_dir, list(zip(a.boxes[::2], a.boxes[1::2])))
    print(json.dumps(R["analysis_4_s2c"], indent=1), flush=True)
    sens = {}
    for v in ASSETS:
        for t in THRESHOLDS:
            if (v, t) == (PRIMARY, gfc.FOREST_PCT) or not fracs[v]:
                continue
            s = summarise(farm_table(labels, fracs[v], t), group)
            sens[f"{v}_{t}"] = {k: s[k] for k in ("n_farms", "share_loss_on_gfc_forest", "hansen_flagged", "gated_flagged",
                                                   "share_flags_not_surviving")}
    R["analysis_5_sensitivity"] = sens
    R["seconds"] = round(time.time() - t0, 1)
    os.makedirs(a.results_dir, exist_ok=True)
    json.dump(R, open(os.path.join(a.results_dir, "gfc2020.json"), "w"), indent=1)
    print(json.dumps(sens, indent=1), "\nsaved", R["seconds"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    for n in ("data_root", "env_file", "out_dir"):
        d.add_argument(n)
    d.add_argument("windows", nargs="*")
    d.add_argument("--budget-gb", type=float, default=1.0)
    d.add_argument("--retry-failed", action="store_true")
    r = sub.add_parser("run")
    for n in ("data_root", "features", "gfc_dir", "results_dir"):
        r.add_argument(n)
    r.add_argument("--boxes", nargs="*", default=[], help="pairs: windows.json farm_preds.npz")
    a = ap.parse_args()
    if a.cmd == "download":
        with exclusive_download():
            download(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
