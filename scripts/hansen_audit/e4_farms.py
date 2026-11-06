"""PROTOCOL_S2 E4 (deviation 5): 2020 land use under Hansen loss on the 94 Indonesian oil palm farms, with the S2C classifier.

usage:
  python -I e4_farms.py download <data_root> <features.npz> <out_dir> [--budget-gb 2]
  python -I e4_farms.py run <data_root> <features.npz> <out_dir> <reference_cells.npz> <grid_index.csv> <s2_dir> <s1_dir> <results_dir>
download: groups farms into <= 0.05 deg windows, writes <out_dir>/windows.json and <out_dir>/win_XX.npz (S2 + S1 2020 cells), resumable,
          atomic writes, own byte guard (system-wide received bytes summed over runs).
run:      fits the S2C classifier once on all reference grids, predicts 2020 land use in each window, writes <results_dir>/farms.json
          and <results_dir>/farm_preds.npz.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hansen_audit import farms_e4 as E  # noqa: E402
from hansen_audit import landuse as L  # noqa: E402
from hansen_audit import s1, s2  # noqa: E402
from hansen_audit.io import index_images, index_labels  # noqa: E402

for k, v in dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_MAX_RETRY="4",
                 GDAL_HTTP_RETRY_DELAY="2", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff").items():
    os.environ.setdefault(k, v)


def farms_and_windows(root, features_npz):
    import rasterio

    z = np.load(features_npz, allow_pickle=True)
    keys = [str(k) for k in z["keys"][(z["country"] == "ID") & (z["crop"] == "Oil Palm")]]
    imgs = index_images(Path(root) / "2020_baseline")
    boxes = []
    for k in keys:
        with rasterio.open(imgs[k][0][1]) as s:
            boxes.append(tuple(s.bounds))
    boxes = np.array(boxes)
    wins = []
    for i, g in enumerate(E.group_windows(boxes)):
        origin, shape = E.window_grid(boxes[g])
        wins.append({"id": f"{i:02d}", "farms": [keys[j] for j in g], "origin": origin.tolist(), "shape": list(shape)})
    return keys, boxes, wins


def download(a):
    import psutil

    os.makedirs(a.out_dir, exist_ok=True)
    keys, boxes, wins = farms_and_windows(a.data_root, a.features)
    json.dump({"farms": keys, "boxes": boxes.tolist(), "windows": wins}, open(os.path.join(a.out_dir, "windows.json"), "w"), indent=1)
    man_path = os.path.join(a.out_dir, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {"runs": [], "windows": {}}
    used_before = sum(r.get("bytes_recv", 0) for r in man["runs"])
    net0 = psutil.net_io_counters().bytes_recv
    run = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "bytes_recv": 0, "stopped_by_budget": False}
    man["runs"].append(run)

    def used():
        return used_before + psutil.net_io_counters().bytes_recv - net0

    def save():
        run["bytes_recv"] = psutil.net_io_counters().bytes_recv - net0
        json.dump(man, open(man_path + ".part", "w"), indent=1)
        os.replace(man_path + ".part", man_path)

    t2, t1 = s2.Token(), s1.Token()
    print(f"{len(keys)} farms in {len(wins)} windows", flush=True)
    for w in wins:
        f = os.path.join(a.out_dir, f"win_{w['id']}.npz")
        if os.path.exists(f):
            continue
        if used() > a.budget_gb * 1e9:
            print(f"budget reached ({used() / 1e9:.2f} GB), stopping", flush=True)
            run["stopped_by_budget"] = True
            save()
            return
        t0, b0 = time.time(), psutil.net_io_counters().bytes_recv
        origin, shape = np.array(w["origin"]), tuple(w["shape"])
        print(f"window {w['id']}: {len(w['farms'])} farms, shape {shape}", flush=True)
        c2, m2 = s2.build_composite(origin, shape, "2020-01-01", "2020-12-31", t2, log=lambda m: print(m, flush=True))
        c1, m1 = s1.build_s1(origin, shape, "2020-01-01", "2020-12-31", t1, log=lambda m: print(m, flush=True))
        meta = {"s2": m2, "s1": m1, "seconds": round(time.time() - t0, 1), "bytes_recv": psutil.net_io_counters().bytes_recv - b0}
        if c2 is not None:
            tmp = f[:-4] + ".part.npz"
            np.savez_compressed(tmp, s2=c2, s2_channels=np.array(s2.CHANNELS),
                                s1=c1 if c1 is not None else np.zeros(0), s1_channels=np.array(s1.CHANNELS), origin=origin)
            os.replace(tmp, f)
        man["windows"][w["id"]] = meta
        print(f"  done in {meta['seconds']} s, {meta['bytes_recv'] / 1e6:.0f} MB; total {used() / 1e9:.2f} GB", flush=True)
        save()
    save()
    print("all done", flush=True)


def label_on_window(label_path, origin, shape):
    import rasterio
    from rasterio.warp import Resampling, reproject

    out = np.full(shape, 255, np.uint8)
    with rasterio.open(label_path) as src:
        reproject(rasterio.band(src, 1), out, dst_transform=s1.hansen_transform(origin), dst_crs="EPSG:4326",
                  resampling=Resampling.nearest, dst_nodata=255)
    return out


def fit_s2c(ref_npz, grid_index, s2_dir, s1_dir):
    import lightgbm as lgb

    rows = list(csv.DictReader(open(grid_index)))
    ref = np.load(ref_npz)
    rng = np.random.default_rng(0)
    X, Y = [], []
    for r in rows:
        gid = r["grid_id"]
        p = "2020" if int(r["label_year"]) == 2020 else "ref"
        z = np.load(os.path.join(s2_dir, f"grid_{gid}_{p}.npz"))
        z1 = np.load(os.path.join(s1_dir, f"grid_{gid}_{p}.npz"))
        F = L.s2c_features(z["cells"], tuple(z["channels"].tolist()), z1["cells"], tuple(z1["channels"].tolist()))
        m = (ref[f"cov_{gid}"].astype(np.float32) >= 0.5) & np.isfinite(F[0])
        idx = np.flatnonzero(m.ravel())
        if len(idx) > L.MAX_TRAIN_PER_GRID:
            idx = rng.choice(idx, L.MAX_TRAIN_PER_GRID, replace=False)
        X.append(F.reshape(F.shape[0], -1)[:, idx].T)
        Y.append(L.group_index(ref[f"classid_{gid}"]).ravel()[idx])
    X, Y = np.concatenate(X), np.concatenate(Y)
    print(f"training on {len(Y)} cells from {len(rows)} grids", flush=True)
    return lgb.LGBMClassifier(**L.LGBM_PARAMS).fit(X, Y), int(len(Y))


def run(a):
    t0 = time.time()
    os.makedirs(a.results_dir, exist_ok=True)
    W = json.load(open(os.path.join(a.out_dir, "windows.json")))
    boxes = dict(zip(W["farms"], np.array(W["boxes"])))
    labels = index_labels(Path(a.data_root) / "hansen_labels")
    clf, n_train = fit_s2c(a.ref_npz, a.grid_index, a.s2_dir, a.s1_dir)
    per_farm, preds, missing = {}, {}, []
    for w in W["windows"]:
        f = os.path.join(a.out_dir, f"win_{w['id']}.npz")
        if not os.path.exists(f):
            missing += w["farms"]
            continue
        z = np.load(f)
        origin, shape = np.array(w["origin"]), tuple(w["shape"])
        c1 = z["s1"] if z["s1"].size else None
        F = L.s2c_features(z["s2"], tuple(z["s2_channels"].tolist()), c1, tuple(z["s1_channels"].tolist()))
        valid = np.isfinite(F[0])
        p = np.full(shape, -1, np.int8)
        if valid.any():
            p[valid] = clf.predict(F[:, valid].T)
        preds[f"win_{w['id']}"] = p
        for k in w["farms"]:
            cells = E.farm_cells(origin, shape, boxes[k])
            lab = label_on_window(labels[k], origin, shape)
            loss = cells & (lab == 2)
            lv = loss & valid
            per_farm[k] = {"window": w["id"], "cells": int(cells.sum()), "loss_cells": int(loss.sum()), "loss_cells_valid": int(lv.sum()),
                           "loss_pred_2020": np.bincount(p[lv], minlength=3).tolist(),
                           "stable_pred_2020": np.bincount(p[cells & (lab == 1) & valid], minlength=3).tolist()}
    counts = np.array([v["loss_pred_2020"] for v in per_farm.values()])
    out = {"classifier": "S2C features, fit on all 72 reference grids", "n_train_cells": n_train, "missing_farms": missing,
           "summary": E.farm_summary(counts), "per_farm": per_farm,
           "caveat": "transported model output: no reference covers these farms; raw predictions, no per-cell correction"}
    json.dump(out, open(os.path.join(a.results_dir, "farms.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.results_dir, "farm_preds.npz"), **preds)
    print(json.dumps(out["summary"], indent=1))
    print(f"done in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    for n in ("data_root", "features", "out_dir"):
        d.add_argument(n)
    d.add_argument("--budget-gb", type=float, default=2.0)
    r = sub.add_parser("run")
    for n in ("data_root", "features", "out_dir", "ref_npz", "grid_index", "s2_dir", "s1_dir", "results_dir"):
        r.add_argument(n)
    a = ap.parse_args()
    download(a) if a.cmd == "download" else run(a)


if __name__ == "__main__":
    main()
