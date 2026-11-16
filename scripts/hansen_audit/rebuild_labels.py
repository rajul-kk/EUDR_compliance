"""PROTOCOL_E10: rebuild every farm's Hansen label raster on its existing grid, with stable forest as class 1.

usage:
  python -I rebuild_labels.py download <data_root> <env_file> <out_dir> [--budget-gb 0.2]
  python -I rebuild_labels.py compare <data_root> <out_dir> <features.npz> <results.json>
download: per farm, treecover2000, lossyear (unmasked to 0) and datamask from Hansen v1.13 on the grid of the label file on
          disk, computed into a label by hansen_audit.reference.hansen_label (255 where datamask is 0) and written as
          <out_dir>/<farm>_hansen_label.tif (same transform, nodata 255). Resumable, under DownloadBudget.
compare:  the PROTOCOL_E10 checks of the rebuilt labels against the files on disk.
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
from hansen_audit.budget import exclusive_download  # noqa: E402
from hansen_audit.io import index_labels  # noqa: E402
from hansen_audit.reference import hansen_label  # noqa: E402

ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
NODATA = 255
WORKERS = 8


def label_from_bands(treecover, lossyear, datamask):
    """The farm label: reference.hansen_label on the bands, 255 where Hansen has no data."""
    lab = hansen_label(treecover, lossyear)
    lab[datamask == 0] = NODATA
    return lab


def fetch_bands(transform, shape):
    import ee

    a, b, c, d, e, f = transform
    h = ee.Image(ASSET)
    img = (h.select("treecover2000").rename("tc").addBands(h.select("lossyear").unmask(0).rename("ly"))
           .addBands(h.select("datamask").rename("dm")).toUint8())
    arr = ee.data.computePixels({
        "expression": img, "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": int(shape[1]), "height": int(shape[0])},
                 "affineTransform": {"scaleX": a, "shearX": b, "translateX": c, "shearY": d, "scaleY": e, "translateY": f},
                 "crsCode": "EPSG:4326"}})
    return np.asarray(arr["tc"]), np.asarray(arr["ly"]), np.asarray(arr["dm"])


def write_label(path, lab, profile):
    import rasterio

    tmp = path + ".part.tif"
    with rasterio.open(tmp, "w", **(profile | {"nodata": NODATA, "dtype": "uint8", "count": 1})) as dst:
        dst.write(lab, 1)
    os.replace(tmp, path)


def download(a):
    import ee
    import rasterio
    from dotenv import dotenv_values

    from hansen_audit.budget import DownloadBudget

    project = dotenv_values(a.env_file).get("GEE_PROJECT_ID")
    if not project:
        sys.exit("GEE_PROJECT_ID missing from the env file")
    ee.Initialize(project=project)
    old = index_labels(Path(a.data_root) / "hansen_labels")
    os.makedirs(a.out_dir, exist_ok=True)
    budget = DownloadBudget(a.out_dir, a.budget_gb, "farms", retry_failed=a.retry_failed)
    todo = [k for k in old if not budget.done(k, os.path.join(a.out_dir, old[k].name))]
    print(f"{len(old)} farms, {len(todo)} to fetch", flush=True)

    def one(k):
        t0 = time.time()
        with rasterio.open(old[k]) as s:
            profile, transform, shape = s.profile, tuple(s.transform)[:6], s.shape
        try:
            tc, ly, dm = fetch_bands(transform, shape)
        except Exception as ex:  # noqa: BLE001 - recorded per farm, retried with --retry-failed
            return k, {"error": f"{type(ex).__name__}: {str(ex).replace(project, '<project>')}"[:300]}, True
        write_label(os.path.join(a.out_dir, old[k].name), label_from_bands(tc, ly, dm), profile)
        return k, {"seconds": round(time.time() - t0, 2), "bytes_recv": int(tc.nbytes * 3)}, False

    n_fail = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        for i in range(0, len(todo), WORKERS):
            if budget.exhausted():
                print(f"budget reached ({budget.used() / 1e9:.3f} GB), stopping", flush=True)
                break
            for k, meta, failed in pool.map(one, todo[i:i + WORKERS]):
                budget.record(k, meta, failed=failed)
                n_fail += failed
            if (i // WORKERS) % 25 == 0:
                print(f"{min(i + WORKERS, len(todo))}/{len(todo)} done, {n_fail} failed, {budget.used() / 1e9:.3f} GB", flush=True)
    budget.close()
    print("all done", flush=True)


def class_shares(labs, keys):
    tot = np.zeros(4)
    for k in keys:
        lab = labs[k]
        tot += [(lab == 0).sum(), (lab == 1).sum(), (lab == 2).sum(), (lab == NODATA).sum()]
    return dict(zip(("class0", "class1", "class2", "nodata"), (tot / tot.sum()).round(4).tolist())) if tot.sum() else None


def compare(a):
    import rasterio

    old_paths = index_labels(Path(a.data_root) / "hansen_labels")
    new_paths = index_labels(Path(a.out_dir))
    keys = sorted(set(old_paths) & set(new_paths))
    old, new = {}, {}
    for k in keys:
        with rasterio.open(old_paths[k]) as s:
            old[k] = s.read(1)
        with rasterio.open(new_paths[k]) as s:
            new[k] = s.read(1)
    class2_differs, old1_not_new1 = [], []
    for k in keys:
        ok = new[k] != NODATA
        if ((old[k] == 2) != (new[k] == 2))[ok].any():
            class2_differs.append(k)
        if ((old[k] == 1) & ok & (new[k] != 1)).any():
            old1_not_new1.append(k)
    z = np.load(a.features, allow_pickle=True)
    group = {str(k): f"{c}|{p}" for k, c, p in zip(z["keys"], z["country"], z["crop"])}
    R = {"protocol": "PROTOCOL_E10.md", "asset": ASSET, "n_rebuilt": len(keys),
         "missing": sorted(set(old_paths) - set(new_paths)),
         "class2_identical_outside_nodata": not class2_differs, "farms_class2_differs": class2_differs,
         "old_class1_subset_of_new_class1": not old1_not_new1, "farms_old1_not_new1": old1_not_new1,
         "shares_old": class_shares(old, keys), "shares_new": class_shares(new, keys),
         "shares_new_by_group": {g: class_shares(new, [k for k in keys if group.get(k) == g])
                                 for g in sorted({group.get(k, "?") for k in keys})}}
    os.makedirs(os.path.dirname(a.results) or ".", exist_ok=True)
    json.dump(R, open(a.results, "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "shares_new_by_group"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    for n in ("data_root", "env_file", "out_dir"):
        d.add_argument(n)
    d.add_argument("--budget-gb", type=float, default=0.2)
    d.add_argument("--retry-failed", action="store_true")
    c = sub.add_parser("compare")
    for n in ("data_root", "out_dir", "features", "results"):
        c.add_argument(n)
    a = ap.parse_args()
    if a.cmd == "download":
        with exclusive_download():
            download(a)
    else:
        compare(a)


if __name__ == "__main__":
    main()
