"""Download Sentinel-1 RTC annual statistics (2020 and reference year) for the reference grids, on their Hansen grid (PROTOCOL_S2C).

usage: python -I s1_download.py <reference_cells.npz> <grid_index.csv> <out_dir> [--grids 01,02] [--budget-gb 4.5]
Writes <out_dir>/grid_XX_<period>.npz (cells, channels) and <out_dir>/manifest.json. Resumable; files are written atomically.
Stops when system-wide received bytes, summed over all runs in the manifest, exceed the budget.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hansen_audit import s1  # noqa: E402

for k, v in dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_MAX_RETRY="4",
                 GDAL_HTTP_RETRY_DELAY="2", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff").items():
    os.environ.setdefault(k, v)


def periods(label_year: int):
    out = {"2020": ("2020-01-01", "2020-12-31")}
    if label_year != 2020:
        out["ref"] = (f"{label_year}-01-01", f"{label_year}-12-31")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_npz")
    ap.add_argument("grid_index")
    ap.add_argument("out_dir")
    ap.add_argument("--grids", default=None)
    ap.add_argument("--budget-gb", type=float, default=4.5)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    man_path = os.path.join(a.out_dir, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {"runs": [], "grids": {}}
    used_before = sum(r.get("bytes_recv", 0) for r in manifest["runs"])
    want = set(a.grids.split(",")) if a.grids else None
    ref = np.load(a.ref_npz)
    tok = s1.Token()
    net0 = psutil.net_io_counters().bytes_recv
    run = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "bytes_recv": 0, "stopped_by_budget": False}
    manifest["runs"].append(run)

    def used():
        return used_before + psutil.net_io_counters().bytes_recv - net0

    def save():
        run["bytes_recv"] = psutil.net_io_counters().bytes_recv - net0
        tmp = man_path + ".part"
        json.dump(manifest, open(tmp, "w"), indent=1)
        os.replace(tmp, man_path)

    for r in csv.DictReader(open(a.grid_index)):
        gid = r["grid_id"]
        if want and gid not in want:
            continue
        origin, shape = ref[f"origin_{gid}"], ref[f"hansen_{gid}"].shape
        for pname, (start, end) in periods(int(r["label_year"])).items():
            f = os.path.join(a.out_dir, f"grid_{gid}_{pname}.npz")
            if os.path.exists(f):
                continue
            if used() > a.budget_gb * 1e9:
                print(f"budget reached ({used() / 1e9:.2f} GB), stopping", flush=True)
                run["stopped_by_budget"] = True
                save()
                return
            t0, b0 = time.time(), psutil.net_io_counters().bytes_recv
            print(f"grid {gid} {pname} ({start[:4]}), {r['island']}", flush=True)
            cells, meta = s1.build_s1(origin, shape, start, end, tok, log=lambda m: print(m, flush=True))
            meta.update(seconds=round(time.time() - t0, 1), bytes_recv=psutil.net_io_counters().bytes_recv - b0)
            if cells is not None:
                tmp = f[:-4] + ".part.npz"
                np.savez_compressed(tmp, cells=cells, channels=np.array(s1.CHANNELS), origin=origin)
                os.replace(tmp, f)
                meta["valid_cell_frac"] = float(np.isfinite(cells[0]).mean())
            manifest["grids"][f"{gid}_{pname}"] = meta
            print(f"  done in {meta['seconds']} s, {meta['bytes_recv'] / 1e6:.0f} MB (system-wide), valid {meta.get('valid_cell_frac')}; "
                  f"total {used() / 1e9:.2f} GB", flush=True)
            save()
    save()
    print("all done", flush=True)


if __name__ == "__main__":
    main()
