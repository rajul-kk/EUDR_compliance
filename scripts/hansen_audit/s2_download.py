"""Download Sentinel-2 composites (2020 and reference year) for the reference grids, on their Hansen grid.

usage: python -I s2_download.py <reference_cells.npz> <grid_index.csv> <out_dir> [--grids 01,02] [--budget-gb 4.5]
Writes <out_dir>/grid_XX_<period>.npz (cells, channels) and <out_dir>/manifest.json (scenes, bytes, timings).
Resumable: finished grid/period files are skipped. Stops when system-wide received bytes since start exceed the budget.
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
from hansen_audit import s2  # noqa: E402

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")


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
    ap.add_argument("--grids", default="")
    ap.add_argument("--budget-gb", type=float, default=4.5)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    man_path = os.path.join(a.out_dir, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {"runs": [], "grids": {}}
    used_before = sum(r.get("bytes_recv", 0) for r in manifest["runs"])
    rows = list(csv.DictReader(open(a.grid_index)))
    want = set(a.grids.split(",")) if a.grids else None
    ref = np.load(a.ref_npz)
    tok = s2.Token()
    net0 = psutil.net_io_counters().bytes_recv
    budget = a.budget_gb * 1e9
    run = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "bytes_recv": 0, "stopped_by_budget": False}
    manifest["runs"].append(run)

    def save_manifest():
        run["bytes_recv"] = psutil.net_io_counters().bytes_recv - net0
        json.dump(manifest, open(man_path, "w"), indent=1)

    for r in rows:
        gid = r["grid_id"]
        if want and gid not in want:
            continue
        origin, shape = ref[f"origin_{gid}"], ref[f"hansen_{gid}"].shape
        for pname, (start, end) in periods(int(r["label_year"])).items():
            f = os.path.join(a.out_dir, f"grid_{gid}_{pname}.npz")
            if os.path.exists(f):
                continue
            used = used_before + psutil.net_io_counters().bytes_recv - net0
            if used > budget:
                print(f"budget reached ({used / 1e9:.2f} GB), stopping", flush=True)
                run["stopped_by_budget"] = True
                save_manifest()
                return
            t0, b0 = time.time(), psutil.net_io_counters().bytes_recv
            print(f"grid {gid} {pname} ({start[:4]}), {r['island']}", flush=True)
            cells, meta = s2.build_composite(origin, shape, start, end, tok, log=lambda m: print(m, flush=True))
            meta.update(seconds=round(time.time() - t0, 1), bytes_recv=psutil.net_io_counters().bytes_recv - b0)
            if cells is not None:
                np.savez_compressed(f, cells=cells, channels=np.array(s2.CHANNELS), origin=origin)
                vf = float(np.mean(cells[s2.CHANNELS.index("VALID_FRAC")] >= s2.MIN_VALID_FRAC))
                meta["valid_cell_frac"] = vf
            manifest["grids"][f"{gid}_{pname}"] = meta
            print(f"  done in {meta['seconds']} s, {meta['bytes_recv'] / 1e6:.0f} MB (system-wide), "
                  f"valid cells {meta.get('valid_cell_frac')}; total {(used_before + psutil.net_io_counters().bytes_recv - net0) / 1e9:.2f} GB",
                  flush=True)
            save_manifest()
    save_manifest()
    print("all done", flush=True)


if __name__ == "__main__":
    main()
