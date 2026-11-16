"""PROTOCOL_E12: OpenStreetMap outlines for every farm, and Hansen flags and the E9 GFC2020 gate inside them.

usage:
  python -I e12_outlines.py fetch <labels_dir> <outlines_dir> [--endpoint URL] [--cap-mb 100]
  python -I e12_outlines.py run <labels_dir> <outlines_dir> <gfc_dir> <features.npz> <results_dir> [--e5 farms.json ...]
fetch: one Overpass request (`out geom meta`) for every way and relation id in <labels_dir>; streams the response under a cap
       on the job's own bytes and the machine-wide download lock; writes overpass.json and manifest.json (no geometries in
       the manifest). Data (c) OpenStreetMap contributors, ODbL 1.0. Skips if present; no retries.
run:   the PROTOCOL_E12 analyses; writes <results_dir>/outlines.json. --e5 takes the E5 and E8 outline farms.json files for
       analysis 5.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from gfc2020 import PRIMARY, load, summarise  # noqa: E402

from hansen_audit import gfc  # noqa: E402
from hansen_audit import outlines as O  # noqa: E402
from hansen_audit.budget import capped_download, exclusive_download  # noqa: E402
from hansen_audit.decisions import DEFAULT_MIN_CELLS, farm_flag  # noqa: E402
from hansen_audit.io import farm_groups, index_labels  # noqa: E402

ENDPOINT = "https://overpass-api.de/api/interpreter"  # PROTOCOL_E8 deviation: the kumi mirror timed out
GATE_FIELDS = ("loss_cells", "share_loss_on_gfc_forest", "hansen_flagged", "gated_flagged", "share_flags_not_surviving")


def overpass_query(keys):
    ways = [k.split("_", 1)[1] for k in keys if k.startswith("way_")]
    rels = [k.split("_", 1)[1] for k in keys if k.startswith("relation_")]
    return f"[out:json][timeout:600];(way(id:{','.join(ways)});relation(id:{','.join(rels)}););out geom meta;"


def fetch(a):
    import requests

    os.makedirs(a.outlines_dir, exist_ok=True)
    out = os.path.join(a.outlines_dir, "overpass.json")
    keys = sorted(index_labels(Path(a.labels_dir)))
    cap = int(a.cap_mb * 1e6)
    if os.path.exists(out):
        print(f"{out} exists; not fetching again")
        if not os.path.exists(os.path.join(a.outlines_dir, "manifest.json")):
            t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(out)))
            write_manifest(a.outlines_dir, out, keys, a.endpoint, t, os.path.getsize(out), cap)
        return
    t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with exclusive_download(), requests.post(a.endpoint, data={"data": overpass_query(keys)}, timeout=900, stream=True,
                                             headers={"User-Agent": "hansen-audit-research/1.0"}) as r:
        r.raise_for_status()
        n = capped_download(r, out, cap)
    reject_partial(out)
    write_manifest(a.outlines_dir, out, keys, a.endpoint, t, n, cap)


def reject_partial(path):
    """Overpass answers 200 with a 'remark' when a query runs out of time or memory, and the elements are then partial.
    Move such a reply aside (so the next fetch tries again) and stop."""
    with open(path, encoding="utf-8") as fh:
        remark = json.load(fh).get("remark")
    if remark:
        os.replace(path, path + ".rejected")
        raise SystemExit(f"Overpass reply is partial ({remark[:200]}); moved to {path}.rejected")


def write_manifest(outlines_dir, out, keys, endpoint, t, n, cap):
    """manifest.json beside the outlines: time (UTC; the file time if written after the fetch), endpoint, bytes, ids
    requested and missing, OSM versions and timestamps. No geometries."""
    els = read_elements(out)
    man = {"fetched_utc": t, "endpoint": endpoint, "bytes": n, "cap_bytes": cap, "requested": len(keys),
           "returned": len(els), "missing": sorted(set(keys) - {O.farm_key(e) for e in els}),
           "objects": {O.farm_key(e): {"version": e.get("version"), "timestamp": e.get("timestamp")} for e in els},
           "licence": "(c) OpenStreetMap contributors, ODbL 1.0"}
    with open(os.path.join(outlines_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    print(json.dumps({k: man[k] for k in ("fetched_utc", "bytes", "requested", "returned")} | {"missing": len(man["missing"])},
                     indent=1))


def read_elements(path):
    """Overpass JSON elements (the response is UTF-8; Windows would otherwise read it as cp1252)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("elements", [])


def read_outline(lab_path, poly):
    """(label array, transform, bool outline cells, share of the outline's area inside the raster bounds)."""
    import rasterio

    with rasterio.open(lab_path) as s:
        lab, tr, b = s.read(1), tuple(s.transform)[:6], s.bounds
    return lab, tr, O.outline_cells_on(tr, lab.shape, poly), O.coverage(poly, (b.left, b.bottom, b.right, b.top))


def grid_keys(tr, inside):
    """Global keys of the outline cells on the label grid (all label rasters share one cell size)."""
    r, c = np.nonzero(inside)
    return zip(np.round(tr[2] / tr[0] + c).astype(int).tolist(), np.round(tr[5] / tr[4] + r).astype(int).tolist())


def farm_rows(labels, els, gfc_dir):
    """Per farm: Hansen x GFC counts over the box and over the outline, outline size and coverage; exclusions with reasons;
    and how many farms' outlines hold each outline cell."""
    rows, excluded, holders = {}, {}, Counter()
    for k in sorted(labels):
        poly = O.element_polygon(els[k]) if k in els else None
        frac = load(gfc_dir, PRIMARY, f"farm_{k}")
        reason = ("not returned by Overpass" if k not in els else "not a closed area" if poly is None else
                  "no GFC2020 grid" if frac is None else None)
        if reason:
            excluded[k] = reason
            continue
        lab, tr, inside, cov = read_outline(labels[k], poly)
        if not inside.any():
            excluded[k] = "no label cell inside the outline"
            continue
        assert frac.shape == lab.shape, (k, frac.shape, lab.shape)
        forest = gfc.gfc_forest(frac)
        rows[k] = {"box": gfc.farm_counts(lab, forest), "outline": gfc.farm_counts(np.where(inside, lab, gfc.NODATA), forest),
                   "outline_cells": int(inside.sum()), "outline_ha": round(float(O.equal_area(poly).area / 1e4), 3),
                   "coverage": round(cov, 4)}
        holders.update(set(grid_keys(tr, inside)))
    return rows, excluded, holders


def flag_compare(rows, keys):
    """Analysis 1 over `keys`: box against outline flags on the same farms."""
    if not keys:
        return None
    fb = farm_flag(np.array([rows[k]["box"]["class2"] for k in keys]))
    fo = farm_flag(np.array([rows[k]["outline"]["class2"] for k in keys]))
    return {"n_farms": len(keys), "box_flagged": int(fb.sum()), "outline_flagged": int(fo.sum()),
            "flagged_outline_not_box": int((fo & ~fb).sum()),
            "share_box_flags_not_flagged_in_outline": gfc.ratio_ci((fb & ~fo).astype(float), fb.astype(float))}


def block(rows, keys, group):
    """Analyses 1 and 2 over `keys`, overall and by group."""
    s = summarise({k: rows[k]["outline"] for k in keys}, group)
    return {"analysis_1_flags": flag_compare(rows, keys),
            "analysis_2_gfc_gate_in_outlines": {f: s[f] for f in GATE_FIELDS},
            "by_group": {g: {"analysis_1_flags": flag_compare(rows, [k for k in keys if group[k] == g]),
                             "analysis_2": {f: s["by_group"][g][f] for f in ("n_farms",) + GATE_FIELDS}}
                         for g in sorted(s["by_group"])}}


def check_e5(rows, paths):
    """Analysis 5: outline flags here against E5/E8 (their grid and box differ, so differences are listed, not dropped)."""
    out = {"compared": 0, "agree": 0, "disagree": []}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            per_farm = json.load(fh)["per_farm"]
        for k, v in per_farm.items():
            if k not in rows:
                continue
            prior, here = int(sum(v["loss_pred_2020"])), rows[k]["outline"]["class2"]
            out["compared"] += 1
            if bool(farm_flag(prior)) == bool(farm_flag(here)):
                out["agree"] += 1
            else:
                out["disagree"].append({"farm": k, "e5_e8_loss_cells": prior, "e12_loss_cells": here,
                                        "e5_e8_coverage": v.get("coverage"), "e12_coverage": rows[k]["coverage"]})
    return out


def run(a):
    t0 = time.time()
    z = np.load(a.features, allow_pickle=True)
    group = farm_groups(z)
    labels = index_labels(Path(a.labels_dir))
    els = {O.farm_key(e): e for e in read_elements(os.path.join(a.outlines_dir, "overpass.json"))}
    rows, excluded, holders = farm_rows(labels, els, a.gfc_dir)
    keys = sorted(rows)
    missing_group = [k for k in keys if k not in group]
    assert not missing_group, f"farms without a group in {a.features}: {missing_group[:5]}"
    cells = np.array([rows[k]["outline_cells"] for k in keys])
    R = {"protocol": "PROTOCOL_E12.md", "labels": Path(a.labels_dir).name, "gfc": f"{PRIMARY} at {gfc.FOREST_PCT} %",
         "n_farms": len(labels), "n_included": len(keys), "excluded_by_reason": dict(Counter(excluded.values())),
         "excluded": excluded, "primary": block(rows, keys, group),
         "analysis_3": {"outline_ha_quartiles": np.percentile([rows[k]["outline_ha"] for k in keys], [25, 50, 75]).round(2).tolist(),
                        "coverage_quartiles": np.percentile([rows[k]["coverage"] for k in keys], [25, 50, 75]).round(3).tolist(),
                        "outlines_under_min_cells": int((cells < DEFAULT_MIN_CELLS).sum()), "min_cells": DEFAULT_MIN_CELLS,
                        "outline_cells": len(holders), "outline_cells_in_two_or_more": sum(v > 1 for v in holders.values())},
         "analysis_4_coverage_ge_0_5": block(rows, [k for k in keys if rows[k]["coverage"] >= 0.5], group),
         "analysis_5_vs_e5_e8": check_e5(rows, a.e5) if a.e5 else None,
         "per_farm": rows}
    R["seconds"] = round(time.time() - t0, 1)
    os.makedirs(a.results_dir, exist_ok=True)
    with open(os.path.join(a.results_dir, "outlines.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
    print(json.dumps({k: R[k] for k in ("n_included", "excluded_by_reason", "analysis_3", "analysis_5_vs_e5_e8")}, indent=1))
    print(json.dumps({k: R["primary"][k] for k in ("analysis_1_flags", "analysis_2_gfc_gate_in_outlines")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    for n in ("labels_dir", "outlines_dir"):
        f.add_argument(n)
    f.add_argument("--endpoint", default=ENDPOINT)
    f.add_argument("--cap-mb", type=float, default=100)
    r = sub.add_parser("run")
    for n in ("labels_dir", "outlines_dir", "gfc_dir", "features", "results_dir"):
        r.add_argument(n)
    r.add_argument("--e5", action="append", help="E5 or E8 outline farms.json (analysis 5)")
    a = ap.parse_args()
    fetch(a) if a.cmd == "fetch" else run(a)


if __name__ == "__main__":
    main()
