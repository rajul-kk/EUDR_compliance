"""PROTOCOL_E5: fetch the farms' OpenStreetMap outlines and recount E4 inside them.

usage:
  python -I e5_outlines.py fetch <e4_out_dir> <outlines_dir>
  python -I e5_outlines.py run <data_root> <e4_out_dir> <outlines_dir> <e4_results_dir> <results_dir>
fetch: one Overpass request for the 94 ways and relations (`out geom meta`); writes <outlines_dir>/overpass.json and
       manifest.json (fetch time, versions, timestamps). Data (c) OpenStreetMap contributors, ODbL. Skips if present.
run:   counts loss and stable cells inside each outline (clipped to the E4 box) with the E4 predictions, and writes
       <results_dir>/farms.json with the outputs listed in PROTOCOL_E5.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from hansen_audit import farms_e4 as E  # noqa: E402
from hansen_audit import outlines as O  # noqa: E402
from hansen_audit.io import index_labels  # noqa: E402

OVERPASS = "https://overpass.kumi.systems/api/interpreter"  # mirror used by src/find_farms.py; overpass-api.de returned 502


def fetch(a):
    import requests

    os.makedirs(a.outlines_dir, exist_ok=True)
    out = os.path.join(a.outlines_dir, "overpass.json")
    if os.path.exists(out):
        print(f"{out} exists; not fetching again")
        return
    farms = json.load(open(os.path.join(a.e4_out_dir, "windows.json")))["farms"]
    ways = [f.split("_", 1)[1] for f in farms if f.startswith("way_")]
    rels = [f.split("_", 1)[1] for f in farms if f.startswith("relation_")]
    q = f"[out:json][timeout:180];(way(id:{','.join(ways)});relation(id:{','.join(rels)}););out geom meta;"
    t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    r = requests.post(OVERPASS, data={"data": q}, headers={"User-Agent": "hansen-audit-research/1.0"}, timeout=300)
    r.raise_for_status()
    d = r.json()
    tmp = out + ".tmp"
    json.dump(d, open(tmp, "w"))
    os.replace(tmp, out)
    els = d.get("elements", [])
    man = {"fetched_utc": t, "endpoint": OVERPASS, "query": q, "bytes": len(r.content), "requested": len(farms),
           "returned": len(els), "missing": sorted(set(farms) - {O.farm_key(e) for e in els}),
           "objects": {O.farm_key(e): {"version": e.get("version"), "timestamp": e.get("timestamp")} for e in els},
           "licence": "(c) OpenStreetMap contributors, ODbL 1.0"}
    json.dump(man, open(os.path.join(a.outlines_dir, "manifest.json"), "w"), indent=1)
    print(json.dumps({k: man[k] for k in ("fetched_utc", "bytes", "requested", "returned", "missing")}, indent=1))


def run(a):
    from e4_farms import label_on_window

    W = json.load(open(os.path.join(a.e4_out_dir, "windows.json")))
    boxes = dict(zip(W["farms"], np.array(W["boxes"])))
    els = {O.farm_key(e): e for e in json.load(open(os.path.join(a.outlines_dir, "overpass.json")))["elements"]}
    e4 = json.load(open(os.path.join(a.e4_results_dir, "farms.json")))["per_farm"]
    preds = np.load(os.path.join(a.e4_results_dir, "farm_preds.npz"))
    labels = index_labels(Path(a.data_root) / "hansen_labels")
    per_farm, excluded, owner = {}, {}, {}
    for w in W["windows"]:
        name = f"win_{w['id']}"
        if name not in preds.files:
            for k in w["farms"]:
                excluded[k] = "no E4 window data"
            continue
        origin, shape = np.array(w["origin"]), tuple(w["shape"])
        p = preds[name]
        valid = p >= 0
        for k in w["farms"]:
            if k not in els:
                excluded[k] = "not returned by Overpass"
                continue
            poly = O.element_polygon(els[k])
            if poly is None:
                excluded[k] = "not a closed area"
                continue
            cells = O.outline_cells(origin, shape, poly) & E.farm_cells(origin, shape, boxes[k])
            if not cells.any():
                excluded[k] = "no cell inside the box"
                continue
            lab = label_on_window(labels[k], origin, shape)
            loss, stable = cells & (lab == 2) & valid, cells & (lab == 1) & valid
            area_ha = float(O.equal_area(poly).area / 1e4)
            per_farm[k] = {"window": w["id"], "cells": int(cells.sum()), "outline_ha": round(area_ha, 2),
                           "coverage": round(O.coverage(poly, boxes[k]), 4),
                           "loss_pred_2020": np.bincount(p[loss], minlength=3).tolist(),
                           "stable_pred_2020": np.bincount(p[stable], minlength=3).tolist()}
            lon = origin[0] + (np.nonzero(cells)[1] + 0.5) * E.HANSEN_RES
            lat = origin[1] - (np.nonzero(cells)[0] + 0.5) * E.HANSEN_RES
            for g in zip(np.floor(lon / E.HANSEN_RES).astype(int).tolist(), np.floor(lat / E.HANSEN_RES).astype(int).tolist()):
                owner.setdefault(g, set()).add(k)

    keys = sorted(per_farm)
    out_c = np.array([per_farm[k]["loss_pred_2020"] for k in keys])
    box_c = np.array([e4[k]["loss_pred_2020"] for k in keys])

    def status(c):
        f = c.sum(1) >= E.MIN_LOSS_CELLS
        return f, f & (c[:, 0] < E.MIN_LOSS_CELLS)

    fo, po = status(out_c)
    fb, pb = status(box_c)

    def plant_share(rows):
        s = np.array(rows).sum(0)
        return {"cells": int(s.sum()), "share_PLANT": float(s[1] / s.sum()) if s.sum() else None}

    full = [i for i, k in enumerate(keys) if per_farm[k]["coverage"] >= 0.5]
    out = {
        "protocol": "PROTOCOL_E5.md", "n_included": len(keys), "excluded": excluded,
        "primary": {"outline": E.farm_summary(out_c), "box_same_farms": E.farm_summary(box_c)},
        "flag_changes": {"flagged_box_not_outline": [keys[i] for i in np.flatnonzero(fb & ~fo)],
                         "flagged_outline_not_box": [keys[i] for i in np.flatnonzero(fo & ~fb)],
                         "plantation_driven_box_not_outline": [keys[i] for i in np.flatnonzero(pb & fo & ~po)],
                         "plantation_driven_outline_not_box": [keys[i] for i in np.flatnonzero(po & fb & ~pb)]},
        "overlap": {"cells_in_outlines": len(owner), "cells_in_more_than_one": sum(len(v) > 1 for v in owner.values())},
        "transport_check": {"outline": plant_share([per_farm[k]["stable_pred_2020"] for k in keys]),
                            "box_same_farms": plant_share([e4[k]["stable_pred_2020"] for k in keys]),
                            "reading": "PLANT share >= 0.40: boxes included forest outside the plantation; <= 0.20: classifier "
                                       "under-calls palm on the farms; between: inconclusive (pre-registered)"},
        "secondary_not_partial": {"n": len(full), "outline": E.farm_summary(out_c[full]) if full else None,
                                  "transport_check": plant_share([per_farm[keys[i]]["stable_pred_2020"] for i in full]) if full else None},
        "per_farm": per_farm,
        "caveat": "transported S2C predictions as in E4; outlines from OpenStreetMap (ODbL), clipped to the E4 boxes",
    }
    os.makedirs(a.results_dir, exist_ok=True)
    json.dump(out, open(os.path.join(a.results_dir, "farms.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("n_included", "excluded", "primary", "flag_changes", "overlap", "transport_check")}, indent=1))
    print(json.dumps(out["secondary_not_partial"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("e4_out_dir")
    f.add_argument("outlines_dir")
    r = sub.add_parser("run")
    for n in ("data_root", "e4_out_dir", "outlines_dir", "e4_results_dir", "results_dir"):
        r.add_argument(n)
    a = ap.parse_args()
    fetch(a) if a.cmd == "fetch" else run(a)


if __name__ == "__main__":
    main()
