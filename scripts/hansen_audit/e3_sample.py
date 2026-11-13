"""E3: reference sample on the farms. Population counts, the draw, and estimates from returned labels.

usage:
  python -I e3_sample.py population <data_root> <e4_out_dir> <farm_preds.npz> <out_json>
  python -I e3_sample.py draw <data_root> <e4_out_dir> <farm_preds.npz> <protocol_md> <out_dir> --n-total N --n-min M --seed S [--domain 6 7 8] [--second-frac F]
  python -I e3_sample.py census <data_root> <e4_out_dir> <farm_preds.npz> <outlines_dir> <protocol_md> <out_dir> --seed S [--second-frac F]
  python -I e3_sample.py estimate <population_json> <key_csv> <labels_csv> [--farms farms.json] [--key2 <key_csv> --labels2 <labels_csv>]
population: every cell centre in any of the 94 farm boxes, counted once (cells shared by overlapping boxes or windows are
            merged on the global Hansen grid), with its stratum (Hansen class x S2C 2020 prediction). Writes N_h and sample
            sizes for a range of interval half-widths. Draws nothing.
draw:       refuses unless <protocol_md> contains the line "Status: frozen". Writes sheet_A.csv / key_A.csv and a second blind
            pass sheet_B.csv / key_B.csv over the same cells, or a seeded share of them (--second-frac), in a different order
            with different ids.
census:     every Hansen loss cell inside a farm's OSM outline (as E5), all labelled; same sheets and the same frozen check.
estimate:   reads labels (sample_id,label) and writes the stratified shares in the loss domain, the S2C map accuracy on the
            farms and, with a second pass, labeller agreement.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from hansen_audit import farm_sample as S  # noqa: E402
from hansen_audit import farms_e4 as E  # noqa: E402
from hansen_audit.io import index_labels  # noqa: E402
from hansen_audit.s2 import HANSEN_RES  # noqa: E402

LOSS_DOMAIN = (6, 7, 8)


def population(data_root, e4_out_dir, preds_npz):
    from e4_farms import label_on_window

    W = json.load(open(os.path.join(e4_out_dir, "windows.json")))
    boxes = dict(zip(W["farms"], np.array(W["boxes"])))
    labels = index_labels(Path(data_root) / "hansen_labels")
    preds = np.load(preds_npz)
    cells = {}
    for w in W["windows"]:
        name = f"win_{w['id']}"
        if name not in preds.files:
            continue
        origin, shape = np.array(w["origin"]), tuple(w["shape"])
        p = preds[name]
        lon = origin[0] + (np.arange(shape[1]) + 0.5) * HANSEN_RES
        lat = origin[1] - (np.arange(shape[0]) + 0.5) * HANSEN_RES
        for k in w["farms"]:
            m = E.farm_cells(origin, shape, boxes[k])
            lab = label_on_window(labels[k], origin, shape)
            st = S.stratum_id(lab, p)
            for r, c in zip(*np.nonzero(m)):
                g = (int(np.floor(lon[c] / HANSEN_RES)), int(np.floor(lat[r] / HANSEN_RES)))
                if g in cells:
                    cells[g]["farms"].add(k)
                    continue
                cells[g] = {"lon": float(lon[c]), "lat": float(lat[r]), "stratum": int(st[r, c]), "hansen": int(lab[r, c]),
                            "pred": int(p[r, c]), "window": w["id"], "farms": {k}}
    return cells


def cmd_population(a):
    cells = population(a.data_root, a.e4_out_dir, a.preds)
    st = np.array([v["stratum"] for v in cells.values()])
    N_h = {int(h): int((st == h).sum()) for h in range(9)}
    loss = sum(N_h[h] for h in LOSS_DOMAIN)
    out = {"n_cells": len(cells), "n_invalid": int((st < 0).sum()), "n_shared_by_boxes": sum(len(v["farms"]) > 1 for v in cells.values()),
           "N_h": N_h, "strata": "3 * hansen (0 non-forest, 1 stable, 2 loss) + S2C 2020 prediction (0 FOREST, 1 PLANT, 2 OTHER)",
           "loss_cells": loss,
           "n_for_halfwidth_p05": {str(d): S.n_for_halfwidth(0.5, d) for d in (0.05, 0.07, 0.1)}}
    json.dump(out, open(a.out_json, "w"), indent=1)
    print(json.dumps(out, indent=1))


def cmd_draw(a):
    text = open(a.protocol).read()
    if "Status: frozen" not in text.splitlines():
        sys.exit(f"{a.protocol} is not frozen (needs a line 'Status: frozen'); nothing drawn")
    cells = population(a.data_root, a.e4_out_dir, a.preds)
    keys = list(cells)
    st = np.array([cells[g]["stratum"] for g in keys])
    dom = [int(d) for d in a.domain]
    N_h = {h: int((st == h).sum()) for h in dom}
    n_h = S.allocate(N_h, a.n_total, a.n_min)
    idx = S.draw(np.where(np.isin(st, dom), st, -1), n_h, a.seed)
    lon = np.array([cells[keys[i]]["lon"] for i in idx])
    lat = np.array([cells[keys[i]]["lat"] for i in idx])
    rows = [{"cell": f"{keys[i][0]}_{keys[i][1]}", "stratum": cells[keys[i]]["stratum"], "hansen": cells[keys[i]]["hansen"], "pred": cells[keys[i]]["pred"],
             "window": cells[keys[i]]["window"], "farms": ";".join(sorted(cells[keys[i]]["farms"]))} for i in idx]
    n_b = write_pair(a.out_dir, lon, lat, rows, a.seed, a.second_frac)
    json.dump({"N_h": N_h, "n_h": n_h, "seed": a.seed, "second_frac": a.second_frac, "n_second_pass": n_b, "protocol": a.protocol},
              open(os.path.join(a.out_dir, "design.json"), "w"), indent=1)
    print(json.dumps({"N_h": N_h, "n_h": n_h, "n_second_pass": n_b}, indent=1))


def write_pair(out_dir, lon, lat, rows, seed, second_frac):
    """Sheet A over every cell; blind sheet B over a seeded subset (all cells at fraction 1). Returns the size of B."""
    os.makedirs(out_dir, exist_ok=True)
    S.write_sheets(os.path.join(out_dir, "sheet_A.csv"), os.path.join(out_dir, "key_A.csv"), lon, lat, rows, seed + 1, "A")
    b = S.second_pass_subset(len(rows), second_frac, seed + 3)
    S.write_sheets(os.path.join(out_dir, "sheet_B.csv"), os.path.join(out_dir, "key_B.csv"), lon[b], lat[b], [rows[i] for i in b],
                   seed + 2, "B")
    return int(len(b))


def outline_set(e4_out_dir, outlines_dir):
    """Global Hansen grid keys of the cells whose centres fall inside a farm's OSM outline and inside its E4 box (as E5)."""
    from hansen_audit import outlines as O

    W = json.load(open(os.path.join(e4_out_dir, "windows.json")))
    boxes = dict(zip(W["farms"], np.array(W["boxes"])))
    els = {O.farm_key(e): e for e in json.load(open(os.path.join(outlines_dir, "overpass.json")))["elements"]}
    out = set()
    for w in W["windows"]:
        origin, shape = np.array(w["origin"]), tuple(w["shape"])
        for k in w["farms"]:
            poly = O.element_polygon(els[k]) if k in els else None
            if poly is None:
                continue
            r, c = np.nonzero(O.outline_cells(origin, shape, poly) & E.farm_cells(origin, shape, boxes[k]))
            lon = origin[0] + (c + 0.5) * HANSEN_RES
            lat = origin[1] - (r + 0.5) * HANSEN_RES
            out.update(zip(np.floor(lon / HANSEN_RES).astype(int).tolist(), np.floor(lat / HANSEN_RES).astype(int).tolist()))
    return out


def cmd_census(a):
    text = open(a.protocol).read()
    if "Status: frozen" not in text.splitlines():
        sys.exit(f"{a.protocol} is not frozen (needs a line 'Status: frozen'); nothing written")
    cells = population(a.data_root, a.e4_out_dir, a.preds)
    inside = outline_set(a.e4_out_dir, a.outlines_dir)
    keys = sorted(g for g in cells if g in inside and cells[g]["stratum"] in LOSS_DOMAIN)
    lon = np.array([cells[g]["lon"] for g in keys])
    lat = np.array([cells[g]["lat"] for g in keys])
    rows = [{"cell": f"{g[0]}_{g[1]}", "stratum": cells[g]["stratum"], "hansen": cells[g]["hansen"], "pred": cells[g]["pred"],
             "window": cells[g]["window"], "farms": ";".join(sorted(cells[g]["farms"]))} for g in keys]
    n_b = write_pair(a.out_dir, lon, lat, rows, a.seed, a.second_frac)
    N_h = {h: sum(r["stratum"] == h for r in rows) for h in LOSS_DOMAIN}
    json.dump({"census": "loss cells inside the OSM outlines", "N_h": N_h, "n": len(rows), "seed": a.seed,
               "second_frac": a.second_frac, "n_second_pass": n_b, "protocol": a.protocol},
              open(os.path.join(a.out_dir, "design.json"), "w"), indent=1)
    print(json.dumps({"N_h": N_h, "n": len(rows), "n_second_pass": n_b}, indent=1))


def read_labels(key_csv, labels_csv):
    key = {r["sample_id"]: r for r in csv.DictReader(open(key_csv))}
    lab = {r["sample_id"]: r["label"].strip().upper() for r in csv.DictReader(open(labels_csv))}
    ids = sorted(key)
    cell = [key[i]["cell"] for i in ids]
    return ids, cell, np.array([int(key[i]["stratum"]) for i in ids]), [lab.get(i, "") for i in ids], key


def cmd_estimate(a):
    pop = json.load(open(a.population))
    N_all = {int(k): v for k, v in pop["N_h"].items()}
    _, cell, s, y, key = read_labels(a.key, a.labels)
    dom = [h for h in LOSS_DOMAIN if N_all.get(h, 0) > 0]
    out = {"loss_domain_shares": S.stratified_shares({h: N_all[h] for h in dom}, s, y, domain=dom)}
    sampled = sorted(set(s.tolist()))
    out["s2c_on_farms"] = S.map_accuracy({h: N_all[h] for h in sampled}, {h: h % 3 for h in sampled}, s, y)
    out["s2c_on_farms"]["note"] = "within the sampled strata only"
    if a.farms:
        f = json.load(open(a.farms))["per_farm"]
        counts = np.array([v["loss_pred_2020"] for v in f.values()])
        ok = np.isin(np.array(y, dtype=object), list(S.CLASSES)) & np.isin(s, LOSS_DOMAIN)
        out["plantation_driven_flags"] = S.plantation_driven_rate(counts, s[ok] % 3, np.array(y, dtype=object)[ok] == "FOREST")
    if a.key2 and a.labels2:
        _, cell2, _, y2, _ = read_labels(a.key2, a.labels2)
        first = dict(zip(cell, y))
        pairs = [(first[c], l2) for c, l2 in zip(cell2, y2) if c in first]
        out["agreement"] = S.agreement([p[0] for p in pairs], [p[1] for p in pairs])
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("population")
    for n in ("data_root", "e4_out_dir", "preds", "out_json"):
        p.add_argument(n)
    d = sub.add_parser("draw")
    for n in ("data_root", "e4_out_dir", "preds", "protocol", "out_dir"):
        d.add_argument(n)
    d.add_argument("--n-total", type=int, required=True)
    d.add_argument("--n-min", type=int, default=0)
    d.add_argument("--seed", type=int, required=True)
    d.add_argument("--domain", nargs="+", default=[str(h) for h in LOSS_DOMAIN])
    d.add_argument("--second-frac", type=float, default=1.0, help="share of the cells on the blind second sheet")
    c = sub.add_parser("census")
    for n in ("data_root", "e4_out_dir", "preds", "outlines_dir", "protocol", "out_dir"):
        c.add_argument(n)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--second-frac", type=float, default=1.0, help="share of the cells on the blind second sheet")
    e = sub.add_parser("estimate")
    for n in ("population", "key", "labels"):
        e.add_argument(n)
    e.add_argument("--farms", help="results/e4/farms.json: adds the farm-level plantation-driven share with measured rates")
    e.add_argument("--key2")
    e.add_argument("--labels2")
    a = ap.parse_args()
    {"population": cmd_population, "draw": cmd_draw, "census": cmd_census, "estimate": cmd_estimate}[a.cmd](a)


if __name__ == "__main__":
    main()
