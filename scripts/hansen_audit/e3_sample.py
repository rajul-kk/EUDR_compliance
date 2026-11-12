"""E3: reference sample on the farms. Population counts, the draw, and estimates from returned labels.

usage:
  python -I e3_sample.py population <data_root> <e4_out_dir> <farm_preds.npz> <out_json>
  python -I e3_sample.py draw <data_root> <e4_out_dir> <farm_preds.npz> <protocol_md> <out_dir> --n-total N --n-min M --seed S [--domain 6 7 8]
  python -I e3_sample.py estimate <population_json> <key_csv> <labels_csv> [--farms farms.json] [--key2 <key_csv> --labels2 <labels_csv>]
population: every cell centre in any of the 94 farm boxes, counted once (cells shared by overlapping boxes or windows are
            merged on the global Hansen grid), with its stratum (Hansen class x S2C 2020 prediction). Writes N_h and sample
            sizes for a range of interval half-widths. Draws nothing.
draw:       refuses unless <protocol_md> contains the line "Status: frozen". Writes sheet_A.csv / key_A.csv and a second blind
            pass sheet_B.csv / key_B.csv over the same cells in a different order with different ids.
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
    os.makedirs(a.out_dir, exist_ok=True)
    S.write_sheets(os.path.join(a.out_dir, "sheet_A.csv"), os.path.join(a.out_dir, "key_A.csv"), lon, lat, rows, a.seed + 1, "A")
    S.write_sheets(os.path.join(a.out_dir, "sheet_B.csv"), os.path.join(a.out_dir, "key_B.csv"), lon, lat, rows, a.seed + 2, "B")
    json.dump({"N_h": N_h, "n_h": n_h, "seed": a.seed, "protocol": a.protocol}, open(os.path.join(a.out_dir, "design.json"), "w"), indent=1)
    print(json.dumps({"N_h": N_h, "n_h": n_h}, indent=1))


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
    e = sub.add_parser("estimate")
    for n in ("population", "key", "labels"):
        e.add_argument(n)
    e.add_argument("--farms", help="results/e4/farms.json: adds the farm-level plantation-driven share with measured rates")
    e.add_argument("--key2")
    e.add_argument("--labels2")
    a = ap.parse_args()
    {"population": cmd_population, "draw": cmd_draw, "estimate": cmd_estimate}[a.cmd](a)


if __name__ == "__main__":
    main()
