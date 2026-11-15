"""PROTOCOL_E8 analysis 3: E4/E5 and E8 pooled over all 200 ID oil palm farms (descriptive).

usage: python -I e8_pooled.py <e4_results> <e5_results> <e8_box> <e8_outline> <e5_outlines_dir> <e8_outlines_dir> <out.json>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hansen_audit import farms_e4 as E  # noqa: E402
from hansen_audit import outlines as O  # noqa: E402


def load(d):
    return json.load(open(os.path.join(d, "farms.json")))


def plant_share(rows):
    s = np.array(rows).reshape(-1, 3).sum(0)
    return {"cells": int(s.sum()), "share_PLANT": float(s[1] / s.sum()) if s.sum() else None}


MIN_STABLE_CELLS = 100  # PROTOCOL_E8 analysis 2: fewer stable-forest cells inside outlines -> no reading


def transport_reading(share):
    """PROTOCOL_E5 reading of the PLANT share of stable-forest cells, with PROTOCOL_E8's minimum cell count."""
    if share["cells"] < MIN_STABLE_CELLS:
        return f"too few cells ({share['cells']} < {MIN_STABLE_CELLS})"
    s = share["share_PLANT"]
    return ("boxes included forest outside the plantation" if s >= 0.40 else
            "classifier under-calls palm on the farms" if s <= 0.20 else "inconclusive")


def polygons(d, keys):
    els = {O.farm_key(e): e for e in json.load(open(os.path.join(d, "overpass.json")))["elements"]}
    return {k: poly for k in keys if k in els and (poly := O.element_polygon(els[k])) is not None}


def main(e4, e5, e8b, e8o, out5, out8, out):
    b4, b8 = load(e4)["per_farm"], load(e8b)["per_farm"]
    assert not set(b4) & set(b8), "farm sets must be disjoint"
    box = {**b4, **b8}
    o5, o8 = load(e5)["per_farm"], load(e8o)["per_farm"]
    assert not set(o5) & set(o8), "farm sets must be disjoint"
    outl = {**o5, **o8}
    kb, ko = sorted(box), sorted(outl)
    p5, p8 = polygons(out5, o5), polygons(out8, o8)
    pairs = [(a, b, float(O.equal_area(x).area / 1e4)) for a, pa in p5.items() for b, pb in p8.items()
             if pa.intersects(pb) and (x := pa.intersection(pb)).area > 0]
    R = {"protocol": "PROTOCOL_E8.md, analysis 3 (descriptive)",
         "box": {"n": len(kb), "summary": E.farm_summary(np.array([box[k]["loss_pred_2020"] for k in kb])),
                 "transport_check": plant_share([box[k]["stable_pred_2020"] for k in kb])},
         "outline": {"n": len(ko), "summary": E.farm_summary(np.array([outl[k]["loss_pred_2020"] for k in ko])),
                     "transport_check": plant_share([outl[k]["stable_pred_2020"] for k in ko])},
         "transport_reading_e8_outline": transport_reading(plant_share([o8[k]["stable_pred_2020"] for k in o8])),
         "transport_reading_pooled_outline": transport_reading(plant_share([outl[k]["stable_pred_2020"] for k in ko])),
         "outline_overlap_e5_e8": {"pairs": len(pairs), "hectares": round(sum(p[2] for p in pairs), 3),
                                   "list": [list(p) for p in pairs]},
         "caveat": "transported S2C predictions; farm bootstrap intervals are optimistic where boxes overlap"}
    json.dump(R, open(out, "w"), indent=1)
    print(json.dumps({k: R[k] for k in ("box", "outline", "transport_reading_e8_outline", "transport_reading_pooled_outline")},
                     indent=1), R["outline_overlap_e5_e8"]["pairs"])


if __name__ == "__main__":
    main(*sys.argv[1:8])
