"""Transported false-flag sensitivity for Indonesian oil palm farms (replaces experiment D, see PROTOCOL deviations).

For each Hansen-flagged farm (>= min_cells loss cells), P(fewer than min_cells of its loss cells are true deforestation)
given a per-cell precision p of Hansen loss. p is swept; the reference-derived values (1 - P(PLANT | loss) from the
grids with reference year <= 2021, and from all grids) are marked. Cells independent (optimistic).
usage: python -I run_sensitivity.py <features.npz> <reference_metrics.json> <out.json>"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hansen_audit import decisions  # noqa: E402
from hansen_audit import model as M  # noqa: E402

z = np.load(sys.argv[1], allow_pickle=True)
ref = json.load(open(sys.argv[2]))
P, y, f = M.make_patches(z["X"], z["Y"], z["V"])
farms = np.where((z["country"] == "ID") & (z["crop"] == "Oil Palm"))[0]
loss = np.array([(y[f == k] == 1).sum() for k in farms])
res = {"id_palm_farms_with_patches": int(len(farms)), "loss_cells_per_farm_quantiles_0_25_50_75_100": np.percentile(loss, [0, 25, 50, 75, 100]).tolist()}
rng = np.random.default_rng(0)
out = {}
for min_cells in (1, 3, 6, 12):
    flagged = loss >= min_cells
    row = {"hansen_flagged_farms": int(flagged.sum()), "flag_rate": float(flagged.mean()), "sweep": {}}
    for p in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        pf = np.array([decisions.expected_false_flag_prob(int(n), p, min_cells) for n in loss[flagged]])
        row["sweep"][str(p)] = float(pf.mean()) if len(pf) else None
    out[str(min_cells)] = row
res["by_min_cells"] = out
anchors = {}
for lab, d in (("all_grids", ref["pooled"]), ("ref_year_le_2021", ref["by_label_year"]["<=2021"]["pooled"])):
    ci = ref["ci95"] if lab == "all_grids" else ref["by_label_year"]["<=2021"]["ci95"]
    p = 1 - d["P(PLANT|loss)"]
    lo, hi = 1 - ci["P(PLANT|loss)"][1], 1 - ci["P(PLANT|loss)"][0]
    a = {"precision_point": p, "precision_ci95": [lo, hi]}
    for min_cells in (6,):
        flagged = loss >= min_cells
        a["expected_false_flag_share_of_flagged_min6"] = {
            s: float(np.mean([decisions.expected_false_flag_prob(int(n), q, min_cells) for n in loss[flagged]]))
            for s, q in (("point", p), ("pessimistic_lo", lo), ("optimistic_hi", hi))}
        # farm bootstrap on point estimate
        fl = loss[flagged]
        bs = [np.mean([decisions.expected_false_flag_prob(int(n), p, min_cells) for n in rng.choice(fl, len(fl))]) for _ in range(500)]
        a["farm_bootstrap_95_point"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    anchors[lab] = a
res["reference_anchored"] = anchors
res["caveat"] = ("precision = 1 - P(reference PLANT | Hansen loss); counts OTHER (shrub, bare, built...) as true loss, so it is an optimistic "
                 "precision. Reference grids are outside the farms; transport is an assumption.")
json.dump(res, open(sys.argv[3], "w"), indent=1)
print(json.dumps(res, indent=1))
