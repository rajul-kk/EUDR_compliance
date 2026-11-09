"""Reference-vs-Hansen confusion on the Zenodo grids, cluster bootstrap over grids.
usage: python -I analyse_reference.py <reference_cells.npz> <grid_index.csv> <out.json>"""
import json, sys
import numpy as np, pandas as pd

GROUPS = {1: "FOREST", 2: "PLANT", 3: "OTHER"}


def class_group_arr(cid):
    g = np.zeros(cid.shape, np.uint8)
    g[np.isin(cid, [9, 12])] = 1
    g[np.isin(cid, [1, 2, 4, 5, 6, 7, 8])] = 2
    g[(g == 0) & (cid >= 0)] = 3
    return g


def table(h, g):
    t = np.zeros((3, 3), np.int64)  # rows hansen 0,1,2 ; cols FOREST,PLANT,OTHER
    for i in range(3):
        for j in range(3):
            t[i, j] = ((h == i) & (g == j + 1)).sum()
    return t


def metrics(t):
    def r(n, d):
        return float(n / d) if d > 0 else float("nan")
    return {
        "loss_cells": int(t[2].sum()),
        "P(PLANT|loss)": r(t[2, 1], t[2].sum()),
        "P(FOREST|loss)": r(t[2, 0], t[2].sum()),
        "P(OTHER|loss)": r(t[2, 2], t[2].sum()),
        "stable_cells": int(t[1].sum()),
        "P(PLANT|stable_forest)": r(t[1, 1], t[1].sum()),
        "P(FOREST|stable_forest)": r(t[1, 0], t[1].sum()),
        "P(hansen>=1|PLANT)": r(t[1:, 1].sum(), t[:, 1].sum()),
        "P(loss|PLANT)": r(t[2, 1], t[:, 1].sum()),
        "P(hansen=0|FOREST)": r(t[0, 0], t[:, 0].sum()),
        "forest_ref_cells": int(t[:, 0].sum()), "plant_ref_cells": int(t[:, 1].sum()),
    }


def main(npz, gi, out, n_boot=2000):
    z = np.load(npz)
    gi = pd.read_csv(gi, dtype={"grid_id": str}).set_index("grid_id")
    per = {}
    for k in z.files:
        if not k.startswith("hansen_"):
            continue
        gid = k[7:]
        h, cid, cov = z[k], z["classid_" + gid], z["cov_" + gid].astype(np.float32)
        cid = np.where(cov >= 0.5, cid, -1)
        n = min(h.shape[0], cid.shape[0]); m = min(h.shape[1], cid.shape[1])
        per[gid] = table(h[:n, :m], class_group_arr(cid[:n, :m]))
    gids = sorted(per)
    res = {"n_grids": len(gids), "pooled": metrics(sum(per.values()))}
    rng = np.random.default_rng(0)
    keys = list(res["pooled"].keys())
    boots = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, len(gids), len(gids))
        mm = metrics(sum(per[gids[i]] for i in pick))
        for k in keys:
            boots[k].append(mm[k])
    res["ci95"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in boots.items()}
    res["by_island"] = {isl: metrics(sum(per[g] for g in gids if gi.loc[g, "island"] == isl)) for isl in sorted(gi.island.unique())}
    res["by_label_year"] = {}
    for lab, cond in (("<=2021", lambda y: y <= 2021), (">=2022", lambda y: y >= 2022)):
        gg = [g for g in gids if cond(int(gi.loc[g, "label_year"]))]
        tt = sum(per[g] for g in gg)
        bs = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(gg), len(gg)); bs.append(metrics(sum(per[gg[i]] for i in pick)))
        m = metrics(tt)
        res["by_label_year"][lab] = {"n_grids": len(gg), "pooled": m,
            "ci95": {k: [float(np.nanpercentile([b[k] for b in bs], 2.5)), float(np.nanpercentile([b[k] for b in bs], 97.5))] for k in m}}
    res["by_grid"] = {g: {"table_hansen_rows_ref_cols_FPO": per[g].tolist(), "label_year": int(gi.loc[g, "label_year"]), "province": gi.loc[g, "province"]} for g in gids}
    # grids with >0 loss cells: share where PLANT is the majority reference
    lc = [(g, per[g][2]) for g in gids if per[g][2].sum() > 0]
    res["grids_with_loss"] = len(lc)
    res["grids_loss_majority_plant"] = int(sum(1 for g, t in lc if t[1] > t[0] and t[1] > t[2]))
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_grids", "pooled", "ci95", "by_island", "grids_with_loss", "grids_loss_majority_plant")}, indent=1))

if __name__ == "__main__":
    main(*sys.argv[1:4])
