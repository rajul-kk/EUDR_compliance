"""PROTOCOL_S2 E1-E3: land-use classifier (grouped CV), temporal check, decomposition of Hansen loss by predicted 2020 land use.

usage: python -I s2_run_landuse.py <reference_cells.npz> <grid_index.csv> <s2_dir> <out_dir> [s2|s2b|s2c] [s1_dir]
The optional last argument picks the feature set (default s2 = PROTOCOL_S2; s2b = PROTOCOL_S2B context features).
Writes <out_dir>/landuse.json (E1), <out_dir>/decompose.json (E2, E3) and <out_dir>/pred_2020.npz (per-grid predicted class map, -1 = invalid).
Breakdowns (island, label_year <= 2021) re-estimate the confusion matrix from the same subset of grids.
"""
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hansen_audit import landuse as L  # noqa: E402

N_BOOT = 2000


def load_grid(gid, label_year, ref, s2_dir, feature_set="s2", s1_dir=None, s1_missing=None):
    def comp(p):
        f = os.path.join(s2_dir, f"grid_{gid}_{p}.npz")
        if not os.path.exists(f):
            return None
        z = np.load(f)
        if feature_set == "s2c":
            f1 = os.path.join(s1_dir, f"grid_{gid}_{p}.npz")
            if os.path.exists(f1):
                z1 = np.load(f1)
                c1, ch1 = z1["cells"], tuple(z1["channels"].tolist())
            else:
                c1, ch1 = None, None
                s1_missing.append(f"{gid}_{p}")
            return L.s2c_features(z["cells"], tuple(z["channels"].tolist()), c1, ch1)
        return L.cell_features(z["cells"], tuple(z["channels"].tolist()), feature_set)

    f2020 = comp("2020")
    fref = f2020 if label_year == 2020 else comp("ref")
    if f2020 is None or fref is None:
        return None
    hansen = ref[f"hansen_{gid}"]
    assert f2020.shape[1:] == hansen.shape, (gid, f2020.shape, hansen.shape)
    return dict(f2020=f2020, fref=fref, hansen=hansen, y=L.group_index(ref[f"classid_{gid}"]),
                annotated=ref[f"cov_{gid}"].astype(np.float32) >= 0.5)


def flat(F, mask):
    return F[:, mask].T


def main(ref_npz, grid_index, s2_dir, out_dir, feature_set="s2", s1_dir=None):
    t0 = time.time()
    n_feat = len(L.FEATURE_SETS[feature_set])
    print(f"feature set {feature_set}: {n_feat} features", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    import lightgbm as lgb

    rows = {r["grid_id"]: r for r in csv.DictReader(open(grid_index))}
    ref = np.load(ref_npz)
    grids, missing, s1_missing = {}, [], []
    for gid, r in rows.items():
        g = load_grid(gid, int(r["label_year"]), ref, s2_dir, feature_set, s1_dir, s1_missing)
        if g is None:
            missing.append(gid)
        else:
            grids[gid] = g
    gids = sorted(grids)
    print(f"{len(gids)} grids with both composites; missing {missing}", flush=True)
    folds = L.assign_folds(gids, [rows[g]["island"] for g in gids])
    rng = np.random.default_rng(0)

    per = {}  # per-grid counts
    pred_maps = {}
    for k in range(L.N_FOLDS):
        train = [g for g in gids if folds[g] != k]
        test = [g for g in gids if folds[g] == k]
        X, Y = [], []
        for g in train:
            d = grids[g]
            m = d["annotated"] & np.isfinite(d["fref"][0])
            idx = np.flatnonzero(m.ravel())
            if len(idx) > L.MAX_TRAIN_PER_GRID:
                idx = rng.choice(idx, L.MAX_TRAIN_PER_GRID, replace=False)
            X.append(d["fref"].reshape(n_feat, -1)[:, idx].T)
            Y.append(d["y"].ravel()[idx])
        X, Y = np.concatenate(X), np.concatenate(Y)
        clf = lgb.LGBMClassifier(**L.LGBM_PARAMS).fit(X, Y)
        print(f"fold {k}: train {len(train)} grids / {len(Y)} cells, test {len(test)} grids ({time.time() - t0:.0f} s)", flush=True)
        for g in test:
            d = grids[g]
            tree = np.isin(d["hansen"], (1, 2))
            vref = np.isfinite(d["fref"][0])
            v20 = np.isfinite(d["f2020"][0])
            m_ref = d["annotated"] & vref
            p_ref = clf.predict(flat(d["fref"], m_ref))
            yr = d["y"][m_ref]
            tree_ref = tree[m_ref]
            p20 = np.full(d["hansen"].shape, -1, np.int8)
            if v20.any():
                p20[v20] = clf.predict(flat(d["f2020"], v20))
            pred_maps[g] = p20
            m20a = d["annotated"] & v20
            per[g] = dict(
                conf_all=L.confusion_counts(yr, p_ref),
                conf_tree=L.confusion_counts(yr[tree_ref], p_ref[tree_ref]),
                conf_2020_vs_ref=L.confusion_counts(d["y"][m20a], p20[m20a]),
                loss_pred=np.bincount(p20[(d["hansen"] == 2) & v20], minlength=3),
                stable_pred=np.bincount(p20[(d["hansen"] == 1) & v20], minlength=3),
                n_loss=int((d["hansen"] == 2).sum()), n_stable=int((d["hansen"] == 1).sum()),
                valid_2020=float(v20.mean()), valid_ref=float(vref.mean()),
            )
        del X, Y, clf

    np.savez_compressed(os.path.join(out_dir, "pred_2020.npz"), **{f"pred_{g}": m for g, m in pred_maps.items()})

    def stack(key, sel):
        return np.stack([per[g][key] for g in sel])

    early = [g for g in gids if int(rows[g]["label_year"]) <= 2021]
    islands = sorted({rows[g]["island"] for g in gids})

    # ---- E1
    e1 = {
        "feature_set": feature_set, "features": list(L.FEATURE_SETS[feature_set]),
        "grids": gids, "missing_grids": missing, "s1_missing_grid_periods": s1_missing, "folds": folds,
        "all_annotated": L.metric_bootstrap(stack("conf_all", gids), N_BOOT),
        "tree_covered": L.metric_bootstrap(stack("conf_tree", gids), N_BOOT),
        "temporal_check_le2021": {
            "grids": early,
            "at_2020": L.metric_bootstrap(stack("conf_2020_vs_ref", early), N_BOOT),
            "at_ref_same_grids": L.metric_bootstrap(stack("conf_all", early), N_BOOT),
        },
        "valid_share": {g: {"2020": per[g]["valid_2020"], "ref": per[g]["valid_ref"]} for g in gids},
    }
    ba_all = e1["all_annotated"]["balanced_accuracy"]["est"]
    ba_2020 = e1["temporal_check_le2021"]["at_2020"]["balanced_accuracy"]["est"]
    e1["transfer_rule"] = {"ba_ref_all": ba_all, "ba_2020_le2021": ba_2020, "drop": ba_all - ba_2020,
                           "unreliable": bool(ba_all - ba_2020 > 0.10)}
    rt = e1["tree_covered"]
    e1["quality_gate"] = {"recall_FOREST_tree": rt["recall_FOREST"]["est"], "recall_PLANT_tree": rt["recall_PLANT"]["est"],
                          "passed": bool(min(rt["recall_FOREST"]["est"], rt["recall_PLANT"]["est"]) >= 0.70)}
    json.dump(e1, open(os.path.join(out_dir, "landuse.json"), "w"), indent=1)

    # ---- E2, E3
    def decomp(sel, key):
        return L.cluster_bootstrap(stack("conf_tree", sel), stack(key, sel), N_BOOT)

    dec = {
        "E2_loss_by_2020_landuse": {
            "all": decomp(gids, "loss_pred"),
            "by_island": {i: decomp([g for g in gids if rows[g]["island"] == i], "loss_pred") for i in islands},
            "le2021": decomp(early, "loss_pred"),
            "valid_loss_cells_share": float(sum(per[g]["loss_pred"].sum() for g in gids) / max(sum(per[g]["n_loss"] for g in gids), 1)),
        },
        "E3_stable_forest_by_2020_landuse": {
            "all": decomp(gids, "stable_pred"),
            "by_island": {i: decomp([g for g in gids if rows[g]["island"] == i], "stable_pred") for i in islands},
            "valid_stable_cells_share": float(sum(per[g]["stable_pred"].sum() for g in gids) / max(sum(per[g]["n_stable"] for g in gids), 1)),
        },
        "transfer_unreliable": e1["transfer_rule"]["unreliable"],
        "quality_gate_passed": e1["quality_gate"]["passed"],
    }
    p = dec["E2_loss_by_2020_landuse"]["all"]["corrected"]["PLANT"]
    dec["E2_headline_allowed"] = bool(p["lo"] > 0.20 or p["hi"] < 0.20) and e1["quality_gate"]["passed"]
    json.dump(dec, open(os.path.join(out_dir, "decompose.json"), "w"), indent=1)
    print(json.dumps({"transfer_rule": e1["transfer_rule"], "quality_gate": e1["quality_gate"],
                      "E2_all": dec["E2_loss_by_2020_landuse"]["all"], "headline_allowed": dec["E2_headline_allowed"]}, indent=1))
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:7])
