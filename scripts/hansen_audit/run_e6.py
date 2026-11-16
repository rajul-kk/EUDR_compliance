"""E6 per PROTOCOL_E6.md: experiment C on the full set of ID oil palm farms, leave-one-group-out transfer, image type.
usage: python -I run_e6.py <e6_features.npz> <out.json> [epochs] [seeds]"""
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hansen_audit import decisions, uncertainty  # noqa: E402
from hansen_audit import model as M  # noqa: E402
from hansen_audit.io import write_json  # noqa: E402

MIN_GROUP = 20
MIN_TYPE = 10
GAP = 0.05
N_BOOT = 2000


def split_farms(farm_ids, seed=0):
    ids = np.random.default_rng(seed).permutation(farm_ids)
    n = len(ids)
    a, b = int(.6 * n), int(.8 * n)
    return ids[:a], ids[a:b], ids[b:]


def counts(yt, pred):
    return np.array([((yt == 1) & (pred == 1)).sum(), ((yt == 0) & (pred == 1)).sum(), ((yt == 1) & (pred == 0)).sum()],
                    float)


def f1(c):
    tp, fp, fn = c
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def bal_acc(yt, pred):
    return float(0.5 * ((pred[yt == 1] == 1).mean() + (pred[yt == 0] == 0).mean()))


def farm_counts(yt, pred, f, farms):
    return np.array([counts(yt[f == k], pred[f == k]) for k in farms])


def boot_gap(ca, cb, seed=0):
    """F1(ca) - F1(cb) over resamples of farms (rows), pooled counts per resample: mean, 2.5 and 97.5 percentiles."""
    rng = np.random.default_rng(seed)
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        j = rng.integers(0, len(ca), len(ca))
        d[i] = f1(ca[j].sum(0)) - f1(cb[j].sum(0))
    return [float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def boot_diff_groups(ca, cb, seed=0):
    """F1(ca) - F1(cb) for two disjoint sets of farms, each resampled on its own."""
    rng = np.random.default_rng(seed)
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        d[i] = (f1(ca[rng.integers(0, len(ca), len(ca))].sum(0)) - f1(cb[rng.integers(0, len(cb), len(cb))].sum(0)))
    return [float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def ensemble_prob(P, y, f, train_farms, test_mask, seeds, epochs, seed=0):
    """Train on train_farms (20 % of them held out for early stopping), mean probability over seeds on P[test_mask]."""
    ids = np.random.default_rng(seed).permutation(train_farms)
    nv = max(1, int(.2 * len(ids)))
    va, tr = ids[:nv], ids[nv:]
    mtr, mva = np.isin(f, tr), np.isin(f, va)
    Ptr, Pva, Pte = M.standardise(P[mtr], P[mva], P[test_mask])
    return np.mean([M.predict_proba(M.train_model(Ptr, y[mtr], Pva, y[mva], "ce", seed=s, epochs=epochs), Pte)[:, 1]
                    for s in range(seeds)], 0)


def exp_c(P, y, f, idp, SEEDS, EPOCHS):
    # verbatim from run_experiments.py, block "Exp C"
    tr, va, te = split_farms(idp, seed=1)
    cal_n = len(te) // 2

    def sel(farms):
        return np.isin(f, farms)

    mu = P[sel(tr)].mean((0, 2, 3), keepdims=True)
    sd = P[sel(tr)].std((0, 2, 3), keepdims=True) + 1e-6
    Ptr, Pva = (P[sel(tr)] - mu) / sd, (P[sel(va)] - mu) / sd
    members = [M.train_model(Ptr, y[sel(tr)], Pva, y[sel(va)], "ce", seed=s, epochs=EPOCHS) for s in range(SEEDS)]
    area = np.zeros((SEEDS, len(te)))
    hansen = np.zeros(len(te))
    for j, k in enumerate(te):
        mk = f == k
        Pk = (P[mk] - mu) / sd
        hansen[j] = (y[mk] == 1).sum()
        for s, mm in enumerate(members):
            area[s, j] = (M.predict_proba(mm, Pk)[:, 1] > .5).sum()
    summ = uncertainty.ensemble_summary(area, decisions.DEFAULT_MIN_CELLS)
    cal, tst = np.arange(len(te)) < cal_n, np.arange(len(te)) >= cal_n
    hw = uncertainty.split_conformal_halfwidth(np.abs(summ["mean"][cal] - hansen[cal]), 0.1)
    dec = uncertainty.decision_with_interval(summ["mean"][tst], hw, decisions.DEFAULT_MIN_CELLS)
    hflag = decisions.farm_flag(hansen[tst])
    mflag = decisions.farm_flag(summ["mean"][tst])
    C = {"n_farms_eval": int(len(te)), "n_cal": int(cal.sum()), "n_test": int(tst.sum()),
         "conformal_halfwidth_cells": hw, "hansen_flag_rate_test": float(hflag.mean()),
         "model_flag_rate_test": float(mflag.mean()), "agreement": float((hflag == mflag).mean()),
         "uncertain_share": float((dec == 2).mean()), "decision_counts_0_1_2": np.bincount(dec, minlength=3).tolist(),
         "mean_ensemble_std_cells": float(summ["std"][tst].mean()),
         "note": "Coverage is w.r.t. the noisy Hansen count, not truth."}
    return C, te[tst], hflag, mflag


def main(feat, out, EPOCHS=15, SEEDS=5):
    t0 = time.time()
    z = np.load(feat, allow_pickle=True)
    P, y, fi = M.make_patches(z["X"], z["Y"], z["V"])
    keys, country, crop, product = z["keys"], z["country"], z["crop"], z["product"]
    f = fi
    farms = np.unique(f)
    group = np.array([f"{country[i]}|{crop[i]}" for i in range(len(keys))])
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    R = {"protocol": "PROTOCOL_E6.md", "command": " ".join(["python -I scripts/hansen_audit/run_e6.py", *sys.argv[1:]]),
         "git": commit, "epochs": EPOCHS, "seeds": SEEDS, "n_farms": int(len(farms)), "n_cells": int(len(y)),
         "farms_by_product": {p: int((product[farms] == p).sum()) for p in ("scene", "composite")}}

    idp = np.array([k for k in farms if country[k] == "ID" and crop[k] == "Oil Palm"])
    C, test_farms, hflag, mflag = exp_c(P, y, f, idp, SEEDS, EPOCHS)
    C["n_id_oil_palm_farms"] = int(len(idp))
    C["agreement_by_product"] = {p: {"n": int((product[test_farms] == p).sum()),
                                     "agreement": float((hflag == mflag)[product[test_farms] == p].mean())
                                     if (product[test_farms] == p).any() else None}
                                 for p in ("scene", "composite")}
    R["exp_c"] = C
    print("C", json.dumps(C), time.time() - t0, flush=True)

    folds = np.array_split(np.random.default_rng(0).permutation(farms), 5)
    p_in = np.full(len(y), np.nan)
    for i, held in enumerate(folds):
        m = np.isin(f, held)
        p_in[m] = ensemble_prob(P, y, f, np.setdiff1d(farms, held), m, SEEDS, EPOCHS)
        print("fold", i, time.time() - t0, flush=True)
    pred_in = (p_in > .5).astype(int)

    sizes = {g: int(len(np.unique(f[group[f] == g]))) for g in np.unique(group[farms])}
    T = {}
    for g in sorted((g for g, n in sizes.items() if n >= MIN_GROUP), key=lambda g: -sizes[g]):
        gm = group[f] == g
        gf = np.unique(f[gm])
        pred_out = (ensemble_prob(P, y, f, np.setdiff1d(farms, gf), gm, SEEDS, EPOCHS) > .5).astype(int)
        ci, co = farm_counts(y[gm], pred_in[gm], f[gm], gf), farm_counts(y[gm], pred_out, f[gm], gf)
        gap = boot_gap(ci, co)
        T[g] = {"n_farms": int(len(gf)), "n_cells": int(gm.sum()), "loss_share": float(y[gm].mean()),
                "f1_in_group": f1(ci.sum(0)), "f1_held_out": f1(co.sum(0)),
                "bal_acc_in_group": bal_acc(y[gm], pred_in[gm]), "bal_acc_held_out": bal_acc(y[gm], pred_out),
                "f1_gap_mean_lo_hi": gap,
                "reading": "does not transfer" if (f1(ci.sum(0)) - f1(co.sum(0)) >= GAP and gap[1] > 0) else "transfers"}
        print("T", g, json.dumps(T[g]), time.time() - t0, flush=True)
    R["transfer"] = T
    R["group_sizes"] = sizes

    IT = {}
    for g in sorted(sizes):
        gf = np.unique(f[group[f] == g])
        by = {p: gf[product[gf] == p] for p in ("scene", "composite")}
        if min(len(v) for v in by.values()) < MIN_TYPE:
            continue
        res = {}
        cs = {}
        for p, fp in by.items():
            m = np.isin(f, fp)
            cs[p] = farm_counts(y[m], pred_in[m], f[m], fp)
            res[p] = {"n_farms": int(len(fp)), "n_cells": int(m.sum()), "loss_share": float(y[m].mean()),
                      "f1_in_group": f1(cs[p].sum(0)), "bal_acc_in_group": bal_acc(y[m], pred_in[m])}
        res["f1_scene_minus_composite_mean_lo_hi"] = boot_diff_groups(cs["scene"], cs["composite"])
        IT[g] = res
        print("I", g, json.dumps(res), flush=True)
    R["image_type"] = IT
    R["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    write_json(R, out, indent=1)
    print("saved", out, R["seconds"])


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(int(x) for x in a[2:4]))
