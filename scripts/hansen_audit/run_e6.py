"""E6 per PROTOCOL_E6.md: experiment C on the full set of ID oil palm farms, leave-one-group-out transfer, image type.
usage: python -I run_e6.py <e6_features.npz> <out.json> [epochs] [seeds]"""
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hansen_audit import model as M  # noqa: E402
from hansen_audit.experiments import farm_decisions  # noqa: E402
from hansen_audit.io import farms_in_group, write_json  # noqa: E402
from hansen_audit.metrics import (  # noqa: E402
    balanced_accuracy,
    boot_f1_diff,
    boot_f1_diff_unpaired,
    f1,
    per_farm_counts,
)

MIN_GROUP = 20
MIN_TYPE = 10
GAP = 0.05
N_BOOT = 2000


def ensemble_prob(P, y, f, train_farms, test_mask, seeds, epochs, seed=0):
    """Train on train_farms (20 % of them held out for early stopping), mean probability over seeds on P[test_mask]."""
    ids = np.random.default_rng(seed).permutation(train_farms)
    nv = max(1, int(.2 * len(ids)))
    va, tr = ids[:nv], ids[nv:]
    mtr, mva = np.isin(f, tr), np.isin(f, va)
    Ptr, Pva, Pte = M.standardise(P[mtr], P[mva], P[test_mask])
    return np.mean([M.predict_proba(M.train_model(Ptr, y[mtr], Pva, y[mva], "ce", seed=s, epochs=epochs), Pte)[:, 1]
                    for s in range(seeds)], 0)


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

    idp = np.intersect1d(farms_in_group(z), farms)
    C, test_farms, hflag, mflag = farm_decisions(P, y, f, idp, SEEDS, EPOCHS)
    C["note"] = "Coverage is w.r.t. the noisy Hansen count, not truth."
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
        ci, co = per_farm_counts(y[gm], pred_in[gm], f[gm], gf), per_farm_counts(y[gm], pred_out, f[gm], gf)
        gap = boot_f1_diff(ci, co, N_BOOT)
        T[g] = {"n_farms": int(len(gf)), "n_cells": int(gm.sum()), "loss_share": float(y[gm].mean()),
                "f1_in_group": f1(ci), "f1_held_out": f1(co),
                "bal_acc_in_group": balanced_accuracy(y[gm], pred_in[gm]), "bal_acc_held_out": balanced_accuracy(y[gm], pred_out),
                "f1_gap_mean_lo_hi": gap,
                "reading": "does not transfer" if (f1(ci) - f1(co) >= GAP and gap[1] > 0) else "transfers"}
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
            cs[p] = per_farm_counts(y[m], pred_in[m], f[m], fp)
            res[p] = {"n_farms": int(len(fp)), "n_cells": int(m.sum()), "loss_share": float(y[m].mean()),
                      "f1_in_group": f1(cs[p]), "bal_acc_in_group": balanced_accuracy(y[m], pred_in[m])}
        res["f1_scene_minus_composite_mean_lo_hi"] = boot_f1_diff_unpaired(cs["scene"], cs["composite"], N_BOOT)
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
