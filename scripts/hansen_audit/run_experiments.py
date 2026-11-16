"""CPU experiments per PROTOCOL.md.
usage: python -I run_experiments.py <features.npz> <reference_metrics.json|none> <out.json> [epochs] [seeds]"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hansen_audit import decisions, noise  # noqa: E402
from hansen_audit import model as M  # noqa: E402
from hansen_audit.experiments import farm_decisions  # noqa: E402
from hansen_audit.io import farms_in_group, read_json, write_json  # noqa: E402
from hansen_audit.metrics import (  # noqa: E402
    balanced_accuracy,
    boot_f1_diff,
    f1,
    per_farm_counts,
    split_farms,
    tp_fp_fn,
)

NOISE = {"mild": (0.05, 0.10), "severe": (0.15, 0.30)}


def exp_a(P, y, f, farms, seeds, epochs, t0):
    """Real noisy labels, all farms: agreement with Hansen on held-out farms."""
    tr, va, te = split_farms(farms)
    mtr, mva, mte = np.isin(f, tr), np.isin(f, va), np.isin(f, te)
    Ptr, Pva, Pte = M.standardise(P[mtr], P[mva], P[mte])
    ytr, yva, yte = y[mtr], y[mva], y[mte]
    A, probs_ce_tr = {}, []
    for name in ["ce", "gce", "sce"]:
        f1s, bas = [], []
        for s in range(seeds):
            m = M.train_model(Ptr, ytr, Pva, yva, name, seed=s, epochs=epochs)
            pr = (M.predict_proba(m, Pte)[:, 1] > .5).astype(int)
            f1s.append(f1(tp_fp_fn(yte, pr)))
            bas.append(balanced_accuracy(yte, pr))
            if name == "ce":
                probs_ce_tr.append(M.predict_proba(m, Ptr))
        A[name] = {"f1_loss_vs_hansen_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)),
                   "bal_acc_mean": float(np.mean(bas)), "bal_acc_sd": float(np.std(bas))}
        print("A", name, A[name], time.time() - t0, flush=True)
    A["anchor_T_hat_from_CE_ensemble"] = noise.anchor_point_T(np.mean(probs_ce_tr, 0)).tolist()
    split = {"train_farms": len(tr), "val_farms": len(va), "test_farms": len(te), "n_test_cells": int(len(yte))}
    return A, split


def exp_b(P, y, f, idp, seeds, epochs, t0):
    """Semi-synthetic noise on ID oil palm: robust losses and forward correction against CE, scored on clean labels."""
    tr, va, te = split_farms(idp)
    mtr, mva, mte = np.isin(f, tr), np.isin(f, va), np.isin(f, te)
    Ptr, Pva, Pte = M.standardise(P[mtr], P[mva], P[mte])
    ytr0, yva0, yte0, fte = y[mtr], y[mva], y[mte], f[mte]
    B = {"n_farms": {"train": len(tr), "val": len(va), "test": len(te)}, "n_test_cells": int(len(yte0)), "settings": {}}
    for lab, (a, b) in NOISE.items():
        Tstar = np.array([[1 - a, a], [b, 1 - b]])
        rng = np.random.default_rng(123)
        ytr_n, yva_n = noise.inject_noise(ytr0, Tstar, rng), noise.inject_noise(yva0, Tstar, rng)
        m0 = M.train_model(Ptr, ytr_n, Pva, yva_n, "ce", seed=0, epochs=epochs)
        That = noise.anchor_point_T(M.predict_proba(m0, Ptr))
        methods = {"ce": None, "gce": None, "sce": None, "forward_oracleT": Tstar, "forward_anchorT": That}
        S = {"T_star": Tstar.tolist(), "T_hat_anchor": That.tolist(), "T_hat_max_abs_err": float(np.abs(That - Tstar).max()),
             "observed_noise_rate_train": float((ytr_n != ytr0).mean()), "methods": {}}
        counts = {}
        for name, T in methods.items():
            lname = "forward" if name.startswith("forward") else name
            f1s, bas, cs = [], [], []
            for s in range(seeds):
                mm = M.train_model(Ptr, ytr_n, Pva, yva_n, lname, T=T, seed=s, epochs=epochs)
                pred = (M.predict_proba(mm, Pte)[:, 1] > .5).astype(int)
                f1s.append(f1(tp_fp_fn(yte0, pred)))
                bas.append(balanced_accuracy(yte0, pred))
                cs.append(per_farm_counts(yte0, pred, fte, te))
            counts[name] = np.mean(cs, 0)
            S["methods"][name] = {"f1_vs_clean_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)),
                                  "bal_acc_mean": float(np.mean(bas)), "bal_acc_sd": float(np.std(bas))}
            print("B", lab, name, S["methods"][name], time.time() - t0, flush=True)
        S["paired_f1_diff_vs_ce_bootstrap_mean_lo_hi"] = {n: boot_f1_diff(counts[n], counts["ce"])
                                                          for n in methods if n != "ce"}
        B["settings"][lab] = S
    return B


def exp_d(ref, y, f, idp):
    """Transported false-flag sensitivity from the reference precision (strict and lenient, point and interval edges)."""
    hl = np.array([(y[f == k] == 1).sum() for k in idp])
    flagged = hl >= decisions.DEFAULT_MIN_CELLS
    D = {"id_palm_farms": int(len(idp)), "hansen_flagged_farms": int(flagged.sum()),
         "min_cells": decisions.DEFAULT_MIN_CELLS,
         "note": "Assumes reference-cell precision transfers to these farms and cells are independent (optimistic)."}
    for lab in ("strict_P_FOREST_given_loss", "lenient_1_minus_P_PLANT_given_loss"):
        for stat, idx in (("point", None), ("pessimistic_ci_edge", 0), ("optimistic_ci_edge", 1)):
            if lab.startswith("strict"):
                prec = ref["pooled"]["P(FOREST|loss)"] if idx is None else ref["ci95"]["P(FOREST|loss)"][idx]
            else:
                prec = 1 - (ref["pooled"]["P(PLANT|loss)"] if idx is None else ref["ci95"]["P(PLANT|loss)"][1 - idx])
            pf = np.array([decisions.expected_false_flag_prob(int(n), prec) for n in hl[flagged]])
            D[f"{lab}__{stat}"] = {"precision": float(prec),
                                   "expected_false_flag_share_of_flagged_farms": float(pf.mean()) if len(pf) else None}
    return D


def main(feat, refjson, out, epochs=15, seeds=5):
    z = np.load(feat, allow_pickle=True)
    P, y, f = M.make_patches(z["X"], z["Y"], z["V"])
    print("patches", P.shape, "loss share", y.mean(), flush=True)
    R = {"protocol": "PROTOCOL.md", "epochs": epochs, "seeds": seeds, "n_patches": int(len(y)),
         "loss_share_all": float(y.mean())}
    t0 = time.time()
    farms = np.unique(f)
    R["expA_real_labels"], R["split_A"] = exp_a(P, y, f, farms, seeds, epochs, t0)
    idp = np.intersect1d(farms_in_group(z), farms)
    R["expB_semisynthetic_ID_oilpalm"] = exp_b(P, y, f, idp, seeds, epochs, t0)
    C = farm_decisions(P, y, f, idp, seeds, epochs)[0]
    C["note"] = "Coverage is w.r.t. the noisy Hansen count, not truth. Small n."
    R["expC_farm_decisions_ID_oilpalm"] = C
    print("C", C, flush=True)
    if refjson != "none" and os.path.exists(refjson):
        R["expD_transported_false_flag_sensitivity"] = exp_d(read_json(refjson), y, f, idp)
        print("D", R["expD_transported_false_flag_sensitivity"], flush=True)
    write_json(R, out, indent=1, default=float)
    print("saved", out, time.time() - t0)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], *(int(x) for x in a[3:5]))
