"""CPU experiments per PROTOCOL.md.
usage: python -I run_experiments.py <features.npz> <reference_metrics.json|none> <out.json> [epochs] [seeds]"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hansen_audit import decisions, noise, uncertainty  # noqa: E402
from hansen_audit import model as M  # noqa: E402

feat, refjson, out = sys.argv[1], sys.argv[2], sys.argv[3]
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 15
SEEDS = int(sys.argv[5]) if len(sys.argv) > 5 else 5
z = np.load(feat, allow_pickle=True)
X, Y, V, keys, country, crop = z["X"], z["Y"], z["V"], z["keys"], z["country"], z["crop"]
P, y, f = M.make_patches(X, Y, V)
print("patches", P.shape, "loss share", y.mean(), flush=True)
R = {"protocol": "PROTOCOL.md", "epochs": EPOCHS, "seeds": SEEDS, "n_patches": int(len(y)),
     "loss_share_all": float(y.mean())}


def split_farms(farm_ids, seed=0):
    ids = np.random.default_rng(seed).permutation(farm_ids)
    n = len(ids)
    a, b = int(.6 * n), int(.8 * n)
    return ids[:a], ids[a:b], ids[b:]


def sel(farms):
    return np.isin(f, farms)


def per_farm_counts(yt, pred, ft, farms):
    tp, fp, fn = (yt == 1) & (pred == 1), (yt == 0) & (pred == 1), (yt == 1) & (pred == 0)
    return np.array([[tp[ft == k].sum(), fp[ft == k].sum(), fn[ft == k].sum()] for k in farms], float)


def f1_from_counts(c):
    tp, fp, fn = c.sum(0)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def boot_diff(ca, cb, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(ca), len(ca))
        d.append(f1_from_counts(ca[i]) - f1_from_counts(cb[i]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def balanced_acc(yt, pred):
    return float(0.5 * ((pred[yt == 1] == 1).mean() + (pred[yt == 0] == 0).mean()))


# ---------- Exp A: real noisy labels, all farms, agreement with Hansen ----------
t0 = time.time()
allf = np.unique(f)
tr, va, te = split_farms(allf)
Ptr, Pva, Pte = M.standardise(P[sel(tr)], P[sel(va)], P[sel(te)])
ytr, yva, yte = y[sel(tr)], y[sel(va)], y[sel(te)]
A = {}
probs_ce_tr = []
for name in ["ce", "gce", "sce"]:
    f1s, bas = [], []
    for s in range(SEEDS):
        m = M.train_model(Ptr, ytr, Pva, yva, name, seed=s, epochs=EPOCHS)
        pr = (M.predict_proba(m, Pte)[:, 1] > .5).astype(int)
        f1s.append(M.f1_binary(yte, pr))
        bas.append(balanced_acc(yte, pr))
        if name == "ce":
            probs_ce_tr.append(M.predict_proba(m, Ptr))
    A[name] = {"f1_loss_vs_hansen_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)),
               "bal_acc_mean": float(np.mean(bas)), "bal_acc_sd": float(np.std(bas))}
    print("A", name, A[name], time.time() - t0, flush=True)
A["anchor_T_hat_from_CE_ensemble"] = noise.anchor_point_T(np.mean(probs_ce_tr, 0)).tolist()
R["expA_real_labels"] = A
R["split_A"] = {"train_farms": len(tr), "val_farms": len(va), "test_farms": len(te), "n_test_cells": int(len(yte))}

# ---------- Exp B: semi-synthetic noise on ID oil palm ----------
idp = np.array([k for k in allf if country[k] == "ID" and crop[k] == "Oil Palm"])
tr, va, te = split_farms(idp)
Ptr, Pva, Pte = M.standardise(P[sel(tr)], P[sel(va)], P[sel(te)])
ytr0, yva0, yte0 = y[sel(tr)], y[sel(va)], y[sel(te)]
fte = f[sel(te)]
B = {"n_farms": {"train": len(tr), "val": len(va), "test": len(te)}, "n_test_cells": int(len(yte0)), "settings": {}}
for lab, (a, b) in {"mild": (0.05, 0.10), "severe": (0.15, 0.30)}.items():
    Tstar = np.array([[1 - a, a], [b, 1 - b]])
    rng = np.random.default_rng(123)
    ytr_n, yva_n = noise.inject_noise(ytr0, Tstar, rng), noise.inject_noise(yva0, Tstar, rng)
    m0 = M.train_model(Ptr, ytr_n, Pva, yva_n, "ce", seed=0, epochs=EPOCHS)
    That = noise.anchor_point_T(M.predict_proba(m0, Ptr))
    methods = {"ce": None, "gce": None, "sce": None, "forward_oracleT": Tstar, "forward_anchorT": That}
    S = {"T_star": Tstar.tolist(), "T_hat_anchor": That.tolist(), "T_hat_max_abs_err": float(np.abs(That - Tstar).max()),
         "observed_noise_rate_train": float((ytr_n != ytr0).mean()), "methods": {}}
    counts = {}
    for name, T in methods.items():
        lname = "forward" if name.startswith("forward") else name
        f1s, bas, cs = [], [], []
        for s in range(SEEDS):
            mm = M.train_model(Ptr, ytr_n, Pva, yva_n, lname, T=T, seed=s, epochs=EPOCHS)
            pred = (M.predict_proba(mm, Pte)[:, 1] > .5).astype(int)
            f1s.append(M.f1_binary(yte0, pred))
            bas.append(balanced_acc(yte0, pred))
            cs.append(per_farm_counts(yte0, pred, fte, te))
        counts[name] = np.mean(cs, 0)
        S["methods"][name] = {"f1_vs_clean_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)),
                              "bal_acc_mean": float(np.mean(bas)), "bal_acc_sd": float(np.std(bas))}
        print("B", lab, name, S["methods"][name], time.time() - t0, flush=True)
    S["paired_f1_diff_vs_ce_bootstrap_mean_lo_hi"] = {n: boot_diff(counts[n], counts["ce"]) for n in methods if n != "ce"}
    B["settings"][lab] = S
R["expB_semisynthetic_ID_oilpalm"] = B

# ---------- Exp C: farm-level decisions, ensemble + conformal (real labels, ID oil palm) ----------
tr, va, te = split_farms(idp, seed=1)
cal_n = len(te) // 2
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
     "note": "Coverage is w.r.t. the noisy Hansen count, not truth. Small n."}
R["expC_farm_decisions_ID_oilpalm"] = C
print("C", C, flush=True)

# ---------- Exp D: transported false-flag sensitivity from reference precision ----------
if refjson != "none" and os.path.exists(refjson):
    ref = json.load(open(refjson))
    hl = np.array([(y[f == k] == 1).sum() for k in idp])
    flagged = hl >= decisions.DEFAULT_MIN_CELLS
    Dd = {"id_palm_farms": int(len(idp)), "hansen_flagged_farms": int(flagged.sum()),
          "min_cells": decisions.DEFAULT_MIN_CELLS,
          "note": "Assumes reference-cell precision transfers to these farms and cells are independent (optimistic)."}
    for lab in ("strict_P_FOREST_given_loss", "lenient_1_minus_P_PLANT_given_loss"):
        for stat, idx in (("point", None), ("pessimistic_ci_edge", 0), ("optimistic_ci_edge", 1)):
            if lab.startswith("strict"):
                prec = ref["pooled"]["P(FOREST|loss)"] if idx is None else ref["ci95"]["P(FOREST|loss)"][idx]
            else:
                prec = 1 - (ref["pooled"]["P(PLANT|loss)"] if idx is None else ref["ci95"]["P(PLANT|loss)"][1 - idx])
            pf = np.array([decisions.expected_false_flag_prob(int(n), prec) for n in hl[flagged]])
            Dd[f"{lab}__{stat}"] = {"precision": float(prec),
                                    "expected_false_flag_share_of_flagged_farms": float(pf.mean()) if len(pf) else None}
    R["expD_transported_false_flag_sensitivity"] = Dd
    print("D", Dd, flush=True)
json.dump(R, open(out, "w"), indent=1, default=float)
print("saved", out, time.time() - t0)
