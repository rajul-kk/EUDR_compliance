"""Experiment C (PROTOCOL.md; rerun on all farms in PROTOCOL_E6): farm-level decisions from a seed ensemble with a
split-conformal interval, scored against the Hansen count."""
import numpy as np

from hansen_audit import decisions, uncertainty
from hansen_audit import model as M
from hansen_audit.metrics import split_farms


def farm_decisions(P, y, f, farm_ids, seeds, epochs):
    """Train `seeds` CE models on 60 % of `farm_ids` (split seed 1), predict each held-out farm's loss area, calibrate the
    conformal half-width on the first half of the test farms and decide on the second half.
    Returns (result dict without a note, decided farm ids, Hansen flags, model flags)."""
    tr, va, te = split_farms(farm_ids, seed=1)
    cal_n = len(te) // 2
    mtr, mva = np.isin(f, tr), np.isin(f, va)
    mu = P[mtr].mean((0, 2, 3), keepdims=True)
    sd = P[mtr].std((0, 2, 3), keepdims=True) + 1e-6
    members = [M.train_model((P[mtr] - mu) / sd, y[mtr], (P[mva] - mu) / sd, y[mva], "ce", seed=s, epochs=epochs)
               for s in range(seeds)]
    area = np.zeros((seeds, len(te)))
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
         "mean_ensemble_std_cells": float(summ["std"][tst].mean())}
    return C, te[tst], hflag, mflag
