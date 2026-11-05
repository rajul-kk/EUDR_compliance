"""Noise transition matrix estimation.

Convention: T[i, j] = P(observed = j | true = i), rows sum to 1.
"""
from __future__ import annotations

import numpy as np


def confusion_counts(true: np.ndarray, obs: np.ndarray, k: int) -> np.ndarray:
    idx = true.astype(np.int64) * k + obs.astype(np.int64)
    return np.bincount(idx.ravel(), minlength=k * k).reshape(k, k)


def _row_norm(c):
    rs = c.sum(1, keepdims=True)
    return np.divide(c, rs, out=np.full(c.shape, np.nan), where=rs > 0)


def transition_from_reference(true, obs, k=2) -> np.ndarray:
    return _row_norm(confusion_counts(np.asarray(true), np.asarray(obs), k).astype(float))


def bootstrap_transition(true, obs, groups, k=2, n_boot=2000, seed=0):
    """Cluster bootstrap over `groups`. Returns (T_hat, lo, hi), each (k,k)."""
    true, obs, groups = map(np.asarray, (true, obs, groups))
    ug, inv = np.unique(groups, return_inverse=True)
    per = np.stack([confusion_counts(true[inv == g], obs[inv == g], k) for g in range(len(ug))]).astype(float)
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, k, k))
    for b in range(n_boot):
        out[b] = _row_norm(per[rng.integers(0, len(ug), len(ug))].sum(0))
    return _row_norm(per.sum(0)), np.nanpercentile(out, 2.5, 0), np.nanpercentile(out, 97.5, 0)


def anchor_point_T(probs: np.ndarray, percentile: float = 97.0) -> np.ndarray:
    """Patrini et al. 2017 estimator. probs: (N,k) predicted P(observed label | x).

    Anchor for class i = example at the given percentile of p_i; T[i,:] = probs[anchor_i].
    Assumes each class has at least one clean, unambiguous example.
    """
    n, k = probs.shape
    T = np.zeros((k, k))
    for i in range(k):
        thr = np.percentile(probs[:, i], percentile)
        sel = probs[probs[:, i] >= thr]
        T[i] = sel[np.argmin(sel[:, i])]
    T = np.clip(T, 1e-6, None)
    return T / T.sum(1, keepdims=True)


def inject_noise(y: np.ndarray, T: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample observed labels given true labels y and transition matrix T."""
    y = np.asarray(y)
    u = rng.random(len(y))
    cum = np.cumsum(T[y], axis=1)
    return (u[:, None] > cum).sum(1).clip(0, T.shape[1] - 1)
