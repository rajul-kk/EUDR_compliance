"""Farm splits and loss metrics shared by the training experiments (PROTOCOL.md, PROTOCOL_E6).

Counts are per farm as rows of (TP, FP, FN) for the loss class; F1 pools the rows it is given, and the bootstrap
resamples farms (rows), never cells."""
import numpy as np


def split_farms(farm_ids, seed=0):
    """(train, val, test) farm ids: a seeded permutation cut at 60 % and 80 %."""
    ids = np.random.default_rng(seed).permutation(farm_ids)
    a, b = int(.6 * len(ids)), int(.8 * len(ids))
    return ids[:a], ids[a:b], ids[b:]


def tp_fp_fn(yt, pred):
    """(TP, FP, FN) for the loss class (1) as a float array."""
    return np.array([((yt == 1) & (pred == 1)).sum(), ((yt == 0) & (pred == 1)).sum(), ((yt == 1) & (pred == 0)).sum()],
                    float)


def per_farm_counts(yt, pred, farm_of, farms):
    """Rows of (TP, FP, FN), one per farm in `farms`; `farm_of` gives each cell's farm."""
    return np.array([tp_fp_fn(yt[farm_of == k], pred[farm_of == k]) for k in farms])


def f1(counts):
    """F1 of the loss class from (TP, FP, FN), pooling rows when given one row per farm."""
    c = np.asarray(counts, float)
    tp, fp, fn = c.sum(0) if c.ndim == 2 else c
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def balanced_accuracy(yt, pred):
    return float(0.5 * ((pred[yt == 1] == 1).mean() + (pred[yt == 0] == 0).mean()))


def _summary(d):
    return [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def boot_f1_diff(ca, cb, n=2000, seed=0):
    """F1(ca) - F1(cb) for two results on the same farms (rows paired), farms resampled together: [mean, 2.5 %, 97.5 %]."""
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for i in range(n):
        j = rng.integers(0, len(ca), len(ca))
        d[i] = f1(ca[j]) - f1(cb[j])
    return _summary(d)


def boot_f1_diff_unpaired(ca, cb, n=2000, seed=0):
    """F1(ca) - F1(cb) for two disjoint sets of farms, each resampled on its own: [mean, 2.5 %, 97.5 %]."""
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for i in range(n):
        d[i] = f1(ca[rng.integers(0, len(ca), len(ca))]) - f1(cb[rng.integers(0, len(cb), len(cb))])
    return _summary(d)
