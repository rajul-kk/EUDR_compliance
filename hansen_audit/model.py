"""Small CPU model and training loop for cell-level loss vs stable classification (PROTOCOL)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import losses as L

NEIGH = 3


def make_patches(X: np.ndarray, Y: np.ndarray, V: np.ndarray, min_valid: float = 0.5):
    """X (F,C,G,G), Y (F,G,G) in {0,1,2}, V (F,G,G).
    Returns patches (N,C,3,3) float32, label (N,) {0 stable-forest,1 loss}, farm index (N,).
    Only cells with Hansen class 1 or 2, valid share >= min_valid, and a fully valid 3x3 window (no NaN) are kept."""
    Fn, C, G, _ = X.shape
    Xp = np.nan_to_num(np.pad(X, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=np.nan), nan=0.0)
    bad = np.pad(np.isnan(X).any(1), ((0, 0), (1, 1), (1, 1)), constant_values=True)
    ps, ys, fs = [], [], []
    for f in range(Fn):
        ii, jj = np.where((Y[f] >= 1) & (V[f] >= min_valid) & ~np.isnan(X[f]).any(0))
        if len(ii) == 0:
            continue
        for i, j in zip(ii, jj):
            ps.append(Xp[f, :, i:i + 3, j:j + 3]); ys.append(Y[f, i, j] - 1); fs.append(f)
    return np.stack(ps).astype(np.float32), np.array(ys, np.int64), np.array(fs, np.int64)


class PatchCNN(nn.Module):
    def __init__(self, c_in=15, w=32, k=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, w, 3, padding=1), nn.ReLU(),
            nn.Conv2d(w, w, 3, padding=1), nn.ReLU(),
            nn.Conv2d(w, w, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(w, k))

    def forward(self, x):
        return self.net(x)


def loss_fn(name, T=None, q=0.7):
    if name == "ce":
        return L.ce
    if name == "gce":
        return lambda lg, y: L.gce(lg, y, q=q)
    if name == "sce":
        return L.sce
    if name == "forward":
        Tt = torch.as_tensor(T, dtype=torch.float32)
        return lambda lg, y: L.forward_ce(lg, y, Tt)
    raise ValueError(name)


def standardise(train_p, *others):
    mu = train_p.mean((0, 2, 3), keepdims=True); sd = train_p.std((0, 2, 3), keepdims=True) + 1e-6
    return [(p - mu) / sd for p in (train_p, *others)]


def train_model(Xtr, ytr, Xva, yva, loss_name="ce", T=None, seed=0, epochs=15, bs=256, lr=1e-3, q=0.7):
    """Early stopping on validation *noisy-label* loss (as in PROTOCOL; no clean data used)."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = PatchCNN(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr)
    lf = loss_fn(loss_name, T, q)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva)
    best, best_state = np.inf, None
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); lf(m(Xt[idx]), yt[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(lf(m(Xv), yv))
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    return m


@torch.no_grad()
def predict_proba(m, X, bs=4096):
    m.eval()
    out = [F.softmax(m(torch.from_numpy(X[i:i + bs])), 1).numpy() for i in range(0, len(X), bs)]
    return np.concatenate(out)


def f1_binary(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)
