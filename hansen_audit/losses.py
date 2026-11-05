"""Noise-robust losses (torch). logits: (N,k), target: (N,) long."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ce(logits, y, weight=None):
    return F.cross_entropy(logits, y, weight=weight)


def gce(logits, y, q: float = 0.7):
    """Generalised cross entropy (Zhang & Sabuncu 2018): (1 - p_y^q)/q. q->0 is CE, q=1 is MAE."""
    p = F.softmax(logits, 1).gather(1, y[:, None]).squeeze(1).clamp_min(1e-7)
    return ((1 - p ** q) / q).mean()


def sce(logits, y, alpha: float = 0.1, beta: float = 1.0):
    """Symmetric CE (Wang et al. 2019): alpha*CE + beta*RCE, log 0 clipped to log(1e-4)."""
    k = logits.shape[1]
    p = F.softmax(logits, 1).clamp(1e-7, 1)
    oh = F.one_hot(y, k).float().clamp(1e-4, 1)
    rce = -(p * oh.log()).sum(1).mean()
    return alpha * F.cross_entropy(logits, y) + beta * rce


def forward_ce(logits, y, T):
    """Forward correction (Patrini 2017): NLL of (p_clean @ T) on the observed label.
    T: (k,k) tensor, T[i,j] = P(obs j | true i)."""
    pn = (F.softmax(logits, 1) @ T).clamp_min(1e-7)
    return F.nll_loss(pn.log(), y)
