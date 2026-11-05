import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch, pytest
from hansen_audit import noise, losses, decisions, uncertainty
from hansen_audit.io import IMG_RE, LAB_RE, farm_key


def test_filename_parsing():
    assert IMG_RE.match("relation_10018408_2020_2020-06-07.tiff")["fid"] == "relation_10018408"
    assert IMG_RE.match("way_61751665_2024_2024-06-03.tiff")["year"] == "2024"
    assert LAB_RE.match("relation_1_hansen_label.tif")
    assert not LAB_RE.match("relation_1_2020.tif")
    assert farm_key("osm_relation_5") == "relation_5"


def test_transition_from_reference_and_bootstrap():
    rng = np.random.default_rng(0)
    T = np.array([[0.9, 0.1], [0.3, 0.7]])
    y = rng.integers(0, 2, 40000)
    o = noise.inject_noise(y, T, rng)
    assert np.abs(noise.transition_from_reference(y, o) - T).max() < 0.02
    g = rng.integers(0, 20, len(y))
    _, lo, hi = noise.bootstrap_transition(y, o, g, n_boot=200)
    assert (lo <= T + 1e-9).all() and (hi >= T - 1e-9).all()


def test_anchor_point_recovers_T_with_clean_anchors():
    T = np.array([[0.8, 0.2], [0.25, 0.75]])
    c = np.concatenate([np.linspace(0.0, 1, 500), [0.0, 1.0] * 20])
    probs = np.stack([1 - c, c], 1) @ T
    assert np.abs(noise.anchor_point_T(probs, percentile=99.9) - T).max() < 0.02


def test_losses_basic_properties():
    torch.manual_seed(0)
    logits = torch.randn(64, 2)
    y = torch.randint(0, 2, (64,))
    assert losses.gce(logits, y, q=1e-3).item() == pytest.approx(losses.ce(logits, y).item(), rel=1e-2)
    assert losses.forward_ce(logits, y, torch.eye(2)).item() == pytest.approx(losses.ce(logits, y).item(), rel=1e-5)
    lg = torch.tensor([[-8.0, 8.0]], requires_grad=True)
    losses.gce(lg, torch.tensor([0])).backward()
    lg2 = torch.tensor([[-8.0, 8.0]], requires_grad=True)
    losses.ce(lg2, torch.tensor([0])).backward()
    assert lg.grad.abs().max().item() < 0.05 < lg2.grad.abs().max().item()
    assert torch.isfinite(losses.sce(logits, y))


def test_forward_correction_recovers_clean_prior():
    T = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    g = torch.Generator().manual_seed(1)
    y = torch.randint(0, 2, (4000,), generator=g)
    u = torch.rand(4000, generator=g)
    yn = (u > T[y].cumsum(1)[:, 0]).long()
    z = torch.zeros(1, 2, requires_grad=True)
    opt = torch.optim.Adam([z], 0.05)
    for _ in range(300):
        opt.zero_grad(); losses.forward_ce(z.expand(4000, 2), yn, T).backward(); opt.step()
    p = torch.softmax(z, 1)[0, 1].item()
    assert abs(p - y.float().mean().item()) < 0.04   # clean prior, not the noisy prior


def test_decisions():
    assert decisions.farm_flag(np.array([0, 5, 6, 100])).tolist() == [False, False, True, True]
    flag = np.array([1, 1, 0, 0, 1, 0], bool); tp = np.array([1, 0, 0, 0, 0, 1], bool)
    est, lo, hi = decisions.false_flag_rate(flag, tp, n_boot=200)
    assert est == pytest.approx(2 / 4) and lo <= est <= hi
    assert decisions.missed_flag_rate(flag, tp, n_boot=50)[0] == pytest.approx(0.5)
    assert decisions.expected_false_flag_prob(3, 0.9) == 0.0
    assert 0 < decisions.expected_false_flag_prob(10, 0.5) < 1
    assert decisions.expected_false_flag_prob(10, 1.0) == 0.0


def test_conformal_coverage():
    rng = np.random.default_rng(0)
    q = uncertainty.split_conformal_halfwidth(np.abs(rng.normal(size=20000)), 0.1)
    assert abs((np.abs(rng.normal(size=50000)) <= q).mean() - 0.9) < 0.01
    assert uncertainty.split_conformal_halfwidth(np.array([1.0]), 0.1) == float("inf")
    assert uncertainty.decision_with_interval([2.0, 10.0, 6.0], 1.0, 6).tolist() == [0, 1, 2]
