import numpy as np
import pytest

from hansen_audit import landuse as L
from hansen_audit import s2


def test_nan_box_mean_ignores_nan_and_edges():
    x = np.array([[1, 2, np.nan], [4, 5, 6], [7, 8, 9]], np.float32)
    m = L.nan_box_mean(x)
    assert m[0, 0] == pytest.approx(np.mean([1, 2, 4, 5]))
    assert m[1, 1] == pytest.approx(np.mean([1, 2, 4, 5, 6, 7, 8, 9]))
    assert np.isnan(L.nan_box_mean(np.full((2, 2), np.nan, np.float32))).all()


def test_cell_features_shape_and_invalid_propagation():
    C = len(s2.CHANNELS)
    cells = np.random.default_rng(0).random((C, 4, 5)).astype(np.float32)
    cells[s2.CHANNELS.index("NDVI"), 0, 0] = np.nan
    f = L.cell_features(cells)
    assert f.shape == (len(L.FEATURES), 4, 5)
    assert np.isnan(f[:, 0, 0]).all()
    assert np.isfinite(f[:, 1, 1]).all()


def test_group_index():
    g = L.group_index(np.array([9, 12, 1, 2, 7, 6, 10, 14, 0, 3]))
    assert g.tolist() == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]


def test_assign_folds_balanced_and_deterministic():
    gids = [f"{i:02d}" for i in range(12)]
    isl = ["A"] * 6 + ["B"] * 6
    f1 = L.assign_folds(gids, isl, k=3)
    assert f1 == L.assign_folds(gids, isl, k=3)
    for isl_name, block in (("A", gids[:6]), ("B", gids[6:])):
        counts = np.bincount([f1[g] for g in block], minlength=3)
        assert counts.tolist() == [2, 2, 2]


def test_confusion_and_metrics():
    c = L.confusion_counts(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2]))
    assert c.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    rec, prec = L.recall_precision(c)
    assert rec.tolist() == [0.5, 1.0, 1.0]
    assert prec.tolist() == [1.0, 0.5, 1.0]
    assert L.balanced_accuracy(c) == pytest.approx(2.5 / 3)


def test_corrected_shares_recovers_truth():
    true_pi = np.array([0.2, 0.5, 0.3])
    C = np.array([[0.8, 0.15, 0.05], [0.1, 0.85, 0.05], [0.1, 0.1, 0.8]])
    q = C.T @ true_pi
    pi, clipped = L.corrected_shares(C * 1000, q * 10000)
    assert np.allclose(pi, true_pi, atol=1e-9)
    assert not clipped


def test_corrected_shares_clips_negative():
    C = np.eye(3) * 0.7 + 0.1
    pi, clipped = L.corrected_shares(C * 100, np.array([0.0, 50.0, 50.0]))
    assert clipped and (pi >= 0).all() and pi.sum() == pytest.approx(1.0)


def test_cluster_bootstrap_identity_confusion():
    G = 10
    conf = np.tile(np.eye(3) * 100, (G, 1, 1))
    pred = np.tile(np.array([20, 50, 30]), (G, 1))
    out = L.cluster_bootstrap(conf, pred, n_boot=50)
    assert out["raw"]["PLANT"]["est"] == pytest.approx(0.5)
    assert out["corrected"]["PLANT"]["est"] == pytest.approx(0.5)
    assert out["corrected"]["PLANT"]["lo"] == pytest.approx(0.5)
    assert out["n_cells"] == 1000


def _brute_window(x, r, fn):
    H, W = x.shape
    out = np.full((H, W), np.nan, np.float64)
    for i in range(H):
        for j in range(W):
            w = x[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1]
            w = w[np.isfinite(w)]
            if w.size:
                out[i, j] = fn(w)
    return out


def test_nan_window_stats_matches_brute_force():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(23, 17)).astype(np.float32)
    x[rng.random(x.shape) < 0.3] = np.nan
    x[:9, :9] = np.nan  # a block larger than the r=3 window, so some windows are empty
    for r in (1, 3):
        m, s = L.nan_window_stats(x, r)
        np.testing.assert_allclose(m, _brute_window(x, r, np.mean), atol=1e-5, equal_nan=True)
        np.testing.assert_allclose(s, _brute_window(x, r, np.std), atol=1e-4, equal_nan=True)
    np.testing.assert_allclose(L.nan_window_stats(x, 1)[0], L.nan_box_mean(x, 1), atol=1e-5, equal_nan=True)


def test_s2b_features_shape_names_and_invalid_propagation():
    rng = np.random.default_rng(0)
    cells = rng.uniform(0.01, 0.5, size=(len(s2.CHANNELS), 20, 30)).astype(np.float32)
    ndvi = s2.CHANNELS.index("NDVI")
    cells[ndvi, 3, 4] = np.nan
    F = L.cell_features(cells, s2.CHANNELS, "s2b")
    assert F.shape == (len(L.S2B_FEATURES), 20, 30) == (41, 20, 30)
    assert np.isnan(F[:, 3, 4]).all()
    assert np.isfinite(F[:, 10, 10]).all()
    names = list(L.S2B_FEATURES)
    np.testing.assert_allclose(F[names.index("NDVI")], cells[ndvi], equal_nan=True)
    b08, b12 = cells[s2.CHANNELS.index("B08")], cells[s2.CHANNELS.index("B12")]
    np.testing.assert_allclose(F[names.index("NBR"), 10, 10], (b08 - b12)[10, 10] / (b08 + b12)[10, 10], rtol=1e-5)
    # the default feature set is unchanged by S2B
    assert L.cell_features(cells, s2.CHANNELS).shape[0] == len(L.FEATURES) == 12


def test_s2c_features_appends_s1_and_handles_missing():
    from hansen_audit import s1

    rng = np.random.default_rng(0)
    cells = rng.uniform(0.01, 0.5, size=(len(s2.CHANNELS), 12, 15)).astype(np.float32)
    cells[s2.CHANNELS.index("NDVI"), 2, 2] = np.nan
    s1c = rng.normal(-12, 2, size=(len(s1.CHANNELS), 12, 15)).astype(np.float32)
    F = L.s2c_features(cells, s2.CHANNELS, s1c, s1.CHANNELS)
    assert F.shape[0] == len(L.S2C_FEATURES) == 54
    names = list(L.S2C_FEATURES)
    np.testing.assert_allclose(F[names.index("VH_med"), 5, 5], s1c[s1.CHANNELS.index("VH_med"), 5, 5])
    assert np.isnan(F[:, 2, 2]).all()
    np.testing.assert_allclose(F[:41], L.cell_features(cells, s2.CHANNELS, "s2b"), equal_nan=True)
    G = L.s2c_features(cells, s2.CHANNELS, None, None)
    assert np.isnan(G[41:]).all() and np.isfinite(G[:41, 5, 5]).all()
