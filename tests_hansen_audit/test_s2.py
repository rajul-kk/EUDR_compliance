import numpy as np
import pytest

from hansen_audit import s2


def test_harmonise_removes_offset_only_for_new_baseline():
    dn = np.array([1000, 2000], np.uint16)
    assert np.allclose(s2.harmonise(dn, "03.00"), [0.1, 0.2])
    assert np.allclose(s2.harmonise(dn, "04.00"), [0.0, 0.1])
    assert np.allclose(s2.harmonise(dn, "05.09"), [0.0, 0.1])
    assert np.allclose(s2.harmonise(dn, None), [0.1, 0.2])


def test_harmonise_clips_nodata_to_zero():
    assert s2.harmonise(np.array([0], np.uint16), "05.00")[0] == 0.0


def test_clear_mask():
    scl = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    assert s2.clear_mask(scl).tolist() == [False, False, True, False, True, True, True, True, False, False, False, True]


def test_grid_bounds():
    w, s, e, n = s2.grid_bounds((100.0, 1.0), (4, 8))
    assert (w, n) == (100.0, 1.0)
    assert e == pytest.approx(100.0 + 8 * 0.00025)
    assert s == pytest.approx(1.0 - 4 * 0.00025)


def test_aggregate_block_mean_and_std():
    fine = np.arange(36, dtype=np.float32).reshape(6, 6)
    m = s2.aggregate(fine)
    assert m.shape == (2, 2)
    assert m[0, 0] == pytest.approx(np.mean([0, 1, 2, 6, 7, 8, 12, 13, 14]))
    sd = s2.aggregate(fine, fn=np.nanstd)
    assert sd[1, 1] == pytest.approx(np.std([21, 22, 23, 27, 28, 29, 33, 34, 35]))


def test_aggregate_ignores_nan():
    fine = np.ones((3, 3), np.float32)
    fine[0, 0] = np.nan
    assert s2.aggregate(fine)[0, 0] == 1.0


def test_composite_median_and_validity():
    S, B, Hs, Ws = 3, len(s2.BANDS), 6, 3  # 2 x 1 cells
    stack = np.zeros((S, B, Hs, Ws), np.float32)
    stack[:, 2] = 0.1  # B04 red
    stack[:, 3] = np.array([0.3, 0.5, 0.9])[:, None, None]  # B08 NIR per scene
    clear = np.ones((S, Hs, Ws), bool)
    clear[2] = False  # scene 3 cloudy everywhere -> median of 0.3, 0.5 = 0.4
    clear[:, 3:, :] = False  # bottom cell: no clear obs -> invalid
    clear[0, 3:, :] = True  # one obs only (< MIN_OBS) -> still invalid
    out = s2.composite_to_cells(stack, clear)
    ch = {c: i for i, c in enumerate(s2.CHANNELS)}
    assert out.shape == (len(s2.CHANNELS), 2, 1)
    assert out[ch["B08"], 0, 0] == pytest.approx(0.4)
    assert out[ch["NDVI"], 0, 0] == pytest.approx((0.4 - 0.1) / 0.5)
    assert out[ch["VALID_FRAC"], 0, 0] == 1.0
    assert np.isnan(out[ch["B08"], 1, 0])
    assert out[ch["VALID_FRAC"], 1, 0] == 0.0
    assert out[ch["NOBS"], 1, 0] == 1.0
