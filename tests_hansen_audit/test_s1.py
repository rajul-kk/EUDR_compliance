import numpy as np

from hansen_audit import s1


def test_pick_scenes_majority_orbit_evenly_spaced():
    recs = [(f"2020-{m:02d}-01", "descending") for m in range(1, 13)] + [("2020-01-05", "ascending")] * 3
    idx = s1.pick_scenes(recs)
    assert len(idx) == 6
    assert all(recs[i][1] == "descending" for i in idx)
    assert [recs[i][0] for i in idx] == sorted(recs[i][0] for i in idx)
    assert recs[idx[0]][0] == "2020-01-01" and recs[idx[-1]][0] == "2020-12-01"


def test_pick_scenes_tie_prefers_descending_and_few_items():
    recs = [("2020-02-01", "ascending"), ("2020-03-01", "descending")]
    assert s1.pick_scenes(recs) == [1]
    assert s1.pick_scenes([]) == []


def test_stats_to_cells_db_and_min_obs():
    S, H, W = 4, 2, 3
    vv = np.full((S, H, W), 0.1, np.float32)
    vh = np.full((S, H, W), 0.01, np.float32)
    vv[2:, 0, 0] = 0  # only 2 valid observations -> NaN
    out = s1.stats_to_cells(vv, vh)
    assert out.shape == (len(s1.CHANNELS), H, W)
    ch = dict(zip(s1.CHANNELS, out))
    np.testing.assert_allclose(ch["VV_med"][1, 1], -10.0, atol=1e-5)
    np.testing.assert_allclose(ch["VH_med"][1, 1], -20.0, atol=1e-5)
    np.testing.assert_allclose(ch["RATIO_med"][1, 1], -10.0, atol=1e-5)
    assert ch["VH_tstd"][1, 1] == 0
    assert np.isnan(out[:5, 0, 0]).all() and ch["NOBS"][0, 0] == 2
