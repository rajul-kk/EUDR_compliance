import numpy as np
import pytest

from hansen_audit import gfc


def test_gfc_forest_threshold_is_inclusive():
    assert gfc.gfc_forest(np.array([49, 50, 100])).tolist() == [False, True, True]
    assert gfc.gfc_forest(np.array([29, 30]), threshold=30).tolist() == [False, True]


def test_farm_counts_excludes_nodata():
    labels = np.array([0, 1, 2, 2, 255])
    forest = np.array([True, True, False, True, True])
    c = gfc.farm_counts(labels, forest)
    assert c == {"class0": 1, "class1": 1, "class2": 2, "class0_gfc": 1, "class1_gfc": 1, "class2_gfc": 1}


def test_gated_flags_use_the_product_rule():
    h, g = gfc.gated_flags(np.array([5, 6, 20]), np.array([5, 6, 3]))
    assert h.tolist() == [False, True, True]
    assert g.tolist() == [False, True, False]


def test_ratio_ci_empty_is_none():
    assert gfc.ratio_ci(np.array([0, 0]), np.array([0, 0])) is None


def test_ratio_ci_pooled_estimate_inside_interval():
    r = gfc.ratio_ci(np.array([1, 3, 0]), np.array([2, 4, 2]))
    assert r["est"] == pytest.approx(0.5)
    assert r["lo"] <= r["est"] <= r["hi"]


def test_agreement_known_kappa():
    a = gfc.agreement(both=1, hansen_only=1, gfc_only=0, neither=2)
    assert a["agreement"] == pytest.approx(0.75)
    assert a["kappa"] == pytest.approx(0.5)


def test_agreement_empty():
    assert gfc.agreement(0, 0, 0, 0)["kappa"] is None


def test_agreement_from_counts_maps_classes():
    col = {k: np.array(v) for k, v in {"class0": [3], "class0_gfc": [1], "class1": [4], "class1_gfc": [3],
                                       "class2": [2], "class2_gfc": [1]}.items()}
    a = gfc.agreement_from_counts(col)
    assert (a["both"], a["hansen_only"], a["gfc_only"], a["neither"]) == (4, 2, 1, 2)


@pytest.mark.parametrize("p,f,want", [(0.8, 0.9, "S2C and GFC2020 agree"), (0.5, 0.9, "S2C and GFC2020 disagree"),
                                      (0.9, 0.4, "S2C and GFC2020 disagree"), (0.6, 0.9, "mixed"),
                                      (None, 0.9, "too few cells")])
def test_s2c_reading(p, f, want):
    assert gfc.s2c_reading(p, f) == want


def test_s2c_reading_names_the_map_it_compares():
    assert gfc.s2c_reading(0.1, 0.9, "Descals") == "S2C and Descals disagree"
