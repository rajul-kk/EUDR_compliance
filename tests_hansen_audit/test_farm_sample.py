import csv
import math

import numpy as np
import pytest

from hansen_audit import farm_sample as S


def test_stratum_id_and_invalid_cells():
    h = np.array([0, 1, 2, 2, 255, 1])
    p = np.array([0, 1, 2, -1, 0, 0])
    assert S.stratum_id(h, p).tolist() == [0, 4, 8, -1, -1, 3]


def test_n_for_halfwidth():
    assert S.n_for_halfwidth(0.5, 0.1) == 97
    assert S.n_for_halfwidth(0.5, 0.1, deff=2) == 193


def test_allocate_floor_proportional_and_caps():
    n = S.allocate({6: 1000, 7: 300, 8: 10, 3: 0}, n_total=200, n_min=30)
    assert sum(n.values()) == 200 and 3 not in n
    assert n[8] == 10 and n[6] > n[7] >= 30
    with pytest.raises(ValueError):
        S.allocate({1: 100, 2: 100}, n_total=10, n_min=30)


def test_allocate_never_exceeds_population():
    n = S.allocate({1: 5, 2: 7}, n_total=100)
    assert n == {1: 5, 2: 7}


def test_draw_is_stratified_without_replacement_and_reproducible():
    strata = np.repeat([0, 4, 8], [50, 30, 20])
    idx = S.draw(strata, {0: 10, 4: 5, 8: 20}, seed=3)
    assert len(set(idx.tolist())) == 35
    assert np.bincount(strata[idx], minlength=9)[[0, 4, 8]].tolist() == [10, 5, 20]
    assert (S.draw(strata, {0: 10, 4: 5, 8: 20}, seed=3) == idx).all()
    with pytest.raises(ValueError):
        S.draw(strata, {8: 21}, seed=0)


def test_shares_equal_simple_mean_under_proportional_allocation():
    N_h = {6: 600, 7: 400}
    s = np.array([6] * 6 + [7] * 4)
    y = ["FOREST"] * 3 + ["PLANT"] * 3 + ["PLANT"] * 4
    r = S.stratified_shares(N_h, s, y)
    assert abs(r["classes"]["PLANT"]["est"] - 0.7) < 1e-12
    assert abs(r["classes"]["FOREST"]["est"] - 0.3) < 1e-12
    var = (0.6 ** 2) * 0.5 * 0.5 / 5
    assert abs(r["classes"]["FOREST"]["se"] - math.sqrt(var)) < 1e-12


def test_shares_weight_by_population_not_sample():
    N_h = {6: 900, 7: 100}
    s = np.array([6] * 10 + [7] * 10)
    y = ["FOREST"] * 10 + ["PLANT"] * 10
    r = S.stratified_shares(N_h, s, y)
    assert abs(r["classes"]["FOREST"]["est"] - 0.9) < 1e-12 and r["classes"]["FOREST"]["se"] == 0.0


def test_shares_domain_drops_unsure_and_reports_empty_strata():
    N_h = {6: 100, 7: 100, 0: 1000}
    s = np.array([6, 6, 6, 0, 0])
    y = ["PLANT", "UNSURE", "FOREST", "OTHER", "OTHER"]
    r = S.stratified_shares(N_h, s, y, domain=[6, 7])
    assert (r["n_labelled"], r["n_dropped"]) == (2, 1)
    assert r["strata_without_labels"] == [7]


def test_map_accuracy_perfect_and_confused():
    N_h = {0: 50, 1: 50}
    r = S.map_accuracy(N_h, {0: 0, 1: 1}, [0, 0, 1, 1], ["FOREST", "FOREST", "PLANT", "PLANT"])
    assert r["overall"] == 1.0 and r["users"]["PLANT"] == 1.0
    r = S.map_accuracy(N_h, {0: 0, 1: 1}, [0, 0, 1, 1], ["FOREST", "PLANT", "PLANT", "PLANT"])
    assert abs(r["overall"] - 0.75) < 1e-12
    assert abs(r["users"]["FOREST"] - 0.5) < 1e-12 and abs(r["producers"]["PLANT"] - 0.5 / 0.75) < 1e-12


def test_agreement_kappa():
    a = ["FOREST", "PLANT", "PLANT", "OTHER", "UNSURE"]
    b = ["FOREST", "PLANT", "OTHER", "OTHER", "PLANT"]
    r = S.agreement(a, b)
    assert r["n"] == 4 and r["agreement"] == 0.75
    po, pe = 0.75, 0.25 * 0.25 + 0.5 * 0.25 + 0.25 * 0.5
    assert abs(r["kappa"] - (po - pe) / (1 - pe)) < 1e-12


def test_sheets_hide_strata_and_shuffle(tmp_path):
    lon, lat = np.array([101.0, 101.1, 101.2]), np.array([-1.0, -1.1, -1.2])
    key = [{"stratum": 8, "farm": "way_1"}, {"stratum": 6, "farm": "way_2"}, {"stratum": 7, "farm": "way_3"}]
    S.write_sheets(tmp_path / "sheet.csv", tmp_path / "key.csv", lon, lat, key, seed=1, prefix="A")
    sheet = list(csv.DictReader(open(tmp_path / "sheet.csv")))
    k = {r["sample_id"]: r for r in csv.DictReader(open(tmp_path / "key.csv"))}
    assert "stratum" not in sheet[0] and "farm" not in sheet[0]
    assert {r["sample_id"] for r in sheet} == set(k)
    for r in sheet:
        i = [f"{x:.6f}" for x in lon].index(r["lon"])
        assert k[r["sample_id"]]["farm"] == key[i]["farm"] and r["lat"] == f"{lat[i]:.6f}"


def test_plantation_driven_rate_extremes_and_missing_labels():
    counts = np.array([[0, 10, 0], [7, 1, 0], [1, 1, 1]])
    all_forest = S.plantation_driven_rate(counts, [0, 1, 2], [True, True, True], n_boot=50, n_draw=20)
    assert all_forest["n_flagged"] == 2 and all_forest["est"] == 0.0 and all_forest["hi"] == 0.0
    none_forest = S.plantation_driven_rate(counts, [0, 1, 2], [False, False, False], n_boot=50, n_draw=20)
    assert none_forest["est"] == 1.0 and none_forest["lo"] == 1.0
    missing = S.plantation_driven_rate(counts, [0, 2], [True, False], n_boot=10)
    assert missing["est"] is None and missing["unlabelled_pred_classes"] == ["PLANT"]


def test_plantation_driven_rate_matches_raw_rule_when_prediction_is_right():
    counts = np.array([[0, 10, 0], [7, 1, 0], [5, 5, 5]])
    r = S.plantation_driven_rate(counts, [0] * 5 + [1] * 5 + [2] * 5, [True] * 5 + [False] * 10, n_boot=50, n_draw=20)
    assert abs(r["est"] - 2 / 3) < 1e-12
