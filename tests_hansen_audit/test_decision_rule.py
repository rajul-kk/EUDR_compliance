import numpy as np

from hansen_audit import decisions as D


def test_threshold_is_more_than_half_a_hectare():
    assert D.plot_decision(5000.0) == D.CLEAR
    assert D.plot_decision(5000.1) == D.FLAGGED
    assert D.plot_decision(0.0, forest_2020_m2=1e4) == D.CLEAR
    assert D.plot_decision(1e5, forest_2020_m2=0) == D.NO_BASELINE_FOREST


def test_interval_straddling_the_threshold_is_uncertain():
    assert D.plot_decision(5500.0, halfwidth_m2=400.0) == D.FLAGGED
    assert D.plot_decision(5500.0, halfwidth_m2=600.0) == D.UNCERTAIN
    assert D.plot_decision(4500.0, halfwidth_m2=400.0) == D.CLEAR
    assert D.plot_decision(4500.0, halfwidth_m2=600.0) == D.UNCERTAIN


def test_cell_rule_is_the_area_rule_at_30_m():
    assert D.DEFAULT_MIN_CELLS == 6
    n = np.arange(0, 40)
    area = [D.plot_decision(k * D.CELL_M2) == D.FLAGGED for k in n]
    assert D.farm_flag(n).tolist() == area


def test_min_cells_for_other_resolutions():
    assert D.min_cells_for(100.0) == 51
    assert D.min_cells_for(400.0) == 13
