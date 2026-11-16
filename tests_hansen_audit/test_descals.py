import numpy as np

from hansen_audit import descals as D


def test_encoding_ok_accepts_years_and_one_non_palm_value():
    assert D.encoding_ok([0, 1990, 2005, 2021], None)[0]
    assert D.encoding_ok([1990, 2021], None)[0]


def test_encoding_ok_rejects_undocumented_values():
    ok, why = D.encoding_ok([0, 1989, 1990, 2021, 2022], None)
    assert not ok and "1989" in why and "2022" in why
    assert not D.encoding_ok([0], None)[0]


def test_indicators_split_at_the_cutoff():
    by2020, in2021 = D.indicators(np.array([0, 1989, 1990, 2020, 2021, 2022]))
    assert by2020.tolist() == [0, 0, 1, 1, 0, 0]
    assert in2021.tolist() == [0, 0, 0, 0, 1, 0]


def test_farm_counts_and_gated_flags():
    lab = np.array([2, 2, 2, 1, 255, 2])
    p20 = np.array([1, 1, 0, 1, 1, 0], bool)
    p21 = np.array([0, 0, 1, 0, 0, 0], bool)
    assert D.farm_counts(lab, p20, p21) == {"class2": 4, "class2_palm2020": 2, "class2_palm2021": 1}
    h, g = D.gated_flags(np.array([10, 8, 3]), np.array([2, 5, 0]))
    assert h.tolist() == [True, True, False] and g.tolist() == [True, False, False]


def test_map_reading_thresholds():
    assert D.map_reading(0.8).startswith("the two maps agree")
    assert D.map_reading(0.5) == "they disagree"
    assert D.map_reading(0.6) == "mixed"
    assert D.map_reading(None) == "too few cells"
