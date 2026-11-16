import numpy as np

from hansen_audit import descals as D


def test_encoding_ok_accepts_years_the_owner_read_values_and_one_non_palm_value():
    assert D.encoding_ok([0, 1990, 2005, 2021], None)[0]
    assert D.encoding_ok([0, 1989, 1990, 2021, 2022], None)[0]


def test_encoding_ok_rejects_other_values():
    ok, why = D.encoding_ok([0, 1988, 1990, 2021], None)
    assert not ok and "1988" in why
    assert not D.encoding_ok([0, 1989, 1990], None, extra=())[0]
    assert not D.encoding_ok([0], None)[0]


def test_indicators_split_into_the_four_categories():
    ind = D.indicators(np.array([0, 1989, 1990, 2020, 2021, 2022]))
    assert ind["y1989"].tolist() == [0, 1, 0, 0, 0, 0]
    assert ind["y1990_2020"].tolist() == [0, 0, 1, 1, 0, 0]
    assert ind["y2021"].tolist() == [0, 0, 0, 0, 1, 0]
    assert ind["y2022"].tolist() == [0, 0, 0, 0, 0, 1]


def test_palm_by_2020_counts_1989_unless_told_not_to():
    shares = {"y1989": np.array([30, 60, 0]), "y1990_2020": np.array([30, 0, 49])}
    assert D.palm_by_2020(shares).tolist() == [True, True, False]
    assert D.palm_by_2020(shares, count_1989=False).tolist() == [False, False, False]
    assert D.palm_by_2020(shares, threshold=0.3, count_1989=False).tolist() == [True, False, True]


def test_farm_counts_and_gated_flags():
    lab = np.array([2, 2, 2, 1, 255, 2])
    p20 = np.array([1, 1, 0, 1, 1, 0], bool)
    p21 = np.array([0, 0, 1, 0, 0, 0], bool)
    p22 = np.array([0, 0, 0, 0, 0, 1], bool)
    assert D.farm_counts(lab, p20, p21, p22) == {"class2": 4, "class2_palm2020": 2, "class2_palm2021": 1,
                                                 "class2_palm2022": 1}
    h, g = D.gated_flags(np.array([10, 8, 3]), np.array([2, 5, 0]))
    assert h.tolist() == [True, True, False] and g.tolist() == [True, False, False]


def test_map_reading_thresholds():
    assert D.map_reading(0.8).startswith("the two maps agree")
    assert D.map_reading(0.5) == "they disagree"
    assert D.map_reading(0.6) == "mixed"
    assert D.map_reading(None) == "too few cells"
