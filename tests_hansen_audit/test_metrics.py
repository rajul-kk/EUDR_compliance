import numpy as np
import pytest

from hansen_audit import metrics as Mt


def test_split_farms_is_a_seeded_60_20_20_partition():
    tr, va, te = Mt.split_farms(np.arange(10), seed=3)
    assert (len(tr), len(va), len(te)) == (6, 2, 2)
    assert sorted(np.concatenate([tr, va, te]).tolist()) == list(range(10))
    assert np.array_equal(Mt.split_farms(np.arange(10), seed=3)[0], tr)


def test_f1_from_one_row_or_pooled_farm_rows():
    c = Mt.tp_fp_fn(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 1]))
    assert c.tolist() == [1, 1, 1] and Mt.f1(c) == pytest.approx(0.5)
    rows = Mt.per_farm_counts(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 1]), np.array([0, 0, 1, 1]), [0, 1])
    assert rows.tolist() == [[1, 0, 1], [0, 1, 0]] and Mt.f1(rows) == pytest.approx(0.5)
    assert Mt.f1([0, 0, 0]) == 0.0


def test_balanced_accuracy():
    assert Mt.balanced_accuracy(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0])) == pytest.approx(0.75)


def test_bootstrap_of_an_f1_difference():
    a = np.array([[5, 1, 1], [3, 0, 2], [4, 2, 0]], float)
    b = np.array([[2, 3, 4], [1, 2, 4], [2, 2, 2]], float)
    assert Mt.boot_f1_diff(a, a, n=50) == [0.0, 0.0, 0.0]
    mean, lo, hi = Mt.boot_f1_diff(a, b, n=200)
    assert lo <= mean <= hi and lo > 0
    mean, lo, hi = Mt.boot_f1_diff_unpaired(a, b, n=200)
    assert lo <= mean <= hi and lo > 0
