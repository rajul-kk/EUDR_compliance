"""Tests for match_histogram in src/preprocessing/histogram_match.py."""
from unittest.mock import patch

import numpy as np


def test_channel_first_layout_and_dtype_survive_the_round_trip():
    from src.preprocessing.histogram_match import match_histogram
    src = np.arange(24, dtype=np.float32).reshape(1, 4, 6)
    ref = src.copy()
    with patch("skimage.exposure.match_histograms", side_effect=lambda s, r, **kw: s):
        out = match_histogram(src, ref)
    np.testing.assert_array_equal(out, src)
    assert out.dtype == np.float32
