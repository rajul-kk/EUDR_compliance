"""Tests for match_histogram in src/preprocessing/histogram_match.py."""
import numpy as np
import pytest
from unittest.mock import patch


def test_output_shape_preserved():
    from src.preprocessing.histogram_match import match_histogram
    src = np.random.randint(0, 256, (6, 64, 64), dtype=np.uint8).astype(np.float32)
    ref = np.random.randint(0, 256, (6, 64, 64), dtype=np.uint8).astype(np.float32)
    with patch("skimage.exposure.match_histograms", side_effect=lambda s, r, **kw: s):
        out = match_histogram(src, ref)
    assert out.shape == src.shape


def test_output_dtype_preserved():
    from src.preprocessing.histogram_match import match_histogram
    src = np.ones((6, 32, 32), dtype=np.float32)
    ref = np.ones((6, 32, 32), dtype=np.float32)
    with patch("skimage.exposure.match_histograms", side_effect=lambda s, r, **kw: s):
        out = match_histogram(src, ref)
    assert out.dtype == np.float32


def test_passthrough_identity():
    from src.preprocessing.histogram_match import match_histogram
    src = np.arange(24, dtype=np.float32).reshape(1, 4, 6)
    ref = src.copy()
    with patch("skimage.exposure.match_histograms", side_effect=lambda s, r, **kw: s):
        out = match_histogram(src, ref)
    np.testing.assert_array_equal(out, src)
