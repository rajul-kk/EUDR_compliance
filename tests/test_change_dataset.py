"""Tests for ChangeDetectionDataset pairing logic and Hansen resize."""
import os
import re
import tempfile

import numpy as np
import pytest


def _make_tif(path: str, data: np.ndarray) -> None:
    """Write a minimal GeoTIFF-like file using rasterio mock or real numpy save."""
    # We're not using real rasterio here; _build_pairs only calls os.listdir and
    # os.path.exists. We only need the files to exist on disk.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)  # just needs to exist; _is_zero_image is mocked


def test_build_pairs_hansen_matches_files(tmp_path):
    """Farm key extracted from t1 filename matches t2 and label."""
    t1 = tmp_path / "t1"
    t2 = tmp_path / "t2"
    lbl = tmp_path / "lbl"
    for d in (t1, t2, lbl):
        d.mkdir()

    (t1 / "way_123_2020.tif").write_bytes(b"")
    (t2 / "way_123_2024.tif").write_bytes(b"")
    (lbl / "way_123_hansen_label.tif").write_bytes(b"")

    from unittest.mock import patch
    # Skip zero-image check — files are empty stubs
    with patch("src.preprocessing.change_dataset.ChangeDetectionDataset._is_zero_image", return_value=False):
        from src.preprocessing.change_dataset import ChangeDetectionDataset
        ds = ChangeDetectionDataset(str(t1), str(t2), str(lbl), label_backend="hansen",
                                    histogram_match=False)

    assert len(ds) == 1
    assert "way_123_hansen_label.tif" in ds.pairs[0][2]


def test_build_pairs_skips_missing_label(tmp_path):
    """Farm pair is skipped when hansen label file is absent."""
    t1 = tmp_path / "t1"
    t2 = tmp_path / "t2"
    lbl = tmp_path / "lbl"
    for d in (t1, t2, lbl):
        d.mkdir()

    (t1 / "node_456_2020.tif").write_bytes(b"")
    (t2 / "node_456_2024.tif").write_bytes(b"")
    # No label file

    from unittest.mock import patch
    with patch("src.preprocessing.change_dataset.ChangeDetectionDataset._is_zero_image", return_value=False):
        from src.preprocessing.change_dataset import ChangeDetectionDataset
        ds = ChangeDetectionDataset(str(t1), str(t2), str(lbl), label_backend="hansen",
                                    histogram_match=False)

    assert len(ds) == 0


def test_build_pairs_skips_zero_images(tmp_path):
    """Farm pair is skipped when t1 image is all zeros."""
    t1 = tmp_path / "t1"
    t2 = tmp_path / "t2"
    lbl = tmp_path / "lbl"
    for d in (t1, t2, lbl):
        d.mkdir()

    (t1 / "rel_789_2020.tif").write_bytes(b"")
    (t2 / "rel_789_2024.tif").write_bytes(b"")
    (lbl / "rel_789_hansen_label.tif").write_bytes(b"")

    from unittest.mock import patch
    with patch("src.preprocessing.change_dataset.ChangeDetectionDataset._is_zero_image", return_value=True):
        from src.preprocessing.change_dataset import ChangeDetectionDataset
        ds = ChangeDetectionDataset(str(t1), str(t2), str(lbl), label_backend="hansen",
                                    histogram_match=False)

    assert len(ds) == 0


def test_hansen_label_resize():
    """Label smaller than image is nearest-neighbor upsampled to image size."""
    from PIL import Image as PILImage

    raw = np.array([[1, 2], [0, 1]], dtype=np.uint8)  # 2×2 label
    target_h, target_w = 4, 4

    # Replicate the resize logic from __getitem__
    resized = np.array(
        PILImage.fromarray(raw).resize((target_w, target_h), PILImage.NEAREST),
        dtype=np.int64,
    )
    assert resized.shape == (target_h, target_w)
    # Corner values must be preserved (nearest-neighbour, no blending)
    assert resized[0, 0] == 1
    assert resized[0, 2] == 2
    assert resized[2, 0] == 0


def test_farm_key_regex():
    """Regex correctly extracts farm_key from both filename formats."""
    pattern = r"(.+?)_(2020|2024)(?:_|\.tiff?)"
    cases = [
        ("way_123456789_2020.tif", "way_123456789"),
        ("node_987_2024.tiff", "node_987"),
        ("rel_42_2020_20231101.tiff", "rel_42"),
    ]
    for fname, expected_key in cases:
        m = re.match(pattern, fname)
        assert m is not None, f"No match for {fname}"
        assert m.group(1) == expected_key
