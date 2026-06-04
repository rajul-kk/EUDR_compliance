"""Per-channel histogram matching to normalise temporal composites."""

import numpy as np


def match_histogram(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match the histogram of src to ref, channel-wise.

    Args:
        src: Float32 array of shape (C, H, W) — the image to transform.
        ref: Float32 array of shape (C, H, W) — the reference distribution.

    Returns:
        Transformed src with the same shape, dtype float32.
    """
    try:
        from skimage.exposure import match_histograms
        # skimage expects (H, W, C); transpose, match, transpose back
        src_hwc = np.transpose(src, (1, 2, 0))
        ref_hwc = np.transpose(ref, (1, 2, 0))
        matched = match_histograms(src_hwc, ref_hwc, channel_axis=-1)
        return np.transpose(matched, (2, 0, 1)).astype(np.float32)
    except ImportError:
        # Fallback: per-channel linear stretch to match mean/std
        out = src.copy().astype(np.float32)
        for c in range(src.shape[0]):
            s, r = src[c], ref[c]
            s_std = s.std() + 1e-8
            r_std = r.std() + 1e-8
            out[c] = (s - s.mean()) / s_std * r_std + r.mean()
        return out
