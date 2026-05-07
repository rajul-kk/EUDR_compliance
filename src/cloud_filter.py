from __future__ import annotations

import numpy as np

try:
    from s2cloudless import S2PixelCloudDetector as _S2Detector
except ImportError:
    _S2Detector = None  # type: ignore[assignment]


class CloudFilter:
    """Wraps s2cloudless to produce per-pixel cloud masks for Sentinel-2 imagery.

    Expects input arrays in the 13-band Sentinel-2 L1C order:
    B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
    Values should be reflectance in [0, 1].
    """

    # s2cloudless band indices (0-based within the 13-band stack)
    REQUIRED_BANDS = [0, 1, 3, 4, 7, 8, 9, 10, 11, 12]  # bands used by s2cloudless

    def __init__(
        self,
        threshold: float = 0.4,
        average_over: int = 4,
        dilation_size: int = 2,
    ) -> None:
        if _S2Detector is None:
            raise ImportError("s2cloudless is required: pip install s2cloudless")
        self._detector = _S2Detector(
            threshold=threshold,
            average_over=average_over,
            dilation_size=dilation_size,
            all_bands=True,
        )

    def get_cloud_mask(self, image: np.ndarray) -> np.ndarray:
        """Return a binary cloud mask (1 = cloud, 0 = clear).

        Args:
            image: Array of shape (H, W, 13) or (1, H, W, 13) with Sentinel-2
                   L1C reflectance values in [0, 1].

        Returns:
            Binary mask of shape (H, W).
        """
        if image.ndim == 3:
            image = image[np.newaxis]  # add batch dim -> (1, H, W, 13)
        cloud_probs = self._detector.get_cloud_probability_maps(image)  # (1, H, W)
        return (cloud_probs[0] >= self._detector.threshold).astype(np.uint8)

    def cloud_coverage_pct(self, image: np.ndarray) -> float:
        """Return percentage of pixels classified as cloud (0–100)."""
        mask = self.get_cloud_mask(image)
        return float(mask.mean()) * 100.0

    @staticmethod
    def scl_cloud_mask(scl_band: np.ndarray) -> np.ndarray:
        """Build a cloud mask from the Sentinel-2 Scene Classification Layer (SCL).

        SCL classes treated as cloud/shadow:
          0 = No Data, 1 = Saturated, 3 = Cloud Shadow,
          8 = Medium Cloud, 9 = High Cloud, 10 = Thin Cirrus

        Args:
            scl_band: 2-D integer array of SCL values.

        Returns:
            Binary mask of shape (H, W) where 1 = cloud/shadow.
        """
        cloud_classes = {0, 1, 3, 8, 9, 10}
        mask = np.zeros_like(scl_band, dtype=np.uint8)
        for cls in cloud_classes:
            mask[scl_band == cls] = 1
        return mask
