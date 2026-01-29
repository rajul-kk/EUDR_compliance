import numpy as np
import rasterio
# The AI Library
from s2cloudless import S2PixelCloudDetector 

class CloudFilter:
    def __init__(self, max_cloud_percentage: float = 20.0):
        self.max_cloud_percentage = max_cloud_percentage
        # AI Configuration: LightGBM model parameters
        self.cloud_detector = S2PixelCloudDetector(
            threshold=0.4, 
            average_over=4, 
            dilation_size=2,
            all_bands=False  # Set True if you have all 13 bands
        )

    def _detect_clouds(self, image_path: str) -> Optional[np.ndarray]:
        """
        Real AI Implementation:
        Uses s2cloudless to generate a probability map and binary mask.
        """
        try:
            with rasterio.open(image_path) as src:
                # s2cloudless requires specific bands: 
                # [B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12]
                # We assume the image is stacked correctly (channels_last)
                image_data = src.read() # Shape: (Bands, Height, Width)
                
                # Reshape for the AI: (Height, Width, Bands)
                image_data = np.moveaxis(image_data, 0, -1)
                
                # Values must be 0.0-1.0 or 0-10000. 
                # If floats are > 1, assume they are raw counts and normalize.
                if image_data.max() > 1.0:
                    image_data = image_data / 10000.0

                # Run Inference
                # Returns shape (1, Height, Width)
                cloud_masks = self.cloud_detector.get_cloud_masks(image_data[np.newaxis, ...])
                
                # Extract single mask: 0=Clear, 1=Cloud
                return cloud_masks[0]

        except Exception as e:
            print(f"Cloud detection failed for {image_path}: {e}")
            return None