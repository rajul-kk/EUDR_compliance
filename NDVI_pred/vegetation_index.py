"""
Vegetation Index Calculator - Calculates NDVI (Normalized Difference Vegetation Index)
NDVI measures the "greenness" of vegetation using NIR and Red bands.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import rasterio
import os


class VegetationIndex:
    """Calculate vegetation indices from satellite imagery."""
    
    # Sentinel-2 band indices (0-indexed for typical array storage)
    BAND_BLUE = 1      # B2: 490nm
    BAND_GREEN = 2     # B3: 560nm
    BAND_RED = 3       # B4: 665nm
    BAND_NIR = 7       # B8: 842nm
    BAND_SWIR1 = 11    # B11: 1610nm
    BAND_SWIR2 = 12    # B12: 2190nm
    
    def __init__(self):
        """Initialize the vegetation index calculator."""
        pass
        
    def load_satellite_image(self, filepath: str) -> Dict[str, Any]:
        """
        Load a Sentinel-2 image using rasterio.
        
        Args:
            filepath: Path to the TIFF file
            
        Returns:
            Dictionary containing:
            - red: Red band array (B04)
            - green: Green band array (B03)
            - blue: Blue band array (B02)
            - nir: NIR band array (B08)
            - profile: Rasterio profile metadata
            - transform: Geospatial transform
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image file not found: {filepath}")
            
        print(f"📂 Loading satellite image: {filepath}")
        
        with rasterio.open(filepath) as src:
            # We requested 4 bands: B04 (Red), B03 (Green), B02 (Blue), B08 (NIR)
            # Rasterio uses 1-based indexing for bands
            
            # Read bands (FLOAT32 as requested in evalscript)
            # Note: evalscript order was [B04, B03, B02, B08]
            red = src.read(1)
            green = src.read(2)
            blue = src.read(3)
            nir = src.read(4)
            
            profile = src.profile.copy()
            transform = src.transform
            
            print(f"   Image Size: {src.width}x{src.height}")
            print(f"   Bands: {src.count}")
            print(f"   CRS: {src.crs}")
            
            return {
                "red": red,
                "green": green,
                "blue": blue,
                "nir": nir,
                "profile": profile,
                "transform": transform
            }
    
    def calculate_ndvi(self, red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """
        Calculate NDVI (Normalized Difference Vegetation Index).
        
        NDVI = (NIR - Red) / (NIR + Red)
        
        Values range from -1 to 1:
        - Dense vegetation: 0.6 to 0.9
        - Sparse vegetation: 0.2 to 0.5
        - Bare soil: -0.1 to 0.2
        - Water: -1 to 0
        
        Args:
            red_band: Red band array (Sentinel-2 B4)
            nir_band: NIR band array (Sentinel-2 B8)
            
        Returns:
            NDVI array with values from -1 to 1
        """
        # Avoid division by zero
        denominator = nir_band.astype(float) + red_band.astype(float)
        denominator[denominator == 0] = np.nan
        
        ndvi = (nir_band.astype(float) - red_band.astype(float)) / denominator
        
        # Clip to valid range
        ndvi = np.clip(ndvi, -1, 1)
        
        return ndvi
    
    def calculate_evi(
        self,
        red_band: np.ndarray,
        nir_band: np.ndarray,
        blue_band: np.ndarray,
        G: float = 2.5,
        C1: float = 6.0,
        C2: float = 7.5,
        L: float = 1.0
    ) -> np.ndarray:
        """
        Calculate EVI (Enhanced Vegetation Index).
        
        EVI is more sensitive in high biomass regions and reduces
        atmospheric influences.
        
        EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)
        
        Args:
            red_band: Red band array
            nir_band: NIR band array
            blue_band: Blue band array
            G, C1, C2, L: EVI coefficients
            
        Returns:
            EVI array
        """
        nir = nir_band.astype(float)
        red = red_band.astype(float)
        blue = blue_band.astype(float)
        
        denominator = nir + C1 * red - C2 * blue + L
        denominator[denominator == 0] = np.nan
        
        evi = G * (nir - red) / denominator
        
        return evi
    
    def calculate_ndmi(self, nir_band: np.ndarray, swir_band: np.ndarray) -> np.ndarray:
        """
        Calculate NDMI (Normalized Difference Moisture Index).
        
        NDMI = (NIR - SWIR) / (NIR + SWIR)
        
        High values indicate high vegetation water content.
        
        Args:
            nir_band: NIR band array (Sentinel-2 B8)
            swir_band: SWIR band array (Sentinel-2 B11)
            
        Returns:
            NDMI array
        """
        denominator = nir_band.astype(float) + swir_band.astype(float)
        denominator[denominator == 0] = np.nan
        
        ndmi = (nir_band.astype(float) - swir_band.astype(float)) / denominator
        
        return ndmi
    
    def classify_land_cover(self, ndvi: np.ndarray) -> np.ndarray:
        """
        Classify land cover based on NDVI values.
        
        Args:
            ndvi: NDVI array
            
        Returns:
            Classification array with values:
            0 = Water
            1 = Bare soil/Urban
            2 = Sparse vegetation
            3 = Moderate vegetation
            4 = Dense vegetation (Forest)
        """
        classification = np.zeros_like(ndvi, dtype=np.uint8)
        
        classification[ndvi < 0] = 0            # Water
        classification[(ndvi >= 0) & (ndvi < 0.2)] = 1     # Bare soil
        classification[(ndvi >= 0.2) & (ndvi < 0.4)] = 2   # Sparse vegetation
        classification[(ndvi >= 0.4) & (ndvi < 0.6)] = 3   # Moderate vegetation
        classification[ndvi >= 0.6] = 4          # Dense vegetation
        
        return classification
    
    def is_forested(self, ndvi: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Create a binary forest mask based on NDVI threshold.
        
        Args:
            ndvi: NDVI array
            threshold: NDVI threshold for forest classification
            
        Returns:
            Binary mask where 1 = forest, 0 = non-forest
        """
        return (ndvi >= threshold).astype(np.uint8)
    
    def calculate_forest_percentage(self, ndvi: np.ndarray, threshold: float = 0.5) -> float:
        """
        Calculate the percentage of area covered by forest.
        
        Args:
            ndvi: NDVI array
            threshold: NDVI threshold for forest classification
            
        Returns:
            Forest coverage percentage (0-100)
        """
        valid_pixels = ~np.isnan(ndvi)
        forest_pixels = (ndvi >= threshold) & valid_pixels
        
        if np.sum(valid_pixels) == 0:
            return 0.0
        
        return (np.sum(forest_pixels) / np.sum(valid_pixels)) * 100


if __name__ == "__main__":
    # Example usage with REAL data
    vi = VegetationIndex()
    
    # Path to the image we validated earlier
    # Note: Use the path to the current best image
    # We'll try to find the latest .tiff in the training_chips directory
    data_dir = "data/training_chips"
    
    image_path = None
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith(".tiff")]
        if files:
            # Pick the most recent one (or just the first one)
            image_path = os.path.join(data_dir, files[0])
            print(f"🎯 Selected input image: {image_path}")
    
    if image_path and os.path.exists(image_path):
        try:
            # 1. Load Data
            data = vi.load_satellite_image(image_path)
            
            # 2. Calculate NDVI
            print("🧮 Calculating NDVI...")
            ndvi = vi.calculate_ndvi(data["red"], data["nir"])
            
            # 3. Analyze Results
            min_val = np.nanmin(ndvi)
            max_val = np.nanmax(ndvi)
            mean_val = np.nanmean(ndvi)
            forest_pct = vi.calculate_forest_percentage(ndvi)
            
            print(f"\n📊 ANALYSIS RESULTS:")
            print(f"   NDVI Range: {min_val:.3f} to {max_val:.3f}")
            print(f"   Mean NDVI:  {mean_val:.3f}")
            print(f"   Forest Cover: {forest_pct:.1f}%")
            
            # 4. Save NDVI Result
            output_path = image_path.replace(".tiff", "_NDVI.tiff")
            
            # Update profile for single band NDVI output
            profile = data["profile"]
            profile.update(
                count=1,
                dtype=rasterio.float32,
                driver="GTiff"
            )
            
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(ndvi.astype(rasterio.float32), 1)
                
            print(f"\n💾 Saved NDVI map to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ No .tiff files found in data/training_chips. Calculating with synthetic data instead...")
        # Fallback to synthetic data
        np.random.seed(42)
        red = np.random.uniform(0.05, 0.15, (100, 100))
        nir = np.random.uniform(0.3, 0.6, (100, 100))
        ndvi = vi.calculate_ndvi(red, nir)
        print(f"Synthetic NDVI range: {np.nanmin(ndvi):.3f} to {np.nanmax(ndvi):.3f}")
