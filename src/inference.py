import logging

import torch

logger = logging.getLogger(__name__)
import argparse
import glob
import os
import sys

import numpy as np
import rasterio

if __package__ in (None, ""):   # run by path (python src/x.py): make the repo root importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.deeplab_model import get_deeplab_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, in_channels=7):
    """
    Load a trained DeepLabV3 model from disk.
    
    Args:
        model_path: Path to .pth file
        in_channels: Number of input channels
    
    Returns:
        model: Loaded PyTorch model in eval mode
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    state = torch.load(model_path, map_location=DEVICE)
    model = get_deeplab_model(state["classifier.4.weight"].shape[0], in_channels, rgb_from_imagenet=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    logger.info("Model loaded from %s", model_path)
    return model


def load_tessera_model(model_path):
    """
    Load a trained TESSERA wrapper model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    from src.tessera_backbone import TesseraSegmentationModel

    model = TesseraSegmentationModel.load_from_checkpoint(model_path, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()

    logger.info("TESSERA model loaded from %s", model_path)
    return model

def load_image(image_path, expected_in_channels=6):
    """
    Load a Sentinel-2 GeoTIFF image and prepare for inference.
    
    Args:
        image_path: Path to .tiff/.tif file
    
    Returns:
        image_tensor: Torch tensor (1, C, H, W)
        profile: Rasterio profile for saving output
    """
    with rasterio.open(image_path) as src:
        image = src.read()  # (C, H, W)
        profile = src.profile

        if image.shape[0] < 5:
            raise ValueError(f"Expected at least 5 bands, got {image.shape[0]} in {os.path.basename(image_path)}")

        image = image.astype(np.float32)

        # Append derived indices to match training channel layout [R,G,B,NIR,SCL,NDVI,NDWI]
        if image.shape[0] == 5:
            red, nir, green = image[0], image[3], image[1]
            ndvi = (nir - red) / (nir + red + 1e-8)
            ndwi = (green - nir) / (green + nir + 1e-8)
            image = np.concatenate([image,
                                    np.expand_dims(ndvi, 0),
                                    np.expand_dims(ndwi, 0)], axis=0)  # → 7 channels

        if image.shape[0] != expected_in_channels:
            raise ValueError(
                f"Expected {expected_in_channels} channels after preprocessing, "
                f"got {image.shape[0]} in {os.path.basename(image_path)}"
            )

        image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, C, H, W)

    return image_tensor, profile

def predict_single_image(model, image_path, output_path=None, expected_in_channels=7,
                         uncertainty=False, mc_passes=20):
    """Run inference on a single image and optionally save the result.

    Args:
        uncertainty: If True, run MC Dropout and save an entropy uncertainty GeoTIFF
                     alongside the prediction (suffix _uncertainty.tif).
        mc_passes:   Number of stochastic passes for MC Dropout (default 20).
    """
    from src.train_utils import mc_dropout_predict

    image_tensor, profile = load_image(image_path, expected_in_channels=expected_in_channels)
    image_tensor = image_tensor.to(DEVICE)

    if uncertainty:
        mean_probs, unc_map = mc_dropout_predict(model, image_tensor, n_passes=mc_passes)
        prediction = torch.argmax(mean_probs, dim=1).squeeze(0).cpu().numpy()
        unc_np = unc_map.squeeze(0).cpu().numpy()
    else:
        with torch.no_grad():
            output = model(image_tensor)['out']
            prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        unc_np = None

    if output_path:
        output_profile = profile.copy()
        output_profile.update({'count': 1, 'dtype': rasterio.uint8, 'compress': 'lzw'})
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(prediction.astype(rasterio.uint8), 1)
        logger.debug("Saved prediction to %s", output_path)

        if unc_np is not None:
            unc_path = output_path.replace('.tif', '_uncertainty.tif')
            unc_profile = profile.copy()
            unc_profile.update({'count': 1, 'dtype': rasterio.float32, 'compress': 'lzw'})
            with rasterio.open(unc_path, 'w', **unc_profile) as dst:
                dst.write(unc_np.astype(np.float32), 1)
            logger.debug("Saved uncertainty map to %s", unc_path)

    return prediction

def batch_inference(model_path, input_dir, output_dir, file_pattern="*.tiff", model_type="deeplab",
                    uncertainty=False, mc_passes=20):
    """
    Run inference on all images in a directory.
    
    Args:
        model_path: Path to trained model
        input_dir: Directory containing input images
        output_dir: Directory to save predictions
        file_pattern: Glob pattern for input files
    
    Returns:
        results: Dictionary mapping input paths to output paths
    """
    if model_type == "deeplab":
        model = load_model(model_path, in_channels=7)
    elif model_type == "tessera":
        model = load_tessera_model(model_path)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if torch.cuda.device_count() > 1:
        logger.info("Using DataParallel across %d GPUs for inference", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    image_files = glob.glob(os.path.join(input_dir, file_pattern))

    if not image_files:
        logger.warning("No images found matching %s in %s", file_pattern, input_dir)
        return {}

    logger.info("Running inference on %d images", len(image_files))

    results = {}
    for i, image_path in enumerate(image_files, 1):
        try:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_predicted.tif")

            logger.info("[%d/%d] Processing %s", i, len(image_files), base_name)
            predict_single_image(model, image_path, output_path, expected_in_channels=7,
                                 uncertainty=uncertainty, mc_passes=mc_passes)

            results[image_path] = output_path

        except Exception as e:
            logger.error("Failed to process %s: %s", os.path.basename(image_path), e)
            continue

    logger.info("Inference complete: %d/%d images processed", len(results), len(image_files))
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Run segmentation inference for DeepLab or TESSERA model.")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--input-dir", required=True, help="Directory containing input GeoTIFF files")
    parser.add_argument("--output-dir", required=True, help="Directory where predictions will be written")
    parser.add_argument("--file-pattern", default="*.tiff", help="Glob pattern for input files")
    parser.add_argument("--model-type", choices=["deeplab", "tessera"], default="deeplab")
    parser.add_argument("--uncertainty", action="store_true",
                        help="Run MC Dropout and save per-pixel entropy uncertainty maps alongside predictions")
    parser.add_argument("--mc-passes", type=int, default=20,
                        help="Number of MC Dropout forward passes (default 20; use 10 for production)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_inference(
        model_path=args.model_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_pattern=args.file_pattern,
        model_type=args.model_type,
        uncertainty=args.uncertainty,
        mc_passes=args.mc_passes,
    )
