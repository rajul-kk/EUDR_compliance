
import subprocess
import sys
import os

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
find_farms_script = os.path.join(current_dir, 'find_farms.py')
sentinel_script = os.path.join(current_dir, 'sentinel_client.py')
generate_script = os.path.join(project_root, 'GEE_dynamic', 'src', 'generate_hybrid.py')
train_script = os.path.join(current_dir, 'ML_farm_net.py')
inference_script = os.path.join(current_dir, 'inference.py')
detection_script = os.path.join(current_dir, 'detect_deforestation.py')

def run_pipeline(skip_farm_discovery=True, skip_image_download=True, skip_inference=True):
    """
    Runs the end-to-end pipeline:
    1. Discover farm locations from OSM (find_farms.py) - Optional
    2. Download Sentinel-2 satellite imagery (sentinel_client.py) - Optional
    3. Generate Hybrid Masks using GEE (generate_hybrid.py)
    4. Train DeepLabV3 Model (ML_farm_net.py)
    5. Run Inference on 2024 images (inference.py) - Optional
    6. Detect Deforestation (detect_deforestation.py) - Optional
    
    Args:
        skip_farm_discovery: Skip Step 1 if farms CSV already exists
        skip_image_download: Skip Step 2 if satellite images already downloaded
        skip_inference: Skip Steps 5-6 if only training is needed
    """
    print("🚀 Starting End-to-End EUDR Compliance Pipeline")
    print(f"Project Root: {project_root}")
    
    # Use the current python executable
    python_exe = sys.executable
    print(f"Using Python: {python_exe}")

    # --- Step 1: Find Farms from OSM ---
    if not skip_farm_discovery:
        print("\n" + "="*60)
        print("[Step 1/6] Discovering Farm Locations from OpenStreetMap...")
        print("="*60)
        try:
            if not os.path.exists(find_farms_script):
                raise FileNotFoundError(f"Script not found: {find_farms_script}")
                
            subprocess.check_call([python_exe, find_farms_script], cwd=project_root)
            print("✅ Farm Discovery Complete (inputs/farms_osm.csv updated).")
        except subprocess.CalledProcessError as e:
            print(f"❌ Farm Discovery Failed with exit code {e.returncode}.")
            print("   Continuing with existing farms_osm.csv if available...")
        except Exception as e:
            print(f"❌ An error occurred during farm discovery: {e}")
            print("   Continuing with existing farms_osm.csv if available...")
    else:
        print("\n[Step 1/6] ⏭️  Skipping Farm Discovery (using existing inputs/farms_osm.csv)")

    # --- Step 2: Download Satellite Images ---
    if not skip_image_download:
        print("\n" + "="*60)
        print("[Step 2/6] Downloading Sentinel-2 Satellite Imagery...")
        print("="*60)
        try:
            if not os.path.exists(sentinel_script):
                raise FileNotFoundError(f"Script not found: {sentinel_script}")
                
            subprocess.check_call([python_exe, sentinel_script], cwd=project_root)
            print("✅ Satellite Image Download Complete.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Image Download Failed with exit code {e.returncode}.")
            return
        except Exception as e:
            print(f"❌ An error occurred during image download: {e}")
            return
    else:
        print("\n[Step 2/6] ⏭️  Skipping Image Download (using existing data/raw_satellite/)")

    # --- Step 3: Generate/Download Masks ---
    print("\n" + "="*60)
    print("[Step 3/6] Generating Hybrid Masks from GEE...")
    print("="*60)
    try:
        # Check if script exists
        if not os.path.exists(generate_script):
            raise FileNotFoundError(f"Script not found: {generate_script}")
            
        subprocess.check_call([python_exe, generate_script], cwd=project_root)
        print("✅ Mask Generation Complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Mask Generation Failed with exit code {e.returncode}.")
        return
    except Exception as e:
        print(f"❌ An error occurred during mask generation: {e}")
        return

    # --- Step 4: Train Model ---
    print("\n" + "="*60)
    print("[Step 4/6] Training DeepLabV3 Model...")
    print("="*60)
    try:
         if not os.path.exists(train_script):
            raise FileNotFoundError(f"Script not found: {train_script}")

         subprocess.check_call([python_exe, train_script], cwd=project_root)
         print("✅ Training Complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training Failed with exit code {e.returncode}.")
        return
    except Exception as e:
        print(f"❌ An error occurred during training: {e}")
        return

    # --- Step 5: Run Inference on 2024 Images ---
    if not skip_inference:
        print("\n" + "="*60)
        print("[Step 5/6] Running Inference on 2024 Satellite Images...")
        print("="*60)
        try:
            if not os.path.exists(inference_script):
                raise FileNotFoundError(f"Script not found: {inference_script}")
                
            subprocess.check_call([python_exe, inference_script], cwd=project_root)
            print("✅ Inference Complete.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Inference Failed with exit code {e.returncode}.")
            return
        except Exception as e:
            print(f"❌ An error occurred during inference: {e}")
            return
    else:
        print("\n[Step 5/6] ⏭️  Skipping Inference")

    # --- Step 6: Detect Deforestation ---
    if not skip_inference:
        print("\n" + "="*60)
        print("[Step 6/6] Detecting Deforestation (2020 vs 2024)...")
        print("="*60)
        try:
            if not os.path.exists(detection_script):
                raise FileNotFoundError(f"Script not found: {detection_script}")
                
            subprocess.check_call([python_exe, detection_script], cwd=project_root)
            print("✅ Deforestation Detection Complete.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Detection Failed with exit code {e.returncode}.")
            return
        except Exception as e:
            print(f"❌ An error occurred during detection: {e}")
            return
    else:
        print("\n[Step 6/6] ⏭️  Skipping Deforestation Detection")

    print("\n🎉 Pipeline Finished Successfully!")
    print("📊 Outputs:")
    print("   - Trained model: models/farm_deeplab.pth")
    if not skip_inference:
        print("   - 2024 predictions: data/predictions_2024/")
        print("   - Compliance report: reports/deforestation_report.csv")
        print("   - Summary stats: reports/summary_stats.json")

if __name__ == "__main__":
    # Default: Skip farm discovery and image download (assumes data already exists)
    # Enable inference and detection for full deforestation analysis
    run_pipeline(
        skip_farm_discovery=True,  # Set to False to search OSM for new farms
        skip_image_download=True,  # Set to False to download new satellite images
        skip_inference=False       # Set to True to only train the model
    )
