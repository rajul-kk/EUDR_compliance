
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

def run_pipeline(skip_farm_discovery=False, skip_image_download=False):
    """
    Runs the end-to-end pipeline:
    1. Discover farm locations from OSM (find_farms.py) - Optional
    2. Download Sentinel-2 satellite imagery (sentinel_client.py) - Optional
    3. Generate Hybrid Masks using GEE (generate_hybrid.py)
    4. Train DeepLabV3 Model (ML_farm_net.py)
    
    Args:
        skip_farm_discovery: Skip Step 1 if farms CSV already exists
        skip_image_download: Skip Step 2 if satellite images already downloaded
    """
    print("🚀 Starting End-to-End EUDR Compliance Pipeline")
    print(f"Project Root: {project_root}")
    
    # Use the current python executable
    python_exe = sys.executable
    print(f"Using Python: {python_exe}")

    # --- Step 1: Find Farms from OSM ---
    if not skip_farm_discovery:
        print("\n" + "="*60)
        print("[Step 1/4] Discovering Farm Locations from OpenStreetMap...")
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
        print("\n[Step 1/4] ⏭️  Skipping Farm Discovery (using existing inputs/farms_osm.csv)")

    # --- Step 2: Download Satellite Images ---
    if not skip_image_download:
        print("\n" + "="*60)
        print("[Step 2/4] Downloading Sentinel-2 Satellite Imagery...")
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
        print("\n[Step 2/4] ⏭️  Skipping Image Download (using existing data/raw_satellite/)")

    # --- Step 3: Generate/Download Masks ---
    print("\n" + "="*60)
    print("[Step 3/4] Generating Hybrid Masks from GEE...")
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
    print("[Step 4/4] Training DeepLabV3 Model...")
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

    print("\n🎉 Pipeline Finished Successfully!")
    print("📊 Trained model saved to: models/farm_deeplab.pth")

if __name__ == "__main__":
    # Default: Skip farm discovery and image download (assumes data already exists)
    # Remove skip flags to run full pipeline from scratch
    run_pipeline(
        skip_farm_discovery=True,  # Set to False to search OSM for new farms
        skip_image_download=True   # Set to False to download new satellite images
    )
