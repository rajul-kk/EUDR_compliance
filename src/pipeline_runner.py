
import subprocess
import sys
import os

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
generate_script = os.path.join(project_root, 'GEE_dynamic', 'src', 'generate_hybrid.py')
train_script = os.path.join(current_dir, 'ML_farm_net.py')

def run_pipeline():
    """
    Runs the end-to-end pipeline:
    1. Download Hybrid Masks using GEE (generate_hybrid.py)
    2. Train DeepLabV3 Model (ML_farm_net.py)
    """
    print("🚀 Starting End-to-End Pipeline")
    print(f"Project Root: {project_root}")
    
    # Use the current python executable
    python_exe = sys.executable
    print(f"Using Python: {python_exe}")

    # --- Step 1: Generate/Download Masks ---
    print("\n" + "="*50)
    print("[Step 1/2] Generating Hybrid Masks from GEE...")
    print("="*50)
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

    # --- Step 2: Train Model ---
    print("\n" + "="*50)
    print("[Step 2/2] Training DeepLabV3 Model...")
    print("="*50)
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

if __name__ == "__main__":
    run_pipeline()
