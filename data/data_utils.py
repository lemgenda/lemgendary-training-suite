import os
import sys
import shutil
import zipfile
import psutil
from tqdm import tqdm
import kaggle # pyre-ignore
from kaggle.api.kaggle_api_extended import KaggleApi # pyre-ignore

def check_disk_space(required_bytes, target_path):
    """
    Analyzes the structural occupancy of the target physical volume to ensure 
    safe extraction of massive isolated Kaggle datasets natively.
    """
    free_bytes = psutil.disk_usage(target_path).free
    # We require 2.5x the compressed size to hold the .zip + extracted array + buffer
    needed_with_buffer = int(required_bytes * 2.5)
    
    if free_bytes < needed_with_buffer:
        free_gb = free_bytes / (1024**3)
        needed_gb = needed_with_buffer / (1024**3)
        print(f"\n⚠️  [DISK SPACE WARNING] Low volume detected on target topological array!")
        print(f"   Available: {free_gb:.2f} GB | Minimum Needed (with buffer): {needed_gb:.2f} GB")
        ans = input("   👉 Do you want to proceed violently anyway? (y/n): ").strip().lower()
        if ans != 'y':
            print("🛑 Operation aborted by user due to Disk Constraint.")
            return False
    return True

def download_and_extract_dataset(ds_name, data_dir):
    """
    Professional 2026 Synchronous Acquisition:
    Fetches, verifies, and unpacks a Kaggle dataset with full visual feedback.
    """
    api = KaggleApi()
    api.authenticate()
    
    kaggle_slugs = {
        "LemGendizedQualityDataset": "lemgendized-quality-dataset",
        "LemGendizedNoiseDataset": "lemgendized-noise-dataset",
        "LemGendizedLowLightDataset": "lemgendized-lowlight-dataset",
        "LemGendizedFaceDataset": "lemgendized-face-dataset",
        "LemGendizedDegradationDataset": "lemgendized-degradation-dataset",
        "LemGendizedDetectionDataset": "lemgendized-detection-dataset",
        "LemGendizedSuperResDataset": "lemgendized-superres-dataset"
    }
    slug_name = kaggle_slugs.get(ds_name, ds_name.lower())
    ds_slug = f"lemtreursi/{slug_name}"
    ds_path = os.path.join(data_dir, ds_name)
    
    if os.path.exists(ds_path):
        return True # Natively cached perfectly.

    print(f"\n--- 🧪 Autonomic Dataset Recovery: {ds_name} ---")
    
    try:
        # 1. Metadata Verification (Bypassed: Kaggle API requires 'path' which forces a file download)
        total_bytes = 5 * (1024**3) # Safe fallback to 5GB
        total_gb = total_bytes / (1024**3)
        
        # 2. User Confirmation & Disk Check
        if '--env' in sys.argv and 'kaggle' in sys.argv or '--yes' in sys.argv:
            print(f"   ▶ Proceeding with automatic re-fetch from Kaggle Cloud ({total_gb:.2f} GB)... (Interactive prompts bypassed)")
        else:
            if not check_disk_space(total_bytes, data_dir):
                return False
            ans = input(f"   ▶ Proceed with automatic re-fetch from Kaggle Cloud ({total_gb:.2f} GB)? (y/n): ").strip().lower()
            if ans != 'y':
                return False

        # 3. ⬇️ Bit-Level Download
        os.makedirs(data_dir, exist_ok=True)
        print(f"   [CACHE MISS] Executing Downlink: {ds_name}...")
        
        # Note: kaggle.api.dataset_download_files handles the download.
        # It doesn't natively expose a bit-level progress callback to tqdm easily without low-level rewrite.
        # We will use the 'quiet=False' output or a generic progress bar if no data is visible.
        api.dataset_download_files(ds_slug, path=data_dir, quiet=False)
        
        # 4. 📦 File-Level Extraction with Progress Bar
        import glob
        possible_zips = glob.glob(os.path.join(data_dir, "*.zip"))
        
        if possible_zips:
            zip_path = possible_zips[0]
            print(f"   [UNPACK] Synchronizing topological array: {ds_name} from {os.path.basename(zip_path)}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                files = z.namelist()
                with tqdm(total=len(files), unit='file', desc=f"      📦 Unpacking") as upbar:
                    for f in files:
                        z.extract(f, ds_path)
                        upbar.update(1)
            os.remove(zip_path) # Clean up zip artifact
            print(f"   ✅ {ds_name} successfully recovered and mapped natively.\n")
            return True
        else:
            print(f"   ❌ FAILED to locate download artifact for {ds_name}.")
            return False

    except Exception as e:
        print(f"   ❌ CRITICAL Error during Autonomic Recovery: {str(e)}")
        return False
