import os
import sys
import shutil
import subprocess
import psutil
from pathlib import Path

# [SENIOR HARDENING v16.0 - SYNC_ID: 1212]

def check_disk_space(required_bytes, target_path):
    """Manifold Safety Check: Ensure structural occupancy is sufficient."""
    free_bytes = psutil.disk_usage(target_path).free
    needed_with_buffer = int(required_bytes * 1.5) # Optimized 2026 buffer
    
    if free_bytes < needed_with_buffer:
        print(f"⚠️ [LOW DISK] Available: {free_bytes/(1024**3):.1f}GB | Needed: {needed_with_buffer/(1024**3):.1f}GB")
        return False
    return True

def download_and_extract_dataset(ds_name, data_dir, source_ref=None):
    """Universal Acquisition Engine (v16.0 Nuclear)."""
    ds_path = os.path.join(data_dir, ds_name)
    if os.path.exists(ds_path): return True

    ref = source_ref
    if not ref:
        # Resolve via config fallback
        pass

    if not ref or "kaggle" in ref:
        return _handle_kaggle(ds_name, data_dir, ref)
    return False

def _handle_kaggle(ds_name, data_dir, ref):
    """Atomic Kaggle Recovery (v16.0)."""
    print(f"🚀 [DATA] Recovering {ds_name} from Kaggle Manifold...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        slug = f"lemtreursi/{ds_name.lower()}"
        if ref and "datasets/" in ref: slug = ref.split("datasets/")[-1]
            
        ds_path = os.path.join(data_dir, ds_name)
        os.makedirs(ds_path, exist_ok=True)
        api.dataset_download_files(slug, path=ds_path, quiet=True, unzip=True)
        
        # Cleanup zips
        for f in os.listdir(ds_path):
            if f.endswith(".zip"): os.remove(os.path.join(ds_path, f))
            
        print(f"✅ [DATA] {ds_name} successfully mapped.")
        return True
    except Exception as e:
        print(f"❌ [DATA ERROR] {e}")
        return False
