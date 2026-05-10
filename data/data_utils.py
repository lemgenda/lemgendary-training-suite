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

def download_and_extract_dataset(ds_name, data_dir, config, source_ref=None):
    """Universal Acquisition Engine (v16.0 Nuclear)."""
    ds_path = os.path.join(data_dir, ds_name)
    if os.path.exists(ds_path): return True

    ref = source_ref
    if not ref:
        # Resolve via config fallback
        pass

    if not ref or "kaggle" in ref:
        return _handle_kaggle(ds_name, data_dir, config, ref)
    return False

def _handle_kaggle(ds_name, data_dir, config, ref):
    """Atomic Kaggle Recovery (v16.2 Nuclear)."""
    print(f"[DATA] Recovering {ds_name} from Kaggle Manifold...")
    try:
        import kagglehub
        import shutil
        
        # 2026 Resilience: Use configured username or default to lemtreursi
        username = config.get("fleet", {}).get("kaggle_username", "lemtreursi")
        
        slug = f"{username}/{ds_name.lower().replace('_', '-')}"
        if ref and "datasets/" in ref: 
            slug = ref.split("datasets/")[-1]
        elif ref and "kaggle://" in ref:
            slug = ref.replace("kaggle://", "")
            
        print(f"   -> Pulling: {slug}")
        print("   -> Starting download (this may take a while for large manifolds)...")
        cache_path = kagglehub.dataset_download(slug)
        print(f"   -> Download complete. Cache path: {cache_path}")
        
        ds_path = os.path.join(data_dir, ds_name)
        if not os.path.exists(ds_path):
            os.makedirs(ds_path, exist_ok=True)
            
        # 2026 Resilience: Move from cache to datasets_root
        if os.path.isfile(cache_path):
            shutil.copy2(cache_path, os.path.join(ds_path, os.path.basename(cache_path)))
        else:
            # Recursive directory merge
            for item in os.listdir(cache_path):
                s = os.path.join(cache_path, item)
                d = os.path.join(ds_path, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            
        print(f"✅ [DATA] {ds_name} successfully synchronized to manifold.")
        return True
    except Exception as e:
        print(f"❌ [DATA ERROR] Acquisition failed for {ds_name}: {e}")
        return False
