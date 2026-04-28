import os
import sys
import shutil
import zipfile
import psutil
import subprocess
from tqdm import tqdm

def check_disk_space(required_bytes, target_path):
    """
    Analyzes the structural occupancy of the target physical volume to ensure 
    safe extraction of massive isolated datasets natively.
    """
    free_bytes = psutil.disk_usage(target_path).free
    # 2026 Safety: 2.5x buffer for zip + extraction + working memory
    needed_with_buffer = int(required_bytes * 2.5)
    
    if free_bytes < needed_with_buffer:
        free_gb = free_bytes / (1024**3)
        needed_gb = needed_with_buffer / (1024**3)
        print(f"\n⚠️  [DISK SPACE WARNING] Low volume detected!")
        print(f"   Available: {free_gb:.2f} GB | Minimum Needed: {needed_gb:.2f} GB")
        ans = input("   👉 Proceed violently anyway? (y/n): ").strip().lower()
        return ans == 'y'
    return True

def download_and_extract_dataset(ds_name, data_dir, source_ref=None):
    """
    Universal 2026 Acquisition Engine (v3.0)
    Supports Kaggle and HuggingFace protocols with Zero-Latency pre-fetch compatibility.
    """
    ds_path = os.path.join(data_dir, ds_name)
    if os.path.exists(ds_path):
        return True

    # 1. Resolve Protocol & Reference
    ref = source_ref
    if not ref:
        # Fallback to config resolution
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        if os.path.exists(config_path):
            import yaml
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                urls = config.get("kaggle_dataset_urls", {})
                ref = urls.get(ds_name)
    
    # 2. Protocol Routing
    if not ref or "kaggle" in ref or "kaggle.com" in ref:
        return _handle_kaggle(ds_name, data_dir, ref)
    elif "hf://" in ref or "huggingface" in ref:
        return _handle_huggingface(ds_name, data_dir, ref)
    
    return False

def _handle_kaggle(ds_name, data_dir, ref):
    print(f"\n--- 🧪 Autonomic Kaggle Recovery: {ds_name} ---")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        # Resolve Slug
        slug = f"lemtreursi/{ds_name.lower()}"
        if ref:
            if "datasets/" in ref: slug = ref.split("datasets/")[-1]
            elif "kaggle://" in ref: slug = ref.replace("kaggle://", "")
            
        os.makedirs(data_dir, exist_ok=True)
        api.dataset_download_files(slug, path=data_dir, quiet=False, unzip=True)
        
        # Cleanup orphaned zip if unzip=True left any
        import glob
        for z in glob.glob(os.path.join(data_dir, "*.zip")):
            os.remove(z)
            
        print(f"   ✅ Kaggle {ds_name} mapped natively.\n")
        return True
    except Exception as e:
        print(f"   ❌ Kaggle Error: {e}")
        return False

def _handle_huggingface(ds_name, data_dir, ref):
    print(f"\n--- 🧪 Autonomic HuggingFace Recovery: {ds_name} ---")
    repo_id = ref.replace("hf://", "").split("huggingface.co/")[-1]
    ds_path = os.path.join(data_dir, ds_name)
    
    try:
        os.makedirs(ds_path, exist_ok=True)
        # 2026: Direct CLI streaming for maximum throughput
        cmd = ["huggingface-cli", "download", repo_id, "--local-dir", ds_path, "--local-dir-use-symlinks", "False"]
        print(f"   [STREAM] {repo_id} -> {ds_path}")
        subprocess.run(cmd, check=True)
        print(f"   ✅ HuggingFace {ds_name} mapped natively.\n")
        return True
    except Exception as e:
        print(f"   ❌ HuggingFace Error: {e}")
        return False
