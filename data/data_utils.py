"""Dataset acquisition helpers: disk-space checks and Kaggle dataset recovery."""

import logging
import os
import shutil

import psutil

from training.cloud.errors import kagglehub_errors
from training.ecosystem import kaggle_owner
from training.utils.lazy_imports import load_kagglehub

# [SENIOR HARDENING v16.0 - SYNC_ID: 1212]

logger = logging.getLogger("lemtrain.data.data_utils")


def check_disk_space(required_bytes, target_path):
    """Manifold Safety Check: Ensure structural occupancy is sufficient."""
    free_bytes = psutil.disk_usage(target_path).free
    needed_with_buffer = int(required_bytes * 1.5)  # Optimized 2026 buffer

    if free_bytes < needed_with_buffer:
        print(
            f"[WARNING] [LOW DISK] Available: {free_bytes / (1024**3):.1f}GB | "
            f"Needed: {needed_with_buffer / (1024**3):.1f}GB"
        )
        return False
    return True


def download_and_extract_dataset(ds_name, data_dir, source_ref=None):
    """Universal Acquisition Engine (v16.0 Nuclear)."""
    ds_path = os.path.join(data_dir, ds_name)
    if os.path.exists(ds_path):
        return True

    ref = source_ref
    if not ref:
        # Resolve via config fallback
        pass

    if not ref or "kaggle" in ref:
        return _handle_kaggle(ds_name, data_dir, ref)
    return False


def _handle_kaggle(ds_name, data_dir, ref):
    """Atomic Kaggle Recovery (v16.2 Nuclear)."""
    print(f"[DATA] Recovering {ds_name} from Kaggle Manifold...")
    try:
        kagglehub = load_kagglehub()
    except ImportError as exc:
        logger.error("Acquisition failed for %s: kagglehub is not installed (%s)", ds_name, exc)
        print(f"[ERROR] [DATA ERROR] Acquisition failed for {ds_name}: {exc}")
        return False
    # OSError covers file-system failures, shutil.Error and requests' network errors.
    acquisition_errors = kagglehub_errors()
    try:
        # The Kaggle account owning the manifolds comes from the shared ecosystem file.
        username = kaggle_owner()

        # Inject Kaggle authentication from local token to prevent 403 errors
        token_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".kaggle_token")
        if os.path.exists(token_path):
            with open(token_path, "r", encoding="utf-8") as f:
                token = f.read().strip()
            token = token.removeprefix("KGAT_")
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = token

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

        print(f"[OK] [DATA] {ds_name} successfully synchronized to manifold.")
        return True
    except acquisition_errors as e:
        logger.exception("Acquisition failed for %s", ds_name)
        print(f"[ERROR] [DATA ERROR] Acquisition failed for {ds_name}: {e}")
        return False
