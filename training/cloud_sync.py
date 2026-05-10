import os
import threading
import subprocess
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
import kagglehub

# [SENIOR HARDENING v16.0 - SYNC_ID: 1042]

class CloudSyncManager:
    """
    Nuclear-Hardened SOTA Synchronizer (v16.0).
    Handles atomic Git-LFS pushes with rebase-resilience and metric-merge protection.
    """
    def __init__(self, model_name, epoch, config):
        self.model_name = model_name
        self.epoch = epoch
        self.config = config
        self.pat = os.environ.get("GITHUB_PAT", "")
        self.hub_user = config.get("hub_user", "lemgenda")
        self.hub_repo = "lemgendary-pretrained-models"
        self.hub_root = Path("/kaggle/working/LemGendaryModels") if os.name != 'nt' else Path(config.get("export_dir", "../LemGendaryModels")).resolve()
        
        # --- Kaggle Hybrid Configuration ---
        # 2026 Resilience: Ensure we use the configured username (defaults to lemtreursi)
        # to avoid 'models.create' permission denied errors on the 'lemgenda' org.
        self.kaggle_username = config.get("kaggle_username", "lemtreursi")
        
        # 2026: Standardized Kaggle Handle construction
        model_slug = self.model_name.replace("_", "-")
        slug_prefix = config.get("kaggle_slug_prefix", "lemgendary-")
        slug_suffix = config.get("kaggle_slug_suffix", "-checkpoints")
        
        # Enforce absolute parity with user identity
        self.kaggle_handle = f"{self.kaggle_username}/{slug_prefix}{model_slug}{slug_suffix}/pytorch/default"

        
        # Paths
        # 2026 Resilience: SOTA checkpoints and metrics are stored within the Model Hub manifold.
        # We sync the model's root directory to capture both metrics.csv and the checkpoints folder.
        self.model_dir = self.hub_root / self.model_name
        self.checkpoint_dir = self.model_dir / "checkpoints"

    def _mask_pat(self, text):
        if not self.pat: return text
        return text.replace(self.pat, "***STEALTH***")

    def _run_git(self, args, cwd=None):
        """Atomic Git Runner with Stealth Masking."""
        try:
            res = subprocess.run(['git'] + args, cwd=cwd or self.hub_root, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"❌ [GIT ERROR] {self._mask_pat(res.stderr)}")
                return False
            return True
        except Exception as e:
            print(f"❌ [SYS ERROR] {e}")
            return False

    def sync(self):
        """Unified Hybrid Sync: Kaggle (SOTA Source of Truth)"""
        print(f"\n📡 [CLOUD SYNC] Cloud Synchronization Phase (Epoch {self.epoch})...")
        
        # 2026: On Kaggle, we bypass GitHub entirely to keep the manifold lean and PAT-free.
        # Checkpoints and metrics.csv are pushed DIRECTLY to the Kaggle Model artifact.
        is_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_WORKING_DIR")
        
        if is_kaggle:
            print("🚀 [KAGGLER] Operating in Kaggle-Native mode. GitHub sync bypassed.")
            self._sync_to_kaggle()
        else:
            # Local/Custom environment: Use legacy hybrid sync
            print("🔄 [HYBRID] Operating in Hybrid mode. Syncing to GitHub and Kaggle.")
            self._sync_to_github()
            self._sync_to_kaggle()

    def _sync_to_kaggle(self):
        """Programmatic SOTA Checkpoint Sync to Kaggle Hub."""
        if not self.checkpoint_dir.exists():
            return

        # We only upload if there are actually checkpoint files to sync
        ckpts = list(self.checkpoint_dir.glob(f"{self.model_name}_*.pth"))
        if not ckpts:
            print(f"📦 [KAGGLER] No checkpoints found for {self.model_name} in {self.checkpoint_dir}. Skipping.")
            return

        try:
            print(f"🚀 [KAGGLER] Syncing SOTA Manifold to Kaggle: {self.kaggle_handle}...")
            # 2026 Resilience: Run upload in a subprocess to isolate verbose/crashing I/O
            import subprocess
            import sys
            # 2026 Resilience: Kernel Format Sentinel
            # Enforce rich-text .ipynb standard by purging raw scripts from the documentation manifold.
            for py_file in self.model_dir.glob("*.py"):
                if "training" in py_file.name or "usage" in py_file.name:
                    print(f"🗑️ [JANITOR] Purging raw script artifact: {py_file.name}")
                    py_file.unlink()

            # Escape backslashes for the script string
            safe_model_dir = str(self.model_dir).replace('\\', '/')

            upload_script = f"""
import kagglehub, os, sys
try:
    kagglehub.model_upload(
        handle='{self.kaggle_handle}',
        local_model_dir='{safe_model_dir}',
        version_notes='SOTA Update: {self.model_name} | Epoch {self.epoch}'
    )
    sys.exit(0)
except Exception as e:
    # We print to stdout so it's captured in the devnull or log if we ever enable it
    print(e)
    sys.exit(1)
"""
            with open(os.devnull, 'w') as devnull:
                res = subprocess.run(
                    [sys.executable, "-c", upload_script],
                    stdout=devnull,
                    stderr=devnull,
                    env=os.environ.copy()
                )
            
            if res.returncode == 0:
                print(f"✅ [KAGGLER] Manifold successfully synchronized to Kaggle Hub!")
            else:
                print(f"⚠️ [KAGGLER] Hub Sync subprocess returned error code {res.returncode}")
        except Exception as e:
            print(f"⚠️ [KAGGLER] Hub Sync failed: {e}")

    def _sync_to_github(self):
        """Legacy Git-LFS Sync for Metrics and Documentation."""
        # 2026 Resilience: Only enforce PAT if in a headless cloud environment.
        # On local, we use the linked account.
        is_cloud = os.environ.get("KAGGLE_WORKING_DIR") or "/kaggle/working" in str(self.hub_root)
        if not self.pat and is_cloud:
            print("⚠️ [SYNC] GITHUB_PAT missing in Cloud environment. Skipping push.")
            return
        
        # 1. Ensure Hub is initialized and anchored
        if not (self.hub_root / ".git").exists():
            print("🛸 [SYNC] Hub not initialized. Readying for first-contact...")
            return

        # 2. Metric Merge-Persistence (Task 8.2)
        # We read the remote state before adding local changes to prevent history loss
        metrics_file = self.hub_root / self.model_name / "metrics.csv"
        if metrics_file.exists():
            # Force a rebase to get the latest remote metrics
            self._run_git(['fetch', 'origin'])
            self._run_git(['rebase', 'origin/main'])
            
        # 3. Finalize Local Manifest
        # The train.py already exported files to the hub_root directory.
        # We just need to stage and push.
        
        # 4. Atomic Push with Rebase-Loop (Task 8.1)
        self._run_git(['config', 'user.email', 'lem.treursic@gmail.com'])
        self._run_git(['config', 'user.name', 'lemgenda'])
        self._run_git(['lfs', 'install'])
        self._run_git(['add', '.'])
        
        # Check if there are changes to commit
        check = subprocess.run(['git', 'diff-index', '--quiet', 'HEAD', '--'], cwd=self.hub_root)
        if check.returncode != 0:
            commit_msg = f"SOTA Update: {self.model_name} | Epoch {self.epoch} | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            self._run_git(['commit', '-m', commit_msg])
            
            # Rebase-Loop to handle concurrent pushes from other kernels
            for attempt in range(3):
                print(f"🚀 [SYNC] Attempting Atomic Push (Try {attempt+1}/3)...")
                if self._run_git(['push', 'origin', 'main']):
                    print(f"✅ [SYNC] Manifold successfully synchronized to Hub!")
                    return
                else:
                    print("🔄 [SYNC] Collision detected. Synchronizing remote manifold...")
                    self._run_git(['pull', '--rebase', '-X', 'theirs', 'origin', 'main'])
            
            print("❌ [SYNC] Exhausted push attempts. Manifold out of sync.")
        else:
            print("✅ [SYNC] Everything up-to-date. No manifold drift detected.")

def trigger_cloud_sync(model_name, epoch, config):
    """Entry point for training loop to trigger background sync."""
    manager = CloudSyncManager(model_name, epoch, config)
    t = threading.Thread(target=manager.sync, daemon=True)
    t.start()
