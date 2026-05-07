import os
import threading
import subprocess
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime

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
        self.hub_user = "lemgenda"
        self.hub_repo = "lemgendary-pretrained-models"
        self.hub_root = Path("/kaggle/working/LemGendaryModels") if os.name != 'nt' else Path(config.get("export_dir", "../LemGendaryModels")).resolve()

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
        if not self.pat:
            print("⚠️ [SYNC] GITHUB_PAT missing. Skipping autonomous manifold push.")
            return

        print(f"\n📡 [CLOUD SYNC] Manifold Synchronization Phase (Epoch {self.epoch})...")
        
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
