# LemGendary Headless Cloud Worker - Auto-Generated
import os
import sys
import subprocess

print("[OK] [CLOUD WORKER] Booting Kaggle High-VRAM GPU Environment...")

# Verify GPU
subprocess.run(["nvidia-smi"])

# Clone/Sync LemGendary Training Suite
repo_url = "https://github.com/lemgenda/lemgendary-training-suite.git"
work_dir = "/kaggle/working/lemgendary-training-suite"

if not os.path.exists(work_dir):
    print("[CLOUD] Cloning Training Suite manifold...")
    subprocess.run(["git", "clone", "--depth", "1", repo_url, work_dir])

os.chdir(work_dir)
sys.path.insert(0, work_dir)

# Launch Cloud Training with Auto-Sync
print("[CLOUD] Launching SOTA Matrix for >> nima_technical <<...")
cmd = [
    sys.executable, "-m", "training.train",
    "--model", "nima_technical",
    "--env", "kaggle",
    "--auto_sync",
    "--yes"
]
subprocess.run(cmd)
print("[CLOUD] Training Execution Completed.")
