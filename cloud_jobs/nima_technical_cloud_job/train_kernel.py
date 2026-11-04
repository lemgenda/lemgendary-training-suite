"""LemGendary headless cloud worker (auto-generated Kaggle kernel for nima_technical)."""

import os
import subprocess
import sys

REPO_URL = "https://github.com/lemgenda/lemgendary-training-suite.git"
WORK_DIR = "/kaggle/working/lemgendary-training-suite"
MODEL_KEY = "nima_technical"


def main() -> None:
    """Probe the GPU, clone the suite if needed, then run Kaggle training with auto-sync."""
    print("[OK] [CLOUD WORKER] Booting Kaggle High-VRAM GPU Environment...")

    gpu_probe = subprocess.run(["nvidia-smi"], check=False)
    if gpu_probe.returncode != 0:
        print(
            f"[WARNING] [CLOUD WORKER] nvidia-smi exited with {gpu_probe.returncode}; "
            "GPU may be unavailable."
        )

    if not os.path.exists(WORK_DIR):
        print("[CLOUD] Cloning Training Suite manifold...")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, WORK_DIR], check=True)

    os.chdir(WORK_DIR)
    sys.path.insert(0, WORK_DIR)

    print(f"[CLOUD] Launching SOTA Matrix for >> {MODEL_KEY} <<...")
    cmd = [
        sys.executable,
        "-m",
        "training.train",
        "--model",
        MODEL_KEY,
        "--env",
        "kaggle",
        "--auto_sync",
        "--yes",
    ]
    subprocess.run(cmd, check=True)
    print("[CLOUD] Training Execution Completed.")


if __name__ == "__main__":
    main()
