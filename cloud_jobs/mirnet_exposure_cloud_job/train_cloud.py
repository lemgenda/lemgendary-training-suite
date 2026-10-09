"""Automated Headless Kaggle Cloud Training Job for mirnet_exposure."""
import os
import subprocess
import sys

print("Initializing LemGendary Cloud Worker for mirnet_exposure...")
res = subprocess.run([sys.executable, "-m", "training.core_loop", "--model", "mirnet_exposure"], check=False)
sys.exit(res.returncode)
