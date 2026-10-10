"""Automated Headless Kaggle Cloud Training Job for forex_predictor."""
import os
import subprocess
import sys

print("Initializing LemGendary Cloud Worker for forex_predictor...")
res = subprocess.run([sys.executable, "-m", "training.core_loop", "--model", "forex_predictor"], check=False)
sys.exit(res.returncode)
