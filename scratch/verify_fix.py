import os
import sys
import torch
import time
import yaml

# Anchor path to project root
project_root = os.getcwd()
sys.path.insert(0, project_root)

from data.dataset import MultiTaskDataset

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_key = "nafnet_denoising"

print("Initializing dataset...")
ds = MultiTaskDataset(config, model_key=model_key, is_train=True)

print(f"Dataset size: {len(ds)}")

# We'll skip fewer items if the dataset is small, but 100 should be enough to see the difference
num_items = min(100, len(ds))

print(f"Simulating skip of {num_items} items with sync_mode = False (Normal Loading)...")
start_time = time.time()
for i in range(num_items):
    _ = ds[i]
normal_time = time.time() - start_time
print(f"Time taken: {normal_time:.4f}s")

print(f"Simulating skip of {num_items} items with sync_mode = True (Fast Skip)...")
ds.sync_mode = True
start_time = time.time()
for i in range(num_items):
    _ = ds[i]
fast_time = time.time() - start_time
print(f"Time taken: {fast_time:.4f}s")

if fast_time < normal_time:
    speedup = normal_time / max(1e-6, fast_time)
    print(f"SUCCESS: Fast skip is {speedup:.1f}x faster!")
else:
    print("WARNING: Fast skip did not show significant speedup in this small test.")
