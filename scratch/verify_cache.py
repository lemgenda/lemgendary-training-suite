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

print("--- I/O BENCHMARK: Class-Level Caching ---")
print("Initializing first instance (Cold Cache)...")
start_time = time.time()
ds1 = MultiTaskDataset(config, model_key=model_key, is_train=True)
cold_time = time.time() - start_time
print(f"Time taken (Cold): {cold_time:.4f}s")

print("\nInitializing second instance (Hot Cache)...")
start_time = time.time()
ds2 = MultiTaskDataset(config, model_key=model_key, is_train=True)
hot_time = time.time() - start_time
print(f"Time taken (Hot): {hot_time:.4f}s")

if hot_time < cold_time:
    speedup = cold_time / max(1e-6, hot_time)
    print(f"\nSUCCESS: Cache-aware initialization is {speedup:.1f}x faster!")
else:
    print("\nWARNING: Cache did not show significant speedup. (May be due to small dataset in test)")

# Verify that file counts match
if len(ds1.samples) == len(ds2.samples):
    print(f"PARITY CHECK: OK (Samples: {len(ds1.samples)})")
else:
    print(f"PARITY CHECK: FAILED! ({len(ds1.samples)} vs {len(ds2.samples)})")
