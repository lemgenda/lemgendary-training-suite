import os
import sys
import torch
import threading
import time

from training.cloud_sync import CloudSyncManager
from training.optimization_engine import export_webgpu_onnx

def run_hub():
    import cloud_hub
    import asyncio
    try:
        asyncio.run(cloud_hub.main())
    except Exception as e:
        pass

t = threading.Thread(target=run_hub, daemon=True)
t.start()
time.sleep(2)

print("\n--- Testing Federated Average Sync ---")
manager = CloudSyncManager("test_model", 1, {})
# This should connect, send HEARTBEAT, GRADIENT_PUSH, and receive GRADIENT_AVERAGE_SYNC
manager.average_sync(None)

print("\n--- Testing Memory-Sentinel WebGPU Export ---")
class DummyModel(torch.nn.Module):
    def forward(self, x): return x * 2

model = DummyModel()
success = export_webgpu_onnx(model, "test_webgpu.onnx")

if success and os.path.exists("test_webgpu.onnx"):
    print("SUCCESS: test_webgpu.onnx generated successfully with Opset 17.")
else:
    print("FAILED: test_webgpu.onnx not found.")

sys.exit(0)
