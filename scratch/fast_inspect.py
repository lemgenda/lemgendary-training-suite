import torch
import os

ckpt_path = r"c:\Development\python\model-training\lemgendary-training-suite\trained-models\checkpoints\nima_technical_best.pth"
if os.path.exists(ckpt_path):
    # Load only metadata to be fast
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    meta = {k: v for k, v in ckpt.items() if k not in ['model_state', 'optimizer_state', 'scheduler_state']}
    print(meta)
else:
    print("Checkpoint not found")
