import torch
import os

ckpt_path = r"c:\Development\python\model-training\lemgendary-training-suite\trained-models\checkpoints\nima_technical_best.pth"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"Epoch: {ckpt.get('epoch')}")
    print(f"Best Val Loss: {ckpt.get('best_val_loss')}")
    print(f"Best Quality Score: {ckpt.get('best_quality_score')}")
    # Extract keys to see if metrics are stored directly
    for k in ckpt.keys():
        if k not in ['model_state', 'optimizer_state', 'scheduler_state']:
            print(f"{k}: {ckpt[k]}")
else:
    print("Checkpoint not found")
