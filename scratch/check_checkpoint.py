import torch
checkpoint_path = r'c:\Development\python\model-training\lemgendary-training-suite\trained-models\checkpoints\nafnet_denoising_latest.pth'
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Iteration: {checkpoint.get('iteration', 'N/A')}")
    print(f"Best Quality Score: {checkpoint.get('best_quality_score', 'N/A')}")
    print(f"SOTA Achieved: {checkpoint.get('sota_achieved', 'N/A')}")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
