"""
Standalone Judicial Audit CLI / API Wrapper

This module provides a lightweight, decoupled validation loader for evaluating
compiled PyTorch (.pth) and ONNX (.onnx) computer vision models (such as NIMA).
It exports evaluation metrics (PLCC, SRCC) into standardized JSON schemas,
suitable for CI/CD pipeline integration and automated edge-testing.

Designed to operate independently of the primary `lemgendary-training-suite` infrastructure.
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.stats
from PIL import Image

# Parse arguments
parser = argparse.ArgumentParser(description="Standalone Judicial Audit CLI")
parser.add_argument("--model_path", type=str, required=True, help="Path to .pth or .onnx model")
parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing images")
parser.add_argument("--labels_csv", type=str, required=True, help="Path to CSV with ground truth labels. Format: filename,prob1,...,prob10 or filename,mean_score")
parser.add_argument("--output_json", type=str, required=True, help="Path to export JSON metrics")
parser.add_argument("--model_type", type=str, default="nima_aesthetic_mobile", help="Model architecture type (only used for .pth)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
args = parser.parse_args()

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn

class SoftmaxWrapper(nn.Module):
    """
    A lightweight PyTorch module wrapper to resolve Unpickling errors from legacy
    exported checkpoints. This mirrors the `SoftmaxWrapper` structure defined in
    the main export pipelines, ensuring `torch.load` can seamlessly instantiate
    full model objects containing this wrapper.
    """
    def __init__(self, inner_model, temperature=1.0):
        super().__init__()
        self.inner_model = inner_model
        self.temperature = temperature
    def forward(self, x):
        logits = self.inner_model(x)
        return torch.nn.functional.softmax(logits / self.temperature, dim=1)

class SimpleDataset(torch.utils.data.Dataset):
    """
    Minimal, zero-dependency PyTorch Dataset using PIL and torchvision.transforms.
    Reads ground truth target scores from a CSV file. It bypasses the complex,
    multi-process augmentations used during full model training for rapid auditing.
    """
    def __init__(self, dataset_dir, labels_csv):
        self.dataset_dir = Path(dataset_dir)
        self.samples = []
        
        with open(labels_csv, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            
            is_prob_distribution = len(header) > 2
            for row in reader:
                filename = row[0]
                if is_prob_distribution:
                    # 10 probabilities
                    probs = [float(x) for x in row[1:11]]
                    # calculate mean
                    mean_score = sum(p * (i + 1) for i, p in enumerate(probs))
                else:
                    mean_score = float(row[1])
                self.samples.append((filename, mean_score))
                
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, target_mean = self.samples[idx]
        img_path = self.dataset_dir / filename
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, target_mean

def load_pytorch_model(model_path, model_type, device):
    """
    Safely load a PyTorch model from a .pth checkpoint or full model export.
    
    This function gracefully handles unpickling of standalone model objects and
    raw state dictionaries, accommodating PyTorch 2.6+ security paradigms
    (weights_only=False).
    
    Args:
        model_path (str): Path to the PyTorch checkpoint (.pt, .pth).
        model_type (str): Key identifying the model architecture (e.g. 'nima_aesthetic_mobile').
        device (str): Target hardware device ('cuda' or 'cpu').
        
    Returns:
        torch.nn.Module: The loaded, evaluation-ready PyTorch model.
    """
    print(f"[*] Loading PyTorch model ({model_type}) from {model_path}")
    if model_type == "nima_aesthetic_mobile" or "nima" in model_type.lower():
        try:
            from models.nima import NIMA_Model
        except ImportError:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from models.nima import NIMA_Model
        model = NIMA_Model()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
        
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict):
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # remove module. prefix if DDP
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        model = state_dict

    model.to(device)
    model.eval()
    return model

def main():
    """
    Main execution loop for the Judicial Audit CLI.
    
    1. Parses arguments to select the target model (.pth or .onnx) and dataset.
    2. Initializes the `SimpleDataset` without background workers to avoid Windows hang bugs.
    3. Triggers the relevant inference pipeline (PyTorch or ONNX Runtime).
    4. Dynamically casts inputs (FP16/FP32) and detects probability vs. logit distributions.
    5. Computes PLCC and SRCC coefficients via scipy.stats.
    6. Writes a formatted JSON report for pipeline verification.
    """
    print("--- LEMGENDARY JUDICIAL AUDIT ---")
    dataset = SimpleDataset(args.dataset_dir, args.labels_csv)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    is_onnx = args.model_path.endswith(".onnx")
    
    predictions = []
    targets = []
    
    if is_onnx:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            print("[ERROR] onnxruntime is not installed. Please install it via 'pip install onnxruntime' or 'onnxruntime-gpu'.")
            sys.exit(1)
            
        print(f"[*] Loading ONNX model from {args.model_path}")
        available_providers = ort.get_available_providers()
        if args.device == 'cuda' and 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider']
        elif args.device == 'cuda' and 'DmlExecutionProvider' in available_providers:
            providers = ['DmlExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        ort_session = ort.InferenceSession(args.model_path, providers=providers)
        input_name = ort_session.get_inputs()[0].name
        input_type = ort_session.get_inputs()[0].type
        
        print("[*] Running ONNX correlation probes...")
        for batch_imgs, batch_targets in dataloader:
            batch_np = batch_imgs.numpy()
            if 'float16' in input_type:
                batch_np = batch_np.astype(np.float16)
            ort_outs = ort_session.run(None, {input_name: batch_np})
            raw_out = torch.tensor(ort_outs[0])
            
            # Check if output is already probabilities (exported model) or logits (raw checkpoint)
            if raw_out.min() >= 0 and raw_out.max() <= 1.0001 and torch.isclose(raw_out.sum(dim=-1), torch.tensor(1.0), atol=1e-3).all():
                probs = raw_out
            else:
                probs = F.softmax(raw_out.clamp(min=-15.0, max=15.0) / 1.0, dim=-1)
                
            weights = torch.arange(1, 11).float()
            p_mean = (probs * weights).sum(dim=-1).numpy()
            predictions.extend(p_mean)
            targets.extend(batch_targets.numpy())
    else:
        model = load_pytorch_model(args.model_path, args.model_type, args.device)
        print("[*] Running PyTorch correlation probes...")
        with torch.no_grad():
            for batch_imgs, batch_targets in dataloader:
                batch_imgs = batch_imgs.to(args.device)
                raw_out = model(batch_imgs)
                
                if raw_out.min() >= 0 and raw_out.max() <= 1.0001 and torch.isclose(raw_out.sum(dim=-1), torch.tensor(1.0), atol=1e-3).all():
                    probs = raw_out
                else:
                    probs = F.softmax(raw_out.clamp(min=-15.0, max=15.0) / 1.0, dim=-1)
                    
                weights = torch.arange(1, 11).float().to(args.device)
                p_mean = (probs * weights).sum(dim=-1).cpu().numpy()
                predictions.extend(p_mean)
                targets.extend(batch_targets.numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)
    
    plcc, _ = scipy.stats.pearsonr(predictions, targets)
    srcc, _ = scipy.stats.spearmanr(predictions, targets)
    
    if np.isnan(plcc): plcc = 0.0
    if np.isnan(srcc): srcc = 0.0
    
    print(f"\n[RESULTS] Samples Evaluated: {len(targets)}")
    print(f"[RESULTS] PLCC: {plcc:.4f} | SRCC: {srcc:.4f}")
    
    output_data = {
        "model_path": str(Path(args.model_path).resolve()),
        "backend": "ONNX" if is_onnx else "PyTorch",
        "model_type": args.model_type if not is_onnx else "Unknown",
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "samples_evaluated": len(targets),
        "metrics": {
            "PLCC": round(float(plcc), 4),
            "SRCC": round(float(srcc), 4)
        }
    }
    
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"[*] JSON export successfully written to {args.output_json}")

if __name__ == "__main__":
    main()
