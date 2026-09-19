"""Standalone Judicial Audit CLI / Diagnostic Tool.

Evaluates compiled PyTorch (.pth) and ONNX (.onnx) models against validation datasets,
computing Pearson (PLCC) and Spearman (SRCC) rank correlation coefficients.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import scipy.stats
import torch
import torch.nn.functional as F
from torchvision import transforms


class SimpleDataset(torch.utils.data.Dataset):
    """Minimal PyTorch Dataset for evaluation."""

    def __init__(self, dataset_dir: str | Path, labels_csv: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.samples: list[tuple[str, float]] = []

        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

            is_prob_distribution = len(header) > 2
            for row in reader:
                if not row:
                    continue
                filename = row[0]
                if is_prob_distribution:
                    probs = [float(x) for x in row[1:11]]
                    mean_score = sum(p * (i + 1) for i, p in enumerate(probs))
                else:
                    mean_score = float(row[1])
                self.samples.append((filename, mean_score))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        filename, target_mean = self.samples[idx]
        img_path = self.dataset_dir / filename
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, target_mean


def load_pytorch_model(model_path: str | Path, model_type: str, device: str) -> torch.nn.Module:
    """Safely load a PyTorch model from a checkpoint.

    Args:
        model_path: Path to PyTorch checkpoint.
        model_type: Model architecture key.
        device: Target execution device.

    Returns:
        torch.nn.Module: Loaded model in eval mode.
    """
    if "nima" in model_type.lower():
        from models.nima import NIMA_Model
        model = NIMA_Model()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    state_dict = torch.load(str(model_path), map_location=device, weights_only=False)
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    elif isinstance(state_dict, torch.nn.Module):
        model = state_dict

    model.to(device)
    model.eval()
    return model


def run_judicial_audit(
    model_path: str | Path,
    dataset_dir: str | Path,
    labels_csv: str | Path,
    output_json: str | Path | None = None,
    model_type: str = "nima_aesthetic_mobile",
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, Any]:
    """Execute judicial audit pipeline and compute PLCC/SRCC correlation metrics.

    Args:
        model_path: Path to .pth or .onnx model.
        dataset_dir: Directory containing evaluation images.
        labels_csv: Path to CSV containing image names and ground truth scores.
        output_json: Optional destination to write JSON audit report.
        model_type: Model architecture key for PyTorch models.
        batch_size: Batch size for inference.
        device: Device to run evaluation on.

    Returns:
        dict[str, Any]: Audit results report.
    """
    path_str = str(model_path)
    is_onnx = path_str.endswith(".onnx")

    dataset = SimpleDataset(dataset_dir, labels_csv)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    predictions: list[float] = []
    targets: list[float] = []

    if is_onnx:
        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Install via pip to audit ONNX models."
            ) from exc

        available_providers = ort.get_available_providers()
        if device == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider"]
        elif device == "cuda" and "DmlExecutionProvider" in available_providers:
            providers = ["DmlExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        ort_session = ort.InferenceSession(path_str, providers=providers)
        input_name = ort_session.get_inputs()[0].name
        input_type = ort_session.get_inputs()[0].type

        for batch_imgs, batch_targets in dataloader:
            batch_np = batch_imgs.numpy()
            if "float16" in input_type:
                batch_np = batch_np.astype(np.float16)
            ort_outs = ort_session.run(None, {input_name: batch_np})
            raw_out = torch.tensor(ort_outs[0])

            if raw_out.min() >= 0 and raw_out.max() <= 1.0001 and torch.isclose(raw_out.sum(dim=-1), torch.tensor(1.0), atol=1e-3).all():
                probs = raw_out
            else:
                probs = F.softmax(raw_out.clamp(min=-15.0, max=15.0) / 1.0, dim=-1)

            weights = torch.arange(1, 11).float()
            p_mean = (probs * weights).sum(dim=-1).numpy().tolist()
            predictions.extend(p_mean)
            targets.extend(batch_targets.numpy().tolist())
    else:
        model = load_pytorch_model(model_path, model_type, device)
        with torch.no_grad():
            for batch_imgs, batch_targets in dataloader:
                batch_imgs = batch_imgs.to(device)
                raw_out = model(batch_imgs)

                if raw_out.min() >= 0 and raw_out.max() <= 1.0001 and torch.isclose(raw_out.sum(dim=-1), torch.tensor(1.0), atol=1e-3).all():
                    probs = raw_out
                else:
                    probs = F.softmax(raw_out.clamp(min=-15.0, max=15.0) / 1.0, dim=-1)

                weights = torch.arange(1, 11).float().to(device)
                p_mean = (probs * weights).sum(dim=-1).cpu().numpy().tolist()
                predictions.extend(p_mean)
                targets.extend(batch_targets.numpy().tolist())

    preds_arr = np.array(predictions)
    targets_arr = np.array(targets)

    if len(preds_arr) > 1 and len(targets_arr) > 1:
        plcc_res = scipy.stats.pearsonr(preds_arr, targets_arr)
        srcc_res = scipy.stats.spearmanr(preds_arr, targets_arr)
        plcc_val = float(getattr(plcc_res, "statistic", plcc_res[0]))
        srcc_val = float(getattr(srcc_res, "statistic", srcc_res[0]))
    else:
        plcc_val = 0.0
        srcc_val = 0.0

    if np.isnan(plcc_val):
        plcc_val = 0.0
    if np.isnan(srcc_val):
        srcc_val = 0.0

    output_data: dict[str, Any] = {
        "model_path": str(Path(model_path).resolve()),
        "backend": "ONNX" if is_onnx else "PyTorch",
        "model_type": model_type if not is_onnx else "Unknown",
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "samples_evaluated": len(targets),
        "metrics": {
            "PLCC": round(plcc_val, 4),
            "SRCC": round(srcc_val, 4),
        },
    }

    if output_json:
        out_path = Path(output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)

    return output_data


def main() -> None:
    """Entry point for command-line invocation."""
    parser = argparse.ArgumentParser(description="Standalone Judicial Audit CLI")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth or .onnx model")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--labels_csv", type=str, required=True, help="Path to CSV with ground truth labels")
    parser.add_argument("--output_json", type=str, required=True, help="Path to export JSON metrics")
    parser.add_argument("--model_type", type=str, default="nima_aesthetic_mobile", help="Model architecture type")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    args = parser.parse_args()

    results = run_judicial_audit(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        labels_csv=args.labels_csv,
        output_json=args.output_json,
        model_type=args.model_type,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"Judicial Audit complete: {results['samples_evaluated']} samples.")
    print(f"PLCC: {results['metrics']['PLCC']:.4f} | SRCC: {results['metrics']['SRCC']:.4f}")


if __name__ == "__main__":
    main()
