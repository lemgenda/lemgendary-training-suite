"""System and Model Topology Audit Service for LemGendary Model Training Suite.

Performs hardware profiling, parameter audits, VRAM headroom checks, and judicial
correlation verification.
"""

from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any
import torch
import yaml

from tools.judicial_audit import run_judicial_audit
from training.checkpoint.manager import audit_disk_space
from training.hardware.discovery import discover_device
from models.factory import get_model
from training.utils.paths import get_project_root


class AuditService:
    """Audits system hardware capabilities, model topologies, and checkpoint fidelity."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or get_project_root()

    def audit_system(self) -> dict[str, Any]:
        """Audit physical compute resources, VRAM headroom, and disk boundaries.

        Returns:
            dict[str, Any]: System audit dictionary.
        """
        device_info = discover_device()
        free_gb = audit_disk_space(target_dir=self.project_root, min_free_gb=5.0)
        disk_ok = free_gb >= 5.0

        vram_allocated_gb = 0.0
        vram_reserved_gb = 0.0
        vram_total_gb = 0.0

        if device_info.is_cuda:
            vram_allocated_gb = round(torch.cuda.memory_allocated() / (1024 ** 3), 3)
            vram_reserved_gb = round(torch.cuda.memory_reserved() / (1024 ** 3), 3)
            device_props = torch.cuda.get_device_properties(0)
            vram_total_gb = round(device_props.total_memory / (1024 ** 3), 3)

        primary_device_name = device_info.device_names[0] if device_info.device_names else str(device_info.device)

        return {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": primary_device_name,
            "device_type": device_info.device_type,
            "gpu_count": device_info.device_count if device_info.is_cuda else 0,
            "vram_total_gb": vram_total_gb,
            "vram_allocated_gb": vram_allocated_gb,
            "vram_reserved_gb": vram_reserved_gb,
            "cpu_threads": os.cpu_count() or 1,
            "disk_free_gb": round(free_gb, 2),
            "disk_healthy": disk_ok,
        }

    def audit_model(self, model_key: str) -> dict[str, Any]:
        """Audit model parameters, architecture, and estimated memory footprint.

        Args:
            model_key: Target model key registered in unified models.

        Returns:
            dict[str, Any]: Model topology report.
        """
        config_path = self.project_root / "config.yaml"
        config: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        models_name = config.get("unified_models", "unified_models_v2.yaml")
        models_path = self.project_root / models_name
        with open(models_path, "r", encoding="utf-8") as f:
            registry = yaml.safe_load(f) or {}

        models_dict = registry.get("models")
        if not isinstance(models_dict, dict):
            models_dict = registry
        model_info = models_dict.get(model_key, {})
        if not model_info:
            raise KeyError(f"Model key '{model_key}' not found in unified models registry.")

        raw_model = get_model(model_key, model_info)
        total_params = sum(p.numel() for p in raw_model.parameters())
        trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)

        # Estimate model weights in FP32 (4 bytes per param)
        fp32_size_mb = round((total_params * 4) / (1024 * 1024), 2)
        fp16_size_mb = round((total_params * 2) / (1024 * 1024), 2)

        return {
            "model_key": model_key,
            "class_name": raw_model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "estimated_fp32_mb": fp32_size_mb,
            "estimated_fp16_mb": fp16_size_mb,
            "target_resolution": model_info.get("resolution"),
            "task_type": model_info.get("type", "unknown"),
        }

    def audit_judicial(
        self,
        model_path: Path | str,
        dataset_dir: Path | str,
        labels_csv: Path | str,
        output_json: Path | str | None = None,
        model_type: str = "nima_aesthetic_mobile",
        batch_size: int = 32,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Execute judicial audit pipeline computing PLCC and SRCC correlations.

        Args:
            model_path: Path to model checkpoint or ONNX file.
            dataset_dir: Directory containing evaluation images.
            labels_csv: Ground truth label CSV.
            output_json: Optional destination to save output JSON.
            model_type: Architecture type string.
            batch_size: Evaluation batch size.
            device: Target execution device ('cuda' or 'cpu').

        Returns:
            dict[str, Any]: Correlation metric report.
        """
        return run_judicial_audit(
            model_path=model_path,
            dataset_dir=dataset_dir,
            labels_csv=labels_csv,
            output_json=output_json,
            model_type=model_type,
            batch_size=batch_size,
            device=device,
        )
