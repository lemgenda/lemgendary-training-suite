"""Common export helpers, path resolution, and model wrapping."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.export.common")


class ExportError(Exception):
    """Raised when model export fails."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def resolve_export_paths(
    model_key: str,
    model_info: dict[str, Any],
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> tuple[str, Path]:
    """Compute base artifact name and production destination directory."""
    root = project_root or get_project_root()
    cfg = config or {}

    model_filename = model_info.get("filename", model_key)
    base_name = f"LemGendary{model_filename}"

    export_dir_rel = cfg.get("export_dir", "../LemGendaryModels")
    production_dir = (root / export_dir_rel / model_key).resolve()
    production_dir.mkdir(parents=True, exist_ok=True)

    return base_name, production_dir


def resolve_checkpoint_file(
    model_key: str,
    production_dir: Path,
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
    user_checkpoint: str | Path | None = None,
) -> Path | None:
    """Locate the best, latest, or user-specified .pth checkpoint file."""
    if user_checkpoint:
        cand = Path(user_checkpoint).resolve()
        if cand.exists() and cand.is_file():
            return cand

    hub_ckpt_dir = production_dir / "checkpoints"
    for suffix in ("_best.pth", "_latest.pth", "_progress.pth"):
        cand = hub_ckpt_dir / f"{model_key}{suffix}"
        if cand.exists() and cand.is_file():
            return cand

    root = project_root or get_project_root()
    cfg = config or {}
    legacy_dir = (root / cfg.get("checkpoint_dir", "checkpoints")).resolve()
    legacy_candidate = legacy_dir / f"{model_key}_best.pth"
    if legacy_candidate.exists() and legacy_candidate.is_file():
        return legacy_candidate

    return None


def extract_input_dimensions(
    model_info: dict[str, Any],
    default_size: int = 256,
) -> tuple[int, int]:
    """Extract height and width from model_info configuration."""
    size_raw = model_info.get("input_size", default_size)
    if size_raw is None:
        size_raw = default_size

    if isinstance(size_raw, (list, tuple)):
        if len(size_raw) == 3:
            h, w = int(size_raw[1]), int(size_raw[2])
        elif len(size_raw) >= 2:
            h, w = int(size_raw[0]), int(size_raw[1])
        else:
            h = w = int(size_raw[0])
    else:
        h = w = int(size_raw)

    return min(h, 512), min(w, 512)


def wrap_quality_model(model: nn.Module, model_info: dict[str, Any]) -> nn.Module:
    """Applies softmax temperature wrapper for quality scoring models if needed."""
    if model_info.get("dataset_type") == "quality":
        try:
            from models.nima import SoftmaxWrapper
            temp_raw = getattr(model, "softmax_temp", torch.tensor(1.0))
            temp = float(temp_raw.item() if isinstance(temp_raw, torch.Tensor) else temp_raw)
            if temp == 1.0:
                stab = model_info.get("stabilizers", {})
                temp = float(stab.get("softmax_temp", 1.0))
            logger.info("Applying production SoftmaxWrapper with temperature=%.2f", temp)
            wrapped = SoftmaxWrapper(model, temperature=temp)
            wrapped.eval()
            return wrapped
        except (ImportError, Exception) as exc:
            logger.debug("SoftmaxWrapper unavailable or unnecessary: %s", exc)
    return model
