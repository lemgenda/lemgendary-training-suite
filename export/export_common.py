"""Unified Export Helpers and Resilience Routines (Legacy Shim)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn
import yaml

from training.export.common import (
    resolve_checkpoint_file as canonical_resolve_checkpoint_file,
    resolve_export_paths as canonical_resolve_export_paths,
    wrap_quality_model,
)
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.export_common")


def init_export_environment() -> str:
    """Initialize environment and return repository root path."""
    sys.setrecursionlimit(2000)
    root = str(get_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def load_export_configs(project_root: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load config.yaml and unified models registry."""
    root = Path(project_root)
    config_path = root / "config.yaml"
    if not config_path.exists():
        logger.error("config.yaml not found at %s", config_path)
        return None, None

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    unified_name = config.get("unified_models", "unified_models_v2.yaml")
    reg_path = root / unified_name
    if not reg_path.exists():
        reg_path = root / "unified_models.yaml"

    if not reg_path.exists():
        logger.error("Unified models YAML not found at %s", reg_path)
        return None, None

    registry = yaml.safe_load(reg_path.read_text(encoding="utf-8")) or {}
    return config, registry


def resolve_export_paths(model_key: str, model_info: dict[str, Any], config: dict[str, Any], project_root: str) -> tuple[str, str]:
    """Compute base artifact name and production directory."""
    base_name, prod_dir = canonical_resolve_export_paths(
        model_key=model_key,
        model_info=model_info,
        config=config,
        project_root=Path(project_root),
    )
    return base_name, str(prod_dir)


def resolve_checkpoint_file(
    model_key: str,
    production_dir: str,
    config: dict[str, Any],
    project_root: str,
    user_checkpoint: str | None = None,
) -> str | None:
    """Locate the best, latest, or progress checkpoint file."""
    res = canonical_resolve_checkpoint_file(
        model_key=model_key,
        production_dir=Path(production_dir),
        config=config,
        project_root=Path(project_root),
        user_checkpoint=user_checkpoint,
    )
    return str(res) if res else None


def build_export_model(model_key: str, config: dict[str, Any], device: torch.device = torch.device("cpu")) -> nn.Module | None:
    """Instantiate model architecture on designated device."""
    try:
        from models.factory import get_model
        return get_model(model_key, config).to(device)
    except Exception as err:
        logger.error("Error during instantiation: %s", err)
        return None


__all__ = [
    "build_export_model",
    "init_export_environment",
    "load_export_configs",
    "resolve_checkpoint_file",
    "resolve_export_paths",
    "wrap_quality_model",
]
