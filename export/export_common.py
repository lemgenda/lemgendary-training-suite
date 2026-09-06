"""Unified Export Helpers and Resilience Routines for LemGendary Models.

Provides common configuration resolution, architecture instantiation,
checkpoint discovery, and production asset wrapping for ONNX and Torch exporters.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn


def init_export_environment():
    """Initializes stdout/stderr UTF-8 encoding, repository root path, and recursion limit."""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
    sys.setrecursionlimit(2000)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(current_dir)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def load_export_configs(project_root: str):
    """Loads config.yaml and the unified models registry."""
    config_path = os.path.join(project_root, "config.yaml")
    if not os.path.exists(config_path):
        print(f" Error: config.yaml not found at {config_path}")
        print("[REMEDY] Ensure your model folder contains a valid config.yaml file.")
        return None, None

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    unified_models_name = config.get("unified_models", "unified_models_v2.yaml")
    unified_models_path = os.path.join(project_root, unified_models_name)
    if not os.path.exists(unified_models_path):
        unified_models_path = os.path.join(project_root, "unified_models.yaml")
    if not os.path.exists(unified_models_path):
        print(" Error: Unified models YAML not found.")
        print("[REMEDY] Run this script from the root of the repository where unified_models.yaml is located.")
        return None, None

    with open(unified_models_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    return config, registry


def resolve_export_paths(model_key: str, model_info: dict, config: dict, project_root: str):
    """Computes base artifact name and production directory."""
    model_filename = model_info.get("filename", model_key)
    base_name = f"LemGendary{model_filename}"
    production_dir_rel = os.path.join(config.get("export_dir", "../LemGendaryModels"), model_key)
    production_dir = os.path.normpath(os.path.join(project_root, production_dir_rel))
    os.makedirs(production_dir, exist_ok=True)
    return base_name, production_dir


def resolve_checkpoint_file(model_key: str, production_dir: str, config: dict, project_root: str, user_checkpoint: str | None = None):
    """Locates the best, latest, or progress checkpoint file."""
    if user_checkpoint:
        resolved = os.path.normpath(user_checkpoint)
        if os.path.exists(resolved):
            return resolved

    hub_ckpt_dir = os.path.join(production_dir, "checkpoints")
    for suffix in ("_best.pth", "_latest.pth", "_progress.pth"):
        candidate = os.path.join(hub_ckpt_dir, f"{model_key}{suffix}")
        if os.path.exists(candidate):
            return candidate

    legacy_dir_rel = config.get("checkpoint_dir", "checkpoints")
    legacy_dir = os.path.normpath(os.path.join(project_root, legacy_dir_rel))
    fallback = os.path.join(legacy_dir, f"{model_key}_best.pth")
    if os.path.exists(fallback):
        return fallback

    return None


def build_export_model(model_key: str, config: dict, device: torch.device = torch.device("cpu")):
    """Instantiates the PyTorch model on the designated device (default CPU)."""
    print(f" [ARCH] Instantiating architecture for {model_key} on {device}...")
    try:
        from models.factory import get_model
        return get_model(model_key, config).to(device)
    except Exception as err:
        print(f" Error during instantiation: {err}")
        return None


def wrap_quality_model(model: nn.Module, model_info: dict):
    """Applies softmax temperature wrapper for quality scoring models."""
    if model_info.get("dataset_type") == "quality":
        from models.nima import SoftmaxWrapper
        temp = getattr(model, "softmax_temp", torch.tensor(1.0)).item()
        if temp == 1.0:
            stab = model_info.get("stabilizers", {})
            temp = stab.get("softmax_temp", 1.0)
        print(f" [WRAP] Applying production Softmax wrapper with Temp={temp}")
        model = SoftmaxWrapper(model, temperature=temp)
        model.eval()
    return model
