"""Model export subsystem providing unified ONNX, TorchScript, WebGPU, and MT5 pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from training.export.common import (
    ExportError,
    extract_input_dimensions,
    resolve_checkpoint_file,
    resolve_export_paths,
    wrap_quality_model,
)
from training.export.mt5_signal import export_forex_onnx
from training.export.onnx import export_onnx
from training.export.torch_standalone import export_torch_standalone
from training.export.webgpu import export_webgpu_onnx
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.export")


def export_all(
    model_key: str,
    checkpoint_path: Path | str | None = None,
    config: dict[str, Any] | None = None,
    output_dir: Path | str | None = None,
    targets: list[str] | None = None,
) -> dict[str, Path]:
    """Orchestrate full production export across specified targets.

    Targets can include: "onnx_fp32", "onnx_fp16", "torch_pt", "webgpu", "forex".
    Defaults to ["onnx_fp32", "onnx_fp16", "torch_pt"].
    """
    project_root = get_project_root()
    cfg = config or {}

    if not cfg:
        cfg_path = project_root / "config.yaml"
        if cfg_path.exists():
            try:
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError) as e:
                logger.warning("Could not read config.yaml: %s", e)

    unified_models_name = cfg.get("unified_models", "unified_models_v2.yaml")
    registry_path = project_root / unified_models_name
    if not registry_path.exists():
        registry_path = project_root / "unified_models.yaml"

    registry: dict[str, Any] = {}
    if registry_path.exists():
        try:
            registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as e:
            logger.warning("Could not read unified models registry: %s", e)

    model_info = registry.get(model_key, {})
    base_name, default_out_dir = resolve_export_paths(model_key, model_info, cfg, project_root)
    destination_dir = Path(output_dir).resolve() if output_dir else default_out_dir
    destination_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = resolve_checkpoint_file(
        model_key=model_key,
        production_dir=destination_dir,
        config=cfg,
        project_root=project_root,
        user_checkpoint=checkpoint_path,
    )

    is_forex = model_info.get("dataset_type") == "forex" or "forex" in model_key.lower()
    export_targets = targets or (["forex"] if is_forex else ["onnx_fp32", "onnx_fp16", "torch_pt"])

    results: dict[str, Path] = {}

    if is_forex or "forex" in export_targets:
        if not ckpt_file:
            raise ExportError("CKPT_MISSING", f"No checkpoint found for forex model {model_key}")
        forex_out = destination_dir / f"{base_name}.onnx"
        active_tfs = model_info.get("kwargs", {}).get("active_timeframes", [1, 5, 15, 60, 240, 1440])
        exported_path = export_forex_onnx(ckpt_file, forex_out, active_timeframes=active_tfs)
        results["forex"] = exported_path
        return results

    # Instantiate model architecture
    from models.factory import get_model
    model = get_model(model_key, cfg).to(torch.device("cpu"))

    if ckpt_file and ckpt_file.exists():
        logger.info("Loading weights from checkpoint: %s", ckpt_file)
        ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        raw_state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        state_dict = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in raw_state.items()
        }
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    model = wrap_quality_model(model, model_info)

    h, w = extract_input_dimensions(model_info)
    dummy_shape = (1, 3, h, w)

    if "onnx_fp32" in export_targets:
        fp32_path = destination_dir / f"{base_name}_FP32.onnx"
        export_onnx(model, fp32_path, dummy_shape=dummy_shape, half=False)
        results["onnx_fp32"] = fp32_path

    if "onnx_fp16" in export_targets:
        fp16_path = destination_dir / f"{base_name}.onnx"
        export_onnx(model, fp16_path, dummy_shape=dummy_shape, half=True)
        results["onnx_fp16"] = fp16_path

    if "torch_pt" in export_targets:
        pt_path = destination_dir / f"{base_name}.pt"
        export_torch_standalone(model, pt_path, metadata={"model_key": model_key})
        results["torch_pt"] = pt_path

    if "webgpu" in export_targets:
        webgpu_path = destination_dir / f"{base_name}_webgpu.onnx"
        export_webgpu_onnx(model, webgpu_path, dummy_input_shape=(1, 3, 512, 512))
        results["webgpu"] = webgpu_path

    logger.info("Completed export_all for %s: %d targets exported.", model_key, len(results))
    return results


__all__ = [
    "ExportError",
    "export_all",
    "export_forex_onnx",
    "export_onnx",
    "export_torch_standalone",
    "export_webgpu_onnx",
    "extract_input_dimensions",
    "resolve_checkpoint_file",
    "resolve_export_paths",
    "wrap_quality_model",
]
