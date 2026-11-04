"""Model export subsystem providing unified ONNX, TorchScript, WebGPU, and MT5 pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from models.factory import get_model
from training.export.base import BaseModelExporter
from training.export.common import (
    ExportError,
    OnnxGraphSpec,
    extract_input_dimensions,
    load_checkpoint_weights,
    resolve_checkpoint_file,
    resolve_export_paths,
    wrap_quality_model,
)
from training.export.exporter_factory import ExporterFactory, get_exporter_for_model
from training.export.forex_exporter import ForexModelExporter
from training.export.llm_exporter import LLMModelExporter, export_llm_tri_format
from training.export.mt5_signal import export_forex_onnx
from training.export.onnx import export_onnx
from training.export.pytorch_exporter import PyTorchModelExporter
from training.export.staging import StagingResult, VariationStagingService
from training.export.torch_standalone import export_torch_standalone
from training.export.universal_exporter import (
    export_tri_format_pytorch,
    export_tri_format_yolo,
    export_universal_model,
    get_model_hub_dir,
)
from training.export.validation import (
    IsolationReport,
    StrictIsolationValidator,
    VariationValidationReport,
)
from training.export.webgpu import export_webgpu_onnx
from training.export.yolo_exporter import YOLOModelExporter
from training.utils.fs import remove_tree
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.export")


@dataclass(frozen=True)
class ExportPlan:
    """Which formats :func:`export_all` writes and whether it stages hub variations.

    ``targets`` can include "onnx_fp32", "onnx_fp16", "torch_pt", "webgpu" and "forex";
    ``None`` selects ["forex"] for Forex models and ["onnx_fp32", "onnx_fp16", "torch_pt"]
    otherwise.
    """

    targets: list[str] | None = None
    stage_dist: bool = True


def _load_export_config(config: dict[str, Any] | None, project_root: Path) -> dict[str, Any]:
    """Return ``config``, or ``config.yaml`` from the project root when none is given."""
    cfg = config or {}
    if not cfg:
        cfg_path = project_root / "config.yaml"
        if cfg_path.exists():
            try:
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError) as e:
                logger.warning("Could not read config.yaml: %s", e)
    return cfg


def _load_export_registry(cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Read the unified models registry named by ``cfg`` (falls back to unified_models.yaml)."""
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
    return registry


def _load_export_model(
    model_key: str, cfg: dict[str, Any], ckpt_file: Path | None, model_info: dict[str, Any]
) -> torch.nn.Module:
    """Build the model on the CPU, load ``ckpt_file`` weights and wrap quality models."""
    model = get_model(model_key, cfg).to(torch.device("cpu"))

    if ckpt_file and ckpt_file.exists():
        logger.info("Loading weights from checkpoint: %s", ckpt_file)
        ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        load_checkpoint_weights(model, ckpt)

    model.eval()
    return wrap_quality_model(model, model_info)


def _export_model_targets(
    model: torch.nn.Module,
    model_key: str,
    export_targets: list[str],
    staging_dir: Path,
    names: tuple[str, tuple[int, int, int, int]],
) -> dict[str, Path]:
    """Write each requested non-Forex target into ``staging_dir``.

    ``names`` is ``(base_name, dummy_shape)``.
    """
    base_name, dummy_shape = names
    results: dict[str, Path] = {}
    if "onnx_fp32" in export_targets:
        fp32_path = staging_dir / f"{base_name}_FP32.onnx"
        export_onnx(
            model, fp32_path, dummy_shape=dummy_shape, graph=OnnxGraphSpec(external_data=True)
        )
        results["onnx_fp32"] = fp32_path
        results["onnx_fp32_data"] = staging_dir / f"{base_name}_FP32.onnx.data"

    if "onnx_fp16" in export_targets:
        fp16_path = staging_dir / f"{base_name}.onnx"
        export_onnx(model, fp16_path, dummy_shape=dummy_shape, half=True)
        results["onnx_fp16"] = fp16_path

    if "torch_pt" in export_targets:
        pt_path = staging_dir / f"{base_name}.pt"
        export_torch_standalone(model, pt_path, metadata={"model_key": model_key})
        results["torch_pt"] = pt_path
        results["pt"] = pt_path

    if "webgpu" in export_targets:
        webgpu_path = staging_dir / f"{base_name}_webgpu.onnx"
        export_webgpu_onnx(model, webgpu_path, dummy_input_shape=(1, 3, 512, 512))
        results["webgpu"] = webgpu_path
    return results


def _export_targets(model_key: str, model_info: dict[str, Any], plan: ExportPlan) -> list[str]:
    """``["forex"]`` for Forex models, else the plan's targets or the three defaults."""
    if model_info.get("dataset_type") == "forex" or "forex" in model_key.lower():
        return ["forex"]
    return plan.targets or ["onnx_fp32", "onnx_fp16", "torch_pt"]


def _export_forex_checkpoint(
    model_key: str, model_info: dict[str, Any], ckpt_file: Path | None, out_path: Path
) -> Path:
    """Export a Forex checkpoint to ONNX over the model's active timeframes."""
    if not ckpt_file:
        raise ExportError("CKPT_MISSING", f"No checkpoint found for forex model {model_key}")
    active_tfs = model_info.get("kwargs", {}).get("active_timeframes", [1, 5, 15, 60, 240, 1440])
    return export_forex_onnx(ckpt_file, out_path, active_timeframes=active_tfs)


def _stage_dist(
    model_key: str, results: dict[str, Path], destination_dir: Path, staging_dir: Path
) -> None:
    """Stage exported files into isolated variations, add them to ``results``, drop staging."""
    staged_res = VariationStagingService().stage_variations(model_key, results, destination_dir)
    results.update(staged_res.staged_files)
    if staging_dir.exists():
        remove_tree(staging_dir, purpose="export staging cleanup")


def export_all(
    model_key: str,
    checkpoint_path: Path | str | None = None,
    config: dict[str, Any] | None = None,
    output_dir: Path | str | None = None,
    plan: ExportPlan | None = None,
) -> dict[str, Path]:
    """Run the full production export for the targets of ``plan`` (see :class:`ExportPlan`)."""
    plan = plan or ExportPlan()
    project_root = get_project_root()
    cfg = _load_export_config(config, project_root)
    model_info = _load_export_registry(cfg, project_root).get(model_key, {})
    base_name, destination_dir = resolve_export_paths(model_key, model_info, cfg, project_root)
    if output_dir:
        destination_dir = Path(output_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = resolve_checkpoint_file(
        model_key=model_key,
        production_dir=destination_dir,
        config=cfg,
        project_root=project_root,
        user_checkpoint=checkpoint_path,
    )

    export_targets = _export_targets(model_key, model_info, plan)
    if "forex" in export_targets:
        results = {
            "forex": _export_forex_checkpoint(
                model_key, model_info, ckpt_file, destination_dir / f"{base_name}.onnx"
            )
        }
        if plan.stage_dist:
            VariationStagingService().stage_variations(model_key, results, destination_dir)
        return results

    model = _load_export_model(model_key, cfg, ckpt_file, model_info)
    staging_dir = (destination_dir / ".export_staging") if plan.stage_dist else destination_dir
    staging_dir.mkdir(parents=True, exist_ok=True)
    results = _export_model_targets(
        model,
        model_key,
        export_targets,
        staging_dir,
        (base_name, (1, 3, *extract_input_dimensions(model_info))),
    )

    if plan.stage_dist:
        _stage_dist(model_key, results, destination_dir, staging_dir)
    logger.info("Completed export_all for %s: %d targets exported.", model_key, len(results))
    return results


__all__ = [
    "BaseModelExporter",
    "ExportError",
    "ExportPlan",
    "ExporterFactory",
    "ForexModelExporter",
    "IsolationReport",
    "LLMModelExporter",
    "OnnxGraphSpec",
    "PyTorchModelExporter",
    "StagingResult",
    "StrictIsolationValidator",
    "VariationStagingService",
    "VariationValidationReport",
    "YOLOModelExporter",
    "export_all",
    "export_forex_onnx",
    "export_llm_tri_format",
    "export_onnx",
    "export_torch_standalone",
    "export_tri_format_pytorch",
    "export_tri_format_yolo",
    "export_universal_model",
    "export_webgpu_onnx",
    "extract_input_dimensions",
    "get_exporter_for_model",
    "get_model_hub_dir",
    "resolve_checkpoint_file",
    "resolve_export_paths",
    "wrap_quality_model",
]
