"""Model export subsystem providing unified ONNX, TorchScript, WebGPU, and MT5 pipelines.

Public names resolve lazily (PEP 562) so importing a light submodule such as
``training.export.validation`` does not load torch, ``models.factory`` or the exporters.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_EXPORTS: dict[str, str] = {
    "BaseModelExporter": "training.export.base",
    "ExportError": "training.export.common",
    "ExportPlan": "training.export.pipeline",
    "ExporterFactory": "training.export.exporter_factory",
    "ForexModelExporter": "training.export.forex_exporter",
    "IsolationReport": "training.export.validation",
    "LLMModelExporter": "training.export.llm_exporter",
    "OnnxGraphSpec": "training.export.common",
    "PyTorchModelExporter": "training.export.pytorch_exporter",
    "StagingResult": "training.export.staging",
    "StrictIsolationValidator": "training.export.validation",
    "VariationStagingService": "training.export.staging",
    "VariationValidationReport": "training.export.validation",
    "YOLOModelExporter": "training.export.yolo_exporter",
    "export_all": "training.export.pipeline",
    "export_forex_onnx": "training.export.mt5_signal",
    "export_llm_tri_format": "training.export.llm_exporter",
    "export_onnx": "training.export.onnx",
    "export_torch_standalone": "training.export.torch_standalone",
    "export_tri_format_pytorch": "training.export.universal_exporter",
    "export_tri_format_yolo": "training.export.universal_exporter",
    "export_universal_model": "training.export.universal_exporter",
    "export_webgpu_onnx": "training.export.webgpu",
    "extract_input_dimensions": "training.export.common",
    "get_exporter_for_model": "training.export.exporter_factory",
    "get_model_hub_dir": "training.export.universal_exporter",
    "resolve_checkpoint_file": "training.export.common",
    "resolve_export_paths": "training.export.common",
    "wrap_quality_model": "training.export.common",
}

if TYPE_CHECKING:
    from training.export.base import BaseModelExporter
    from training.export.common import (
        ExportError,
        OnnxGraphSpec,
        extract_input_dimensions,
        resolve_checkpoint_file,
        resolve_export_paths,
        wrap_quality_model,
    )
    from training.export.exporter_factory import ExporterFactory, get_exporter_for_model
    from training.export.forex_exporter import ForexModelExporter
    from training.export.llm_exporter import LLMModelExporter, export_llm_tri_format
    from training.export.mt5_signal import export_forex_onnx
    from training.export.onnx import export_onnx
    from training.export.pipeline import ExportPlan, export_all
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


def __getattr__(name: str) -> Any:
    """Import the defining module of ``name`` on first access and cache it."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


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
