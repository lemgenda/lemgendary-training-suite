"""Universal model exporter for LemGendary AI Training Suite.

Enforces uniform model export across all architectures and training methods:
1. [ModelName].pt: FP32 PyTorch model weights (isolated in dist/kaggle/pytorch/)
2. [ModelName]_FP32.onnx + [ModelName]_FP32.onnx.data: FP32 with external binary weights sidecar
   (isolated in dist/kaggle/onnx_fp32/)
3. [ModelName].onnx: FP16 precision with integrated self-contained weights (isolated in dist/kaggle/onnx_fp16/)

All exports persist exclusively to LemGendaryModels/[ModelName]/ and NEVER to lemgendary-training-suite/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from training.export.exporter_factory import ExporterFactory
from training.export.pytorch_exporter import PyTorchModelExporter
from training.export.staging import StagingResult, VariationStagingService
from training.export.yolo_exporter import YOLOModelExporter
from training.registry import get_model_registry
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.export.universal")


def get_model_hub_dir(model_key: str, project_root: Path | None = None) -> Path:
    """Return the authoritative target directory in LemGendaryModels."""
    root = (project_root or get_project_root()).resolve()
    hub_dir = (root.parent / "LemGendaryModels" / model_key).resolve()
    hub_dir.mkdir(parents=True, exist_ok=True)
    return hub_dir


def export_tri_format_pytorch(
    model: nn.Module,
    model_key: str,
    dummy_input: torch.Tensor,
    output_dir: Path | str | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    opset: int = 17,
    stage_dist: bool = True,
) -> dict[str, Path]:
    """Export standard PyTorch model into all 3 required formats with automatic variation staging.

    Returns dict with keys: 'onnx_fp16', 'onnx_fp32', 'onnx_fp32_data', 'pt'
    """
    target_dir = Path(output_dir).resolve() if output_dir else get_model_hub_dir(model_key)
    staging_dir = target_dir / ".export_staging" if stage_dist else target_dir
    staging_dir.mkdir(parents=True, exist_ok=True)

    exporter = PyTorchModelExporter()
    exported = exporter.export(
        model=model,
        model_key=model_key,
        output_dir=staging_dir,
        dummy_input=dummy_input,
        dynamic_axes=dynamic_axes,
        input_names=input_names,
        output_names=output_names,
        opset=opset,
    )

    if stage_dist:
        staging_svc = VariationStagingService()
        st_res = staging_svc.stage_variations(model_key=model_key, source_files=exported, base_dir=target_dir)
        if staging_dir.exists():
            import shutil
            shutil.rmtree(staging_dir, ignore_errors=True)
        if st_res.staged_files:
            exported = st_res.staged_files

    return exported


def export_tri_format_yolo(
    trainer: Any,
    model_key: str = "yolov8n",
    output_dir: Path | str | None = None,
    stage_dist: bool = True,
) -> dict[str, Path]:
    """Export Ultralytics YOLO model into all 3 required formats with automatic variation staging."""
    target_dir = Path(output_dir).resolve() if output_dir else get_model_hub_dir(model_key)
    staging_dir = target_dir / ".export_staging" if stage_dist else target_dir
    staging_dir.mkdir(parents=True, exist_ok=True)

    exporter = YOLOModelExporter()
    exported = exporter.export(
        model=trainer,
        model_key=model_key,
        output_dir=staging_dir,
    )

    if stage_dist:
        staging_svc = VariationStagingService()
        st_res = staging_svc.stage_variations(model_key=model_key, source_files=exported, base_dir=target_dir)
        if staging_dir.exists():
            import shutil
            shutil.rmtree(staging_dir, ignore_errors=True)
        if st_res.staged_files:
            exported = st_res.staged_files

    return exported


def export_universal_model(
    model_key: str,
    model: Any | None = None,
    output_dir: Path | str | None = None,
    stage_dist: bool = True,
    **kwargs: Any,
) -> tuple[dict[str, Path], StagingResult | None]:
    """Universal export facade resolving exporter via ExporterFactory and staging variations."""
    reg = get_model_registry()
    spec = reg.get_model(model_key)
    target_dir = Path(output_dir).resolve() if output_dir else get_model_hub_dir(model_key)
    staging_dir = target_dir / ".export_staging" if stage_dist else target_dir
    staging_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        from models.factory import get_model

        model = get_model(model_key)

    exporter = ExporterFactory.get_exporter(model_key, spec)
    exported = exporter.export(
        model=model,
        model_key=model_key,
        output_dir=staging_dir,
        spec=spec,
        **kwargs,
    )

    staging_result = None
    if stage_dist:
        staging_svc = VariationStagingService(spec=spec)
        staging_result = staging_svc.stage_variations(
            model_key=model_key,
            source_files=exported,
            base_dir=target_dir,
            validate=True,
        )
        if staging_dir.exists():
            import shutil
            shutil.rmtree(staging_dir, ignore_errors=True)
        if staging_result.staged_files:
            exported = staging_result.staged_files

    return exported, staging_result


def main() -> None:
    """CLI entrypoint for universal exporter."""
    parser = argparse.ArgumentParser(description="Universal Model Exporter & Variation Stager")
    parser.add_argument("--model", type=str, required=True, help="Model key (e.g. yolov8n, ai_helper_lemgend)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom export destination")
    parser.add_argument("--stage-dist", action="store_true", default=True, help="Stage into dist/kaggle/ subfolders")
    parser.add_argument("--stage-only", action="store_true", help="Only stage existing binaries without re-exporting")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    if args.stage_only:
        staging_svc = VariationStagingService()
        res = staging_svc.stage_variations(args.model, base_dir=args.output_dir, validate=True)
        if res.report:
            print(res.report.summary())
        if not res.success:
            raise SystemExit(1)
        print(f"Successfully staged variations for {args.model}")
        return

    exported, st_res = export_universal_model(
        model_key=args.model,
        output_dir=args.output_dir,
        stage_dist=args.stage_dist,
    )
    print(f"Exported {len(exported)} artifacts for {args.model}:")
    for k, v in exported.items():
        print(f"  {k}: {v}")

    if st_res and st_res.report:
        print("\n" + st_res.report.summary())


if __name__ == "__main__":
    main()
