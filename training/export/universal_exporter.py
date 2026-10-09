"""Universal model exporter for LemGendary AI Training Suite.

Enforces uniform model export across all architectures and training methods:
1. [ModelName].onnx: FP16 precision with integrated self-contained weights
2. [ModelName]_FP32.onnx + [ModelName]_FP32.onnx.data: FP32 with external binary weights sidecar
3. [ModelName].pt: FP32 PyTorch model weights

All exports persist exclusively to LemGendaryModels/[ModelName]/ and NEVER to lemgendary-training-suite/.
"""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn

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
) -> dict[str, Path]:
    """Export standard PyTorch model into all 3 required formats.

    Returns dict with keys: 'onnx_fp16', 'onnx_fp32', 'onnx_fp32_data', 'pt'
    """
    target_dir = Path(output_dir).resolve() if output_dir else get_model_hub_dir(model_key)
    target_dir.mkdir(parents=True, exist_ok=True)

    in_names = input_names or ["input"]
    out_names = output_names or ["output"]
    dyn_axes = dynamic_axes or {
        in_names[0]: {0: "batch", 2: "height", 3: "width"},
        out_names[0]: {0: "batch", 2: "height", 3: "width"},
    }

    model_to_export = model.module if hasattr(model, "module") else model
    model_to_export.eval()
    device = next(model_to_export.parameters()).device

    exported_files: dict[str, Path] = {}

    # -------------------------------------------------------------------------
    # 1. FP32 PyTorch Model (.pt)
    # -------------------------------------------------------------------------
    pt_path = target_dir / f"{model_key}.pt"
    logger.info("Saving FP32 PyTorch model to %s...", pt_path)
    try:
        torch.save(model_to_export.state_dict(), str(pt_path))
        exported_files["pt"] = pt_path
        logger.info("Successfully saved FP32 PyTorch model to %s", pt_path)
    except Exception as exc:
        logger.error("Failed saving FP32 PyTorch model to %s: %s", pt_path, exc)

    # -------------------------------------------------------------------------
    # 2. FP32 ONNX Model with Separate Weights Sidecar (.onnx + .onnx.data)
    # -------------------------------------------------------------------------
    fp32_onnx_path = target_dir / f"{model_key}_FP32.onnx"
    fp32_data_filename = f"{model_key}_FP32.onnx.data"
    fp32_data_path = target_dir / fp32_data_filename

    logger.info("Exporting FP32 ONNX model with external sidecar to %s...", fp32_onnx_path)
    buf_fp32 = io.StringIO()
    try:
        dummy_fp32 = dummy_input.to(dtype=torch.float32, device=device)
        model_to_export.to(dtype=torch.float32)

        with redirect_stdout(buf_fp32), redirect_stderr(buf_fp32):
            torch.onnx.export(
                model_to_export,
                (dummy_fp32,),
                str(fp32_onnx_path),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=in_names,
                output_names=out_names,
                dynamic_axes=dyn_axes,
            )

        # Convert to external data sidecar
        onnx_model = onnx.load(str(fp32_onnx_path))
        onnx.external_data_helper.convert_model_to_external_data(
            onnx_model,
            all_tensors_to_one_file=True,
            location=fp32_data_filename,
            size_threshold=1024,
            convert_attribute=False,
        )
        onnx.save(onnx_model, str(fp32_onnx_path))

        exported_files["onnx_fp32"] = fp32_onnx_path
        exported_files["onnx_fp32_data"] = fp32_data_path
        logger.info(
            "Successfully exported FP32 ONNX with sidecar: %s (graph: %.2f KB, weights: %.2f MB)",
            fp32_onnx_path.name,
            fp32_onnx_path.stat().st_size / 1024,
            fp32_data_path.stat().st_size / (1024 * 1024) if fp32_data_path.exists() else 0.0,
        )
    except Exception as exc:
        logger.error("Failed exporting FP32 ONNX sidecar for %s: %s", model_key, exc)

    # -------------------------------------------------------------------------
    # 3. FP16 ONNX Model with Integrated Weights (.onnx)
    # -------------------------------------------------------------------------
    fp16_onnx_path = target_dir / f"{model_key}.onnx"
    logger.info("Exporting FP16 integrated ONNX model to %s...", fp16_onnx_path)
    buf_fp16 = io.StringIO()
    try:
        dummy_fp16 = dummy_input.to(dtype=torch.float16, device=device)
        model_to_export.to(dtype=torch.float16)

        with redirect_stdout(buf_fp16), redirect_stderr(buf_fp16):
            torch.onnx.export(
                model_to_export,
                (dummy_fp16,),
                str(fp16_onnx_path),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=in_names,
                output_names=out_names,
                dynamic_axes=dyn_axes,
            )

        exported_files["onnx_fp16"] = fp16_onnx_path
        logger.info(
            "Successfully exported FP16 integrated ONNX: %s (%.2f MB)",
            fp16_onnx_path.name,
            fp16_onnx_path.stat().st_size / (1024 * 1024),
        )
    except Exception as exc:
        logger.warning(
            "Direct FP16 torch.onnx.export failed for %s (%s). Falling back to onnxconverter float16 conversion...",
            model_key,
            exc,
        )
        try:
            # Fallback: export FP32 without sidecar temporarily, then convert to FP16
            tmp_fp32 = target_dir / f"{model_key}_temp_fp32.onnx"
            model_to_export.to(dtype=torch.float32)
            dummy_fp32 = dummy_input.to(dtype=torch.float32, device=device)
            torch.onnx.export(
                model_to_export,
                (dummy_fp32,),
                str(tmp_fp32),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=in_names,
                output_names=out_names,
                dynamic_axes=dyn_axes,
            )
            from onnxconverter_common import float16
            m_fp32 = onnx.load(str(tmp_fp32))
            m_fp16 = float16.convert_float_to_float16(m_fp32)
            onnx.save(m_fp16, str(fp16_onnx_path))
            if tmp_fp32.exists():
                tmp_fp32.unlink()
            exported_files["onnx_fp16"] = fp16_onnx_path
            logger.info("Successfully converted and saved FP16 ONNX model via onnxconverter fallback to %s", fp16_onnx_path)
        except Exception as fallback_exc:
            logger.error("Failed FP16 fallback export for %s: %s", model_key, fallback_exc)
    finally:
        # Revert model back to float32
        model_to_export.to(dtype=torch.float32)

    return exported_files


def export_tri_format_yolo(
    trainer: Any,
    model_key: str = "yolov8n",
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Export Ultralytics YOLO model into all 3 required formats."""
    target_dir = Path(output_dir).resolve() if output_dir else get_model_hub_dir(model_key)
    target_dir.mkdir(parents=True, exist_ok=True)

    exported_files: dict[str, Path] = {}

    # 1. FP32 PyTorch Model (.pt)
    pt_path = target_dir / f"{model_key}.pt"
    try:
        best_pt = getattr(trainer, "best", None)
        if best_pt and Path(best_pt).exists():
            import shutil
            shutil.copy2(str(best_pt), str(pt_path))
            exported_files["pt"] = pt_path
            logger.info("Mirrored YOLO FP32 PyTorch model to %s", pt_path)
    except Exception as exc:
        logger.error("Failed saving YOLO .pt to %s: %s", pt_path, exc)

    yolo_model = None
    if hasattr(trainer, "export"):
        yolo_model = trainer
    elif hasattr(trainer, "best") and Path(getattr(trainer, "best", "")).exists():
        from ultralytics import YOLO
        yolo_model = YOLO(str(trainer.best))
    elif pt_path.exists():
        from ultralytics import YOLO
        yolo_model = YOLO(str(pt_path))

    # 2. FP32 ONNX with Separate Weights Sidecar
    fp32_onnx_path = target_dir / f"{model_key}_FP32.onnx"
    fp32_data_path = target_dir / f"{model_key}_FP32.onnx.data"
    try:
        if yolo_model is not None and hasattr(yolo_model, "export"):
            raw_onnx = yolo_model.export(format="onnx", half=False, imgsz=640)
            if raw_onnx and Path(raw_onnx).exists():
                onnx_model = onnx.load(str(raw_onnx))
                onnx.external_data_helper.convert_model_to_external_data(
                    onnx_model,
                    all_tensors_to_one_file=True,
                    location=f"{model_key}_FP32.onnx.data",
                    size_threshold=1024,
                    convert_attribute=False,
                )
                onnx.save(onnx_model, str(fp32_onnx_path))
                exported_files["onnx_fp32"] = fp32_onnx_path
                exported_files["onnx_fp32_data"] = fp32_data_path
                logger.info("Successfully exported YOLO FP32 ONNX with sidecar to %s", fp32_onnx_path)
    except Exception as exc:
        logger.error("Failed exporting YOLO FP32 ONNX sidecar: %s", exc)

    # 3. FP16 ONNX with Integrated Weights
    fp16_onnx_path = target_dir / f"{model_key}.onnx"
    try:
        if yolo_model is not None and hasattr(yolo_model, "export"):
            raw_half_onnx = yolo_model.export(format="onnx", half=True, imgsz=640)
            if raw_half_onnx and Path(raw_half_onnx).exists():
                if Path(raw_half_onnx).resolve() != fp16_onnx_path.resolve():
                    import shutil
                    shutil.copy2(str(raw_half_onnx), str(fp16_onnx_path))
                exported_files["onnx_fp16"] = fp16_onnx_path
                logger.info("Successfully exported YOLO FP16 integrated ONNX to %s", fp16_onnx_path)
    except Exception as exc:
        logger.error("Failed exporting YOLO FP16 ONNX: %s", exc)

    # Clean up any misplaced/orphaned {model_key}.onnx.data
    orphaned_data = target_dir / f"{model_key}.onnx.data"
    if orphaned_data.exists():
        try:
            orphaned_data.unlink()
            logger.info("Cleaned up orphaned sidecar data %s", orphaned_data)
        except OSError:
            pass

    return exported_files
