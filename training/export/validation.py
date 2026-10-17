"""Strict Isolation Validator for Kaggle model variations.

Enforces zero cross-contamination and binary isolation across all 3 variations:
1. dist/kaggle/pytorch/   -> Strictly [ModelStem].pt
2. dist/kaggle/onnx_fp32/ -> Strictly [ModelStem]_FP32.onnx and [ModelStem]_FP32.onnx.data
3. dist/kaggle/onnx_fp16/ -> Strictly [ModelStem].onnx (self-contained, no external tensors)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from training.registry import ModelSpecification, get_model_registry
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.export.validation")


@dataclass
class VariationValidationReport:
    """Validation report for a single variation directory."""

    variation: str
    directory: Path
    is_valid: bool = True
    files_found: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class IsolationReport:
    """Comprehensive isolation report for all model variations."""

    model_key: str
    is_valid: bool = True
    variations: dict[str, VariationValidationReport] = field(default_factory=dict)
    all_errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Format a human-readable summary of validation status."""
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [f"Isolation Validation [{status}] for model '{self.model_key}':"]
        for var_name, report in self.variations.items():
            sub_status = "OK" if report.is_valid else "ERROR"
            lines.append(f"  - Variation '{var_name}': {sub_status} ({len(report.files_found)} files)")
            for err in report.errors:
                lines.append(f"      [ERROR] {err}")
            for warn in report.warnings:
                lines.append(f"      [WARN] {warn}")
        return "\n".join(lines)


class StrictIsolationValidator:
    """Audits staged variation directories to guarantee strict binary isolation."""

    ALLOWED_EXTENSIONS_BY_VARIATION: ClassVar[dict[str, set[str]]] = {
        "pytorch": {".pt"},
        "onnx_fp32": {".onnx", ".data"},
        "onnx_fp16": {".onnx"},
    }

    def __init__(self, spec: ModelSpecification | None = None) -> None:
        self.spec = spec

    def _resolve_stem(self, model_key: str) -> str:
        """Resolve canonical binary stem."""
        if self.spec and self.spec.filename:
            return self.spec.filename
        reg = get_model_registry()
        spec = reg.get_model(model_key)
        if spec and spec.filename:
            return spec.filename
        return model_key

    def validate_pytorch_variation(
        self,
        variation_dir: Path,
        expected_stem: str,
    ) -> VariationValidationReport:
        """Audit dist/kaggle/pytorch/ directory."""
        report = VariationValidationReport(variation="pytorch", directory=variation_dir)
        if not variation_dir.exists():
            report.is_valid = False
            report.errors.append(f"Directory {variation_dir} does not exist.")
            return report

        files = [p for p in variation_dir.iterdir() if p.is_file()]
        report.files_found = [f.name for f in files]

        # 1. Reject foreign file types
        allowed_exts = self.ALLOWED_EXTENSIONS_BY_VARIATION["pytorch"]
        for f in files:
            if f.suffix not in allowed_exts:
                report.is_valid = False
                report.errors.append(
                    f"Foreign file detected in pytorch variation: '{f.name}'. Only *.pt files are permitted."
                )

        # 2. Check for expected binary
        pt_files = [f for f in files if f.suffix == ".pt"]
        if not pt_files:
            report.is_valid = False
            report.errors.append("No .pt file found in pytorch variation directory.")
        elif len(pt_files) > 1:
            report.is_valid = False
            report.errors.append(f"Multiple .pt files detected in pytorch variation: {[f.name for f in pt_files]}")

        return report

    def validate_onnx_fp32_variation(
        self,
        variation_dir: Path,
        expected_stem: str,
    ) -> VariationValidationReport:
        """Audit dist/kaggle/onnx_fp32/ directory."""
        report = VariationValidationReport(variation="onnx_fp32", directory=variation_dir)
        if not variation_dir.exists():
            report.is_valid = False
            report.errors.append(f"Directory {variation_dir} does not exist.")
            return report

        files = [p for p in variation_dir.iterdir() if p.is_file()]
        report.files_found = [f.name for f in files]

        # 1. Reject foreign file types
        for f in files:
            # Allow .onnx and .data
            is_onnx = f.suffix == ".onnx"
            is_sidecar = f.name.endswith(".onnx.data") or f.suffix == ".data"
            if not (is_onnx or is_sidecar):
                report.is_valid = False
                report.errors.append(
                    f"Foreign file detected in onnx_fp32 variation: '{f.name}'. "
                    f"Only *_FP32.onnx and external sidecar files are permitted."
                )

        # 2. Check FP32 onnx model presence
        onnx_files = [f for f in files if f.suffix == ".onnx"]
        if not onnx_files:
            report.is_valid = False
            report.errors.append("No .onnx file found in onnx_fp32 variation directory.")
        else:
            for of in onnx_files:
                if not of.name.endswith("_FP32.onnx"):
                    report.warnings.append(
                        f"ONNX model '{of.name}' does not follow canonical *_FP32.onnx naming convention."
                    )

        return report

    def validate_onnx_fp16_variation(
        self,
        variation_dir: Path,
        expected_stem: str,
    ) -> VariationValidationReport:
        """Audit dist/kaggle/onnx_fp16/ directory."""
        report = VariationValidationReport(variation="onnx_fp16", directory=variation_dir)
        if not variation_dir.exists():
            report.is_valid = False
            report.errors.append(f"Directory {variation_dir} does not exist.")
            return report

        files = [p for p in variation_dir.iterdir() if p.is_file()]
        report.files_found = [f.name for f in files]

        # 1. Strictly forbid .data sidecar files (FP16 must be self-contained)
        for f in files:
            if f.suffix == ".data" or f.name.endswith(".onnx.data"):
                report.is_valid = False
                report.errors.append(
                    f"External data sidecar detected in onnx_fp16 variation: '{f.name}'. "
                    f"ONNX FP16 variation must be self-contained with embedded weights."
                )
            elif f.suffix != ".onnx":
                report.is_valid = False
                report.errors.append(
                    f"Foreign file detected in onnx_fp16 variation: '{f.name}'. Only *.onnx files are permitted."
                )

        # 2. Check FP16 onnx model presence
        onnx_files = [f for f in files if f.suffix == ".onnx"]
        if not onnx_files:
            report.is_valid = False
            report.errors.append("No .onnx file found in onnx_fp16 variation directory.")
        elif len(onnx_files) > 1:
            report.is_valid = False
            report.errors.append(f"Multiple .onnx files found in onnx_fp16 variation: {[f.name for f in onnx_files]}")

        return report

    def validate_all(
        self,
        model_key: str,
        base_dir: Path | str | None = None,
    ) -> IsolationReport:
        """Audit all variations under dist/kaggle/ for the given model."""
        if base_dir:
            model_hub_dir = Path(base_dir).resolve()
        else:
            root = get_project_root()
            model_hub_dir = (root.parent / "LemGendaryModels" / model_key).resolve()

        dist_kaggle = model_hub_dir / "dist" / "kaggle"
        expected_stem = self._resolve_stem(model_key)

        overall_report = IsolationReport(model_key=model_key, is_valid=True)

        # Validate PyTorch
        pt_dir = dist_kaggle / "pytorch"
        pt_rep = self.validate_pytorch_variation(pt_dir, expected_stem)
        overall_report.variations["pytorch"] = pt_rep
        if not pt_rep.is_valid:
            overall_report.is_valid = False
            overall_report.all_errors.extend(pt_rep.errors)

        # Validate ONNX FP32
        fp32_dir = dist_kaggle / "onnx_fp32"
        fp32_rep = self.validate_onnx_fp32_variation(fp32_dir, expected_stem)
        overall_report.variations["onnx_fp32"] = fp32_rep
        if not fp32_rep.is_valid:
            overall_report.is_valid = False
            overall_report.all_errors.extend(fp32_rep.errors)

        # Validate ONNX FP16
        fp16_dir = dist_kaggle / "onnx_fp16"
        fp16_rep = self.validate_onnx_fp16_variation(fp16_dir, expected_stem)
        overall_report.variations["onnx_fp16"] = fp16_rep
        if not fp16_rep.is_valid:
            overall_report.is_valid = False
            overall_report.all_errors.extend(fp16_rep.errors)

        return overall_report
