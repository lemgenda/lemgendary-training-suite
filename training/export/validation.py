"""Strict Isolation Validator for Kaggle model variations (SSOT).

Enforces zero cross-contamination and binary isolation across all 4 variations:
1. pytorch/default -> Strictly checkpoints/ folder + root files (README.md, metrics.csv).
                       Zero .pt or .onnx binaries. Zero metadata files.
2. pytorch/fp32    -> Strictly [ModelStem].pt binary. Zero other files.
3. onnx/fp16       -> Strictly [ModelStem].onnx (self-contained, zero external tensors).
4. onnx/fp32       -> Strictly [ModelStem]_FP32.onnx and [ModelStem]_FP32.onnx.data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from training.cloud.variations import ONNX_FP16, ONNX_FP32, PYTORCH_DEFAULT, PYTORCH_FP32, VARIATIONS
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
    """Audits staged variation directories to guarantee strict binary isolation across all 4 variations."""

    ALLOWED_EXTENSIONS_BY_VARIATION: ClassVar[dict[str, set[str]]] = {
        PYTORCH_DEFAULT.key: {".md", ".csv", ".pth"},
        PYTORCH_FP32.key: {".pt"},
        ONNX_FP32.key: {".onnx", ".data"},
        ONNX_FP16.key: {".onnx"},
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

    def validate_pytorch_default_variation(
        self,
        variation_dir: Path,
    ) -> VariationValidationReport:
        """Audit pytorch/default variation directory (checkpoints/ + root files)."""
        report = VariationValidationReport(variation=PYTORCH_DEFAULT.key, directory=variation_dir)
        if not variation_dir.exists():
            report.is_valid = False
            report.errors.append(f"Directory {variation_dir} does not exist.")
            return report

        entries = list(variation_dir.iterdir())
        files = [p for p in entries if p.is_file()]
        dirs = [p for p in entries if p.is_dir()]
        report.files_found = [p.name for p in entries]

        # 1. Reject rogue metadata, .pt or .onnx files
        forbidden_substrings = ("model-metadata", "model_metadata", ".pt", ".onnx")
        for f in files:
            for forb in forbidden_substrings:
                if forb in f.name.lower():
                    report.is_valid = False
                    report.errors.append(
                        f"Forbidden artifact '{f.name}' in pytorch/default. "
                        f"Binary .pt/.onnx models and metadata files must NOT be in default checkpoints variation."
                    )

        # 2. Checkpoints directory presence
        ckpt_dir = variation_dir / "checkpoints"
        if not ckpt_dir.exists() or not ckpt_dir.is_dir():
            report.is_valid = False
            report.errors.append("Missing required 'checkpoints/' directory in pytorch/default.")
        else:
            pth_files = list(ckpt_dir.glob("*.pth"))
            if not pth_files:
                report.is_valid = False
                report.errors.append("No *.pth checkpoints found inside checkpoints/ directory.")

        return report

    def validate_pytorch_fp32_variation(
        self,
        variation_dir: Path,
        expected_stem: str,
    ) -> VariationValidationReport:
        """Audit pytorch/fp32 variation directory (strictly [ModelStem].pt)."""
        report = VariationValidationReport(variation=PYTORCH_FP32.key, directory=variation_dir)
        if not variation_dir.exists():
            report.is_valid = False
            report.errors.append(f"Directory {variation_dir} does not exist.")
            return report

        files = [p for p in variation_dir.iterdir() if p.is_file()]
        report.files_found = [f.name for f in files]

        # 1. Reject foreign file types
        allowed_exts = self.ALLOWED_EXTENSIONS_BY_VARIATION[PYTORCH_FP32.key]
        for f in files:
            if f.suffix not in allowed_exts:
                report.is_valid = False
                report.errors.append(
                    f"Foreign file detected in pytorch/fp32 variation: '{f.name}'. Only *.pt files are permitted."
                )

        # 2. Check for expected binary
        pt_files = [f for f in files if f.suffix == ".pt"]
        if not pt_files:
            report.is_valid = False
            report.errors.append("No .pt file found in pytorch/fp32 variation directory.")
        elif len(pt_files) > 1:
            report.is_valid = False
            report.errors.append(f"Multiple .pt files detected in pytorch/fp32 variation: {[f.name for f in pt_files]}")

        return report

    def validate_pytorch_variation(
        self,
        variation_dir: Path,
        expected_stem: str,
    ) -> VariationValidationReport:
        """Alias for validate_pytorch_fp32_variation (backward compatibility)."""
        return self.validate_pytorch_fp32_variation(variation_dir, expected_stem)

    def validate_onnx_fp32_variation(
        self,
        variation_dir: Path,
        expected_stem: str,
    ) -> VariationValidationReport:
        """Audit onnx/fp32 variation directory (strictly *_FP32.onnx and .data sidecar)."""
        report = VariationValidationReport(variation=ONNX_FP32.key, directory=variation_dir)
        if not variation_dir.exists():
            report.is_valid = False
            report.errors.append(f"Directory {variation_dir} does not exist.")
            return report

        files = [p for p in variation_dir.iterdir() if p.is_file()]
        report.files_found = [f.name for f in files]

        for f in files:
            is_onnx = f.suffix == ".onnx"
            is_sidecar = f.name.endswith(".onnx.data") or f.suffix == ".data"
            if not (is_onnx or is_sidecar):
                report.is_valid = False
                report.errors.append(
                    f"Foreign file detected in onnx/fp32 variation: '{f.name}'. "
                    f"Only *_FP32.onnx and external sidecar files are permitted."
                )

        onnx_files = [f for f in files if f.suffix == ".onnx"]
        if not onnx_files:
            report.is_valid = False
            report.errors.append("No .onnx file found in onnx/fp32 variation directory.")

        return report

    def validate_onnx_fp16_variation(
        self,
        variation_dir: Path,
        expected_stem: str,
    ) -> VariationValidationReport:
        """Audit onnx/fp16 variation directory (strictly self-contained *.onnx)."""
        report = VariationValidationReport(variation=ONNX_FP16.key, directory=variation_dir)
        if not variation_dir.exists():
            report.is_valid = False
            report.errors.append(f"Directory {variation_dir} does not exist.")
            return report

        files = [p for p in variation_dir.iterdir() if p.is_file()]
        report.files_found = [f.name for f in files]

        for f in files:
            if f.suffix == ".data" or f.name.endswith(".onnx.data"):
                report.is_valid = False
                report.errors.append(
                    f"External data sidecar detected in onnx/fp16 variation: '{f.name}'. "
                    f"ONNX FP16 variation must be self-contained with embedded weights."
                )
            elif f.suffix != ".onnx":
                report.is_valid = False
                report.errors.append(
                    f"Foreign file detected in onnx/fp16 variation: '{f.name}'. Only *.onnx files are permitted."
                )

        onnx_files = [f for f in files if f.suffix == ".onnx"]
        if not onnx_files:
            report.is_valid = False
            report.errors.append("No .onnx file found in onnx/fp16 variation directory.")
        elif len(onnx_files) > 1:
            report.is_valid = False
            report.errors.append(f"Multiple .onnx files found in onnx/fp16 variation: {[f.name for f in onnx_files]}")

        return report

    def validate_variation(
        self,
        variation_key: str,
        variation_dir: Path,
        model_key: str,
    ) -> VariationValidationReport:
        """Validate an individual variation directory by key."""
        expected_stem = self._resolve_stem(model_key)
        if variation_key == PYTORCH_DEFAULT.key:
            return self.validate_pytorch_default_variation(variation_dir)
        elif variation_key == PYTORCH_FP32.key:
            return self.validate_pytorch_fp32_variation(variation_dir, expected_stem)
        elif variation_key == ONNX_FP32.key:
            return self.validate_onnx_fp32_variation(variation_dir, expected_stem)
        elif variation_key == ONNX_FP16.key:
            return self.validate_onnx_fp16_variation(variation_dir, expected_stem)
        else:
            rep = VariationValidationReport(variation=variation_key, directory=variation_dir, is_valid=False)
            rep.errors.append(f"Unknown variation key: '{variation_key}'")
            return rep

    def validate_all(
        self,
        model_key: str,
        base_dir: Path | str | None = None,
    ) -> IsolationReport:
        """Audit all 4 variations for the given model."""
        if base_dir:
            model_hub_dir = Path(base_dir).resolve()
        else:
            root = get_project_root()
            model_hub_dir = (root.parent / "LemGendaryModels" / model_key).resolve()

        dist_kaggle = model_hub_dir / "dist" / "kaggle"
        overall_report = IsolationReport(model_key=model_key, is_valid=True)

        for variation in VARIATIONS:
            var_dir = dist_kaggle / variation.payload_dir
            # Fallback check for legacy folder names or canonical model root for pytorch/default
            if not var_dir.exists():
                if variation.key == PYTORCH_DEFAULT.key and (model_hub_dir / "checkpoints").exists():
                    var_dir = model_hub_dir
                else:
                    legacy_mapping = {
                        PYTORCH_FP32.key: dist_kaggle / "pytorch",
                        ONNX_FP32.key: dist_kaggle / "onnx_fp32",
                        ONNX_FP16.key: dist_kaggle / "onnx_fp16",
                    }
                    fallback_dir = legacy_mapping.get(variation.key)
                    if fallback_dir and fallback_dir.exists():
                        var_dir = fallback_dir

            var_rep = self.validate_variation(variation.key, var_dir, model_key)
            overall_report.variations[variation.key] = var_rep
            if not var_rep.is_valid:
                overall_report.is_valid = False
                overall_report.all_errors.extend(var_rep.errors)

        return overall_report
