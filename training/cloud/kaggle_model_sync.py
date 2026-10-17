"""Kaggle Model Hub Synchronization Manager.

Orchestrates validation, metadata synchronization, and publishing of isolated model variations
(PyTorch, ONNX FP32, ONNX FP16) to Kaggle Models Hub via kagglehub.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.cloud.credentials import resolve_kaggle_credentials
from training.cloud.manager import CloudSyncError
from training.cloud.metadata_generator import KaggleMetadataService
from training.export.validation import IsolationReport, StrictIsolationValidator
from training.registry import ModelSpecification, get_model_registry
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.cloud.kaggle_model_sync")

VARIATION_FRAMEWORK_MAP: dict[str, tuple[str, str]] = {
    "pytorch": ("pytorch", "default"),
    "onnx_fp32": ("onnx", "fp32"),
    "onnx_fp16": ("onnx", "fp16"),
}


@dataclass
class VariationPushResult:
    """Result of pushing a single model variation to Kaggle Models Hub."""

    variation: str
    handle: str
    success: bool
    files: list[str] = field(default_factory=list)
    error: str | None = None
    dry_run: bool = False


@dataclass
class ModelPushResult:
    """Result of multi-variation push operation."""

    model_key: str
    success: bool
    variations: dict[str, VariationPushResult] = field(default_factory=dict)
    isolation_report: IsolationReport | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize push result to dictionary."""
        return {
            "model_key": self.model_key,
            "success": self.success,
            "variations": {
                k: {
                    "variation": v.variation,
                    "handle": v.handle,
                    "success": v.success,
                    "files": v.files,
                    "error": v.error,
                    "dry_run": v.dry_run,
                }
                for k, v in self.variations.items()
            },
            "is_valid_isolation": self.isolation_report.is_valid if self.isolation_report else False,
            "errors": self.errors,
        }


class KaggleModelSyncManager:
    """Orchestrates Kaggle Models Hub uploads with mandatory binary isolation validation."""

    def __init__(
        self,
        username: str | None = None,
        key: str | None = None,
        hub_root: Path | str | None = None,
    ) -> None:
        user, resolved_key = resolve_kaggle_credentials(override_user=username, override_key=key)
        self.username = user or "lemtreursi"
        self.key = resolved_key

        if hub_root:
            self.hub_root = Path(hub_root).resolve()
        else:
            self.hub_root = (get_project_root().parent / "LemGendaryModels").resolve()

        self.registry = get_model_registry()
        self.metadata_service = KaggleMetadataService(owner_slug=self.username)
        self.validator = StrictIsolationValidator()

    def get_model_slug(self, model_key: str) -> str:
        """Resolve canonical Kaggle model slug."""
        spec = self.registry.get_model(model_key)
        if spec and spec.kaggle_metadata and spec.kaggle_metadata.slug:
            return spec.kaggle_metadata.slug
        return model_key.replace("_", "-").lower()

    def get_handle(self, model_key: str, variation: str = "pytorch") -> str:
        """Construct standard Kaggle model handle: <owner>/<model_slug>/<framework>/<variation_slug>."""
        slug = self.get_model_slug(model_key)
        framework, var_slug = VARIATION_FRAMEWORK_MAP.get(variation, (variation, "default"))
        return f"{self.username}/{slug}/{framework}/{var_slug}"

    def get_model_dir(self, model_key: str) -> Path:
        """Resolve root folder in LemGendaryModels for model."""
        return self.hub_root / model_key

    def ensure_model_metadata(self, model_key: str) -> Path:
        """Ensure model-metadata.json is materialized in the model directory."""
        model_dir = self.get_model_dir(model_key)
        model_dir.mkdir(parents=True, exist_ok=True)
        meta_file = model_dir / "model-metadata.json"
        if not meta_file.exists():
            meta_file = self.metadata_service.write_metadata(model_key, output_dir=model_dir)
        return meta_file

    def validate_isolation(self, model_key: str) -> IsolationReport:
        """Audit binary isolation across all variation subdirectories."""
        model_dir = self.get_model_dir(model_key)
        return self.validator.validate_all(model_key, base_dir=model_dir)

    def push_variation(
        self,
        model_key: str,
        variation: str,
        version_notes: str | None = None,
        dry_run: bool = False,
    ) -> VariationPushResult:
        """Validate and upload a single isolated model variation to Kaggle Models Hub."""
        if variation not in VARIATION_FRAMEWORK_MAP:
            return VariationPushResult(
                variation=variation,
                handle="",
                success=False,
                error=f"Unsupported variation '{variation}'. Allowed: {list(VARIATION_FRAMEWORK_MAP.keys())}",
                dry_run=dry_run,
            )

        model_dir = self.get_model_dir(model_key)
        variation_dir = model_dir / "dist" / "kaggle" / variation
        handle = self.get_handle(model_key, variation)

        if not variation_dir.exists():
            return VariationPushResult(
                variation=variation,
                handle=handle,
                success=False,
                error=f"Variation directory does not exist: {variation_dir}",
                dry_run=dry_run,
            )

        files = [p for p in variation_dir.iterdir() if p.is_file()]
        if not files:
            return VariationPushResult(
                variation=variation,
                handle=handle,
                success=False,
                error=f"Variation directory is empty: {variation_dir}",
                dry_run=dry_run,
            )

        file_names = [f.name for f in files]

        # Audit isolation for this variation
        stem = self.validator._resolve_stem(model_key)
        if variation == "pytorch":
            v_report = self.validator.validate_pytorch_variation(variation_dir, stem)
        elif variation == "onnx_fp32":
            v_report = self.validator.validate_onnx_fp32_variation(variation_dir, stem)
        elif variation == "onnx_fp16":
            v_report = self.validator.validate_onnx_fp16_variation(variation_dir, stem)
        else:
            v_report = None

        if v_report and not v_report.is_valid:
            return VariationPushResult(
                variation=variation,
                handle=handle,
                success=False,
                files=file_names,
                error=f"Strict isolation audit failed: {'; '.join(v_report.errors)}",
                dry_run=dry_run,
            )

        # Materialize model-metadata.json
        self.ensure_model_metadata(model_key)

        if dry_run:
            logger.info("[DRY-RUN] Verified Kaggle model variation payload %s (%s)", handle, variation_dir)
            return VariationPushResult(
                variation=variation,
                handle=handle,
                success=True,
                files=file_names,
                dry_run=True,
            )

        # Upload via kagglehub
        try:
            kagglehub = importlib.import_module("kagglehub")
        except ImportError as exc:
            return VariationPushResult(
                variation=variation,
                handle=handle,
                success=False,
                files=file_names,
                error="kagglehub SDK is not installed in runtime environment.",
                dry_run=False,
            )

        try:
            notes = version_notes or f"Production SOTA Release: {model_key} [{variation.upper()}]"
            logger.info("Uploading %s variation to Kaggle Models Hub: %s ...", variation, handle)
            kagglehub.model_upload(
                handle=handle,
                local_model_dir=str(variation_dir),
                version_notes=notes,
            )
            logger.info("[OK] Successfully uploaded %s to Kaggle Models Hub", handle)
            return VariationPushResult(
                variation=variation,
                handle=handle,
                success=True,
                files=file_names,
                dry_run=False,
            )
        except Exception as exc:
            logger.error("Failed uploading %s to Kaggle Models Hub: %s", handle, exc)
            return VariationPushResult(
                variation=variation,
                handle=handle,
                success=False,
                files=file_names,
                error=str(exc),
                dry_run=False,
            )

    def push_model(
        self,
        model_key: str,
        variation: str = "all",
        version_notes: str | None = None,
        dry_run: bool = False,
    ) -> ModelPushResult:
        """Push specified or all variations for a model to Kaggle Models Hub."""
        isolation_rep = self.validate_isolation(model_key)

        variations_to_push = (
            list(VARIATION_FRAMEWORK_MAP.keys())
            if variation == "all"
            else [variation]
        )

        push_result = ModelPushResult(
            model_key=model_key,
            success=True,
            isolation_report=isolation_rep,
        )

        for var in variations_to_push:
            var_res = self.push_variation(
                model_key=model_key,
                variation=var,
                version_notes=version_notes,
                dry_run=dry_run,
            )
            push_result.variations[var] = var_res
            if not var_res.success:
                push_result.success = False
                if var_res.error:
                    push_result.errors.append(f"[{var}] {var_res.error}")

        return push_result
