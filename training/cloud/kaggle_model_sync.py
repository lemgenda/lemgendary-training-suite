"""Kaggle Model Hub Synchronization Manager (SSOT).

Orchestrates validation, metadata synchronization via the official Kaggle REST API,
and publishing of strictly isolated model variations (PyTorch default, PyTorch FP32, ONNX FP16, ONNX FP32)
to Kaggle Models Hub via kagglehub.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.cloud.credentials import resolve_kaggle_credentials
from training.cloud.kaggle_api_sync import KaggleApiSyncService
from training.cloud.metadata_generator import KaggleMetadataService
from training.cloud.model_metadata import METADATA_FILENAME, ModelMetadata
from training.cloud.variations import (
    ONNX_FP16,
    ONNX_FP32,
    PYTORCH_DEFAULT,
    PYTORCH_FP32,
    VARIATION_BY_KEY,
    VARIATIONS,
    KaggleVariation,
)
from training.export.validation import IsolationReport, StrictIsolationValidator
from training.registry import ModelSpecification, get_model_registry
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.cloud.kaggle_model_sync")


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
    metadata_sync_result: dict[str, Any] | None = None
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
            "metadata_sync": self.metadata_sync_result,
            "errors": self.errors,
        }


class KaggleModelSyncManager:
    """Orchestrates Kaggle Models Hub uploads across all 4 isolated variations."""

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
        self.api_sync_service = KaggleApiSyncService(username=self.username, key=self.key)
        self.validator = StrictIsolationValidator()
        self._staging_service = None

    @property
    def staging_service(self):
        if self._staging_service is None:
            from training.export.staging import VariationStagingService
            self._staging_service = VariationStagingService()
        return self._staging_service

    def get_model_slug(self, model_key: str) -> str:
        """Resolve canonical Kaggle model slug."""
        spec = self.registry.get_model(model_key)
        if spec and spec.kaggle_metadata and spec.kaggle_metadata.slug:
            return spec.kaggle_metadata.slug
        return model_key.replace("_", "-").lower()

    def get_handle(self, model_key: str, variation_key: str) -> str:
        """Construct standard Kaggle model handle: <owner>/<model_slug>/<framework>/<variation_slug>."""
        slug = self.get_model_slug(model_key)
        variation = VARIATION_BY_KEY.get(variation_key)
        if variation:
            return variation.handle(self.username, slug)
        return f"{self.username}/{slug}/{variation_key}"

    def get_model_dir(self, model_key: str) -> Path:
        """Resolve root folder in LemGendaryModels for model."""
        return self.hub_root / model_key

    def get_variation_dir(self, model_key: str, variation_key: str) -> Path:
        """Resolve local variation upload directory under dist/kaggle/."""
        model_dir = self.get_model_dir(model_key)
        dist_dir = model_dir / "dist" / "kaggle"
        variation = VARIATION_BY_KEY.get(variation_key)
        if not variation:
            return dist_dir / variation_key

        if variation is PYTORCH_DEFAULT:
            payload_dir = dist_dir / PYTORCH_DEFAULT.payload_dir
            if payload_dir.exists():
                return payload_dir
            return model_dir
        elif variation is PYTORCH_FP32:
            return dist_dir / "pytorch"
        elif variation is ONNX_FP32:
            return dist_dir / "onnx_fp32"
        elif variation is ONNX_FP16:
            return dist_dir / "onnx_fp16"
        return dist_dir / variation_key

    def ensure_model_metadata(self, model_key: str) -> Path:
        """Ensure both model_metadata.yaml and model-metadata.json are materialized."""
        model_dir = self.get_model_dir(model_key)
        model_dir.mkdir(parents=True, exist_ok=True)
        yaml_file = model_dir / METADATA_FILENAME
        if not yaml_file.exists():
            return self.metadata_service.write_metadata(model_key, output_dir=model_dir)
        return yaml_file

    def validate_isolation(self, model_key: str) -> IsolationReport:
        """Audit binary isolation across all 4 variation subdirectories."""
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
        if variation not in VARIATION_BY_KEY:
            return VariationPushResult(
                variation=variation,
                handle="",
                success=False,
                error=f"Unsupported variation '{variation}'. Allowed: {list(VARIATION_BY_KEY.keys())}",
                dry_run=dry_run,
            )

        variation_dir = self.get_variation_dir(model_key, variation)
        handle = self.get_handle(model_key, variation)

        if not variation_dir.exists():
            return VariationPushResult(
                variation=variation,
                handle=handle,
                success=False,
                error=f"Variation directory does not exist: {variation_dir}",
                dry_run=dry_run,
            )

        # Audit isolation for this variation
        v_report = self.validator.validate_variation(variation, variation_dir, model_key)
        if not v_report.is_valid:
            return VariationPushResult(
                variation=variation,
                handle=handle,
                success=False,
                files=v_report.files_found,
                error=f"Strict isolation audit failed: {'; '.join(v_report.errors)}",
                dry_run=dry_run,
            )

        files = [p for p in variation_dir.iterdir() if p.is_file() or p.is_dir()]
        file_names = [f.name for f in files]

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
        except ImportError:
            return VariationPushResult(
                variation=variation,
                handle=handle,
                success=False,
                files=file_names,
                error="kagglehub SDK is not installed in runtime environment.",
                dry_run=False,
            )

        try:
            var_obj = VARIATION_BY_KEY[variation]
            notes = version_notes or f"Production SOTA Release: {model_key} [{var_obj.label}]"
            logger.info("Uploading %s variation to Kaggle Models Hub: %s ...", variation, handle)

            # If uploading PYTORCH_DEFAULT from the canonical model_dir, upload via a clean transient staging
            # directory containing strictly checkpoints/, README.md, and metrics.csv to guarantee isolation.
            upload_dir = variation_dir
            temp_staging_dir = None
            if variation == PYTORCH_DEFAULT.key and variation_dir == self.get_model_dir(model_key):
                import tempfile
                temp_staging_dir = tempfile.TemporaryDirectory()
                temp_path = Path(temp_staging_dir.name)
                temp_ckpt_dir = temp_path / "checkpoints"
                temp_ckpt_dir.mkdir(parents=True, exist_ok=True)
                src_ckpts = variation_dir / "checkpoints"
                if src_ckpts.exists():
                    import shutil
                    for pth in src_ckpts.glob("*.pth"):
                        shutil.copy2(pth, temp_ckpt_dir / pth.name)
                for root_f in ("README.md", "metrics.csv"):
                    src_f = variation_dir / root_f
                    if src_f.exists():
                        import shutil
                        shutil.copy2(src_f, temp_path / root_f)
                upload_dir = temp_path

            try:
                kagglehub.model_upload(
                    handle=handle,
                    local_model_dir=str(upload_dir),
                    version_notes=notes,
                )
            finally:
                if temp_staging_dir is not None:
                    temp_staging_dir.cleanup()

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

    def sync_kaggle_metadata(self, model_key: str, dry_run: bool = False) -> dict[str, Any]:
        """Synchronize model and variation metadata to Kaggle via REST API."""
        self.ensure_model_metadata(model_key)
        metadata = self.metadata_service.build(model_key)
        if dry_run:
            logger.info("[DRY-RUN] Validated metadata synchronization for %s", model_key)
            return {"status": "validated_dry_run", "model": metadata.model.slug}
        return self.api_sync_service.sync_metadata(metadata)

    def push_model(
        self,
        model_key: str,
        variation: str = "all",
        version_notes: str | None = None,
        dry_run: bool = False,
    ) -> ModelPushResult:
        """Stage, validate, sync metadata, and push specified or all 4 variations for a model."""
        # 1. Stage variations first to ensure canonical file layout
        self.staging_service.stage_variations(model_key)
        self.ensure_model_metadata(model_key)

        isolation_rep = self.validate_isolation(model_key)

        variations_to_push = (
            list(VARIATION_BY_KEY.keys())
            if variation == "all"
            else [variation]
        )

        push_result = ModelPushResult(
            model_key=model_key,
            success=True,
            isolation_report=isolation_rep,
        )

        # 2. Push variation binaries to Kaggle Models Hub
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

        # 3. Synchronize Kaggle Metadata via REST API
        if push_result.success or dry_run:
            try:
                sync_meta_res = self.sync_kaggle_metadata(model_key, dry_run=dry_run)
                push_result.metadata_sync_result = sync_meta_res
                if sync_meta_res.get("errors"):
                    push_result.errors.extend(sync_meta_res["errors"])
            except Exception as meta_exc:
                err_str = f"Metadata synchronization error: {meta_exc}"
                logger.error(err_str)
                push_result.errors.append(err_str)

        return push_result

    def pull_variation(
        self,
        model_key: str,
        variation: str,
        force_download: bool = False,
    ) -> bool:
        """Download one variation from Kaggle Models Hub into canonical model directory."""
        if variation not in VARIATION_BY_KEY:
            raise ValueError(f"Unknown variation '{variation}'. Allowed: {list(VARIATION_BY_KEY.keys())}")

        handle = self.get_handle(model_key, variation)
        target_dir = self.get_variation_dir(model_key, variation)
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            kagglehub = importlib.import_module("kagglehub")
        except ImportError:
            raise RuntimeError("kagglehub SDK is not installed in runtime environment.")

        logger.info("Pulling Kaggle variation %s -> %s", handle, target_dir)
        downloaded_path = kagglehub.model_download(
            handle=handle,
            output_dir=str(target_dir),
            force_download=force_download,
        )

        # For pytorch/default, ensure checkpoints and metrics.csv land in canonical model_root,
        # and eliminate duplicate staging directories so there is strictly one source of truth.
        if variation == PYTORCH_DEFAULT.key:
            model_root = self.get_model_dir(model_key)
            src_ckpt_dir = target_dir / "checkpoints"
            dst_ckpt_dir = model_root / "checkpoints"
            if src_ckpt_dir.exists() and src_ckpt_dir.resolve() != dst_ckpt_dir.resolve():
                dst_ckpt_dir.mkdir(parents=True, exist_ok=True)
                for f in src_ckpt_dir.iterdir():
                    if f.is_file():
                        import shutil
                        dest_f = dst_ckpt_dir / f.name
                        if not dest_f.exists() or f.stat().st_size != dest_f.stat().st_size:
                            shutil.copy2(f, dest_f)
            # Check metrics.csv and README.md in downloaded target_dir
            for rf_name in ("metrics.csv", "README.md"):
                src_rf = target_dir / rf_name
                dst_rf = model_root / rf_name
                if src_rf.exists() and src_rf.resolve() != dst_rf.resolve():
                    import shutil
                    if not dst_rf.exists() or src_rf.stat().st_size != dst_rf.stat().st_size:
                        shutil.copy2(src_rf, dst_rf)

            # Purge duplicate dist/kaggle/pytorch_default directory to prevent dual files on disk
            if target_dir.exists() and target_dir.resolve() != model_root.resolve():
                import shutil
                shutil.rmtree(target_dir, ignore_errors=True)

        return bool(downloaded_path)

    def pull_model(self, model_key: str, force_download: bool = False) -> bool:
        """Download all 4 variations of a model from Kaggle Models Hub."""
        success = True
        for var_key in VARIATION_BY_KEY:
            try:
                var_ok = self.pull_variation(model_key, var_key, force_download=force_download)
                if not var_ok:
                    success = False
            except Exception as exc:
                logger.warning("Failed pulling variation %s for %s: %s", var_key, model_key, exc)
                success = False
        return success
