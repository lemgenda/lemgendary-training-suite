"""Kaggle Model Hub Synchronization Manager (SSOT).

Orchestrates validation, metadata synchronization via the official Kaggle REST API,
and publishing of strictly isolated model variations (PyTorch default, PyTorch FP32,
ONNX FP16, ONNX FP32) to Kaggle Models Hub via kagglehub.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from training.cloud.credentials import resolve_kaggle_credentials
from training.cloud.errors import KAGGLE_API_ERRORS, kagglehub_errors
from training.cloud.hub_versions import VersionLookup
from training.cloud.kaggle_api_sync import KaggleApiSyncService
from training.cloud.kaggle_client import KaggleAuthError
from training.cloud.metadata_generator import KaggleMetadataService
from training.cloud.model_metadata import METADATA_FILENAME, ModelMetadata
from training.cloud.model_pull import merge_downloaded_payload
from training.cloud.push_observer import PushObserver
from training.cloud.sync_manifest import SyncManifest, payload_hashes
from training.cloud.variations import (
    ALL_VARIATIONS,
    CHECKPOINTS_DIR,
    ONNX_FP16,
    ONNX_FP32,
    PYTORCH_DEFAULT,
    PYTORCH_FP32,
    VARIATION_BY_KEY,
    default_payload_files,
    is_default_payload_file,
    selected_variations,
)
from training.cloud.version_notes import ReleaseFacts, build_version_notes, release_facts
from training.ecosystem import kaggle_owner
from training.export.validation import IsolationReport, StrictIsolationValidator
from training.registry import get_model_registry
from training.utils.lazy_imports import load_variation_staging
from training.utils.paths import get_models_hub_root

logger = logging.getLogger("lemtrain.cloud.kaggle_model_sync")


# Upload sub-folder of each dist/kaggle variation other than the default PyTorch payload.
_UPLOAD_SUBDIRS = {
    PYTORCH_FP32.key: "pytorch",
    ONNX_FP32.key: "onnx_fp32",
    ONNX_FP16.key: "onnx_fp16",
}


def _stage_default_payload(variation_dir: Path, temp_path: Path) -> None:
    """Link (or copy) only the ``pytorch/default`` payload into ``temp_path``.

    The payload is :func:`training.cloud.variations.default_payload_files`: ``checkpoints/*.pth``,
    ``README.md`` and ``metrics.csv``; notebooks, ``dist/`` and stray ``.pt`` files never ship.
    """
    (temp_path / CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)
    for rel in default_payload_files(variation_dir):
        try:
            os.link(variation_dir / rel, temp_path / rel)
        except OSError as exc:
            logger.debug("Cannot hard-link %s (%s); copying it", rel, exc)
            shutil.copy2(variation_dir / rel, temp_path / rel)


@dataclass
class VariationPushResult:
    """Result of pushing a single model variation to Kaggle Models Hub."""

    variation: str
    handle: str
    success: bool
    files: list[str] = field(default_factory=list)
    error: str | None = None
    dry_run: bool = False
    skipped: bool = False
    """True when the payload equals the last uploaded one: no new Hub version (K-MDL-5)."""


@dataclass(frozen=True)
class _Upload:
    """One checked upload: the variation, its payload folder, notes and dry-run flag."""

    model_key: str
    variation: str
    payload_dir: Path
    notes: str
    dry_run: bool


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
                    "skipped": v.skipped,
                }
                for k, v in self.variations.items()
            },
            "is_valid_isolation": self.isolation_report.is_valid
            if self.isolation_report
            else False,
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
        self.username = user or kaggle_owner()

        if hub_root:
            self.hub_root = Path(hub_root).resolve()
        else:
            self.hub_root = get_models_hub_root()

        self.registry = get_model_registry()
        self.metadata_service = KaggleMetadataService(owner_slug=self.username)
        self.api_sync_service = KaggleApiSyncService(username=self.username, key=resolved_key)
        self.validator = StrictIsolationValidator()
        self._staging_service = None

    @property
    def staging_service(self):
        """Return the lazily created variation staging service."""
        if self._staging_service is None:
            self._staging_service = load_variation_staging()()
        return self._staging_service

    def get_model_slug(self, model_key: str) -> str:
        """Resolve canonical Kaggle model slug."""
        spec = self.registry.get_model(model_key)
        if spec and spec.kaggle_metadata and spec.kaggle_metadata.slug:
            return spec.kaggle_metadata.slug
        return model_key.replace("_", "-").lower()

    def get_handle(self, model_key: str, variation_key: str) -> str:
        """Construct Kaggle model handle: <owner>/<model_slug>/<framework>/<variation_slug>."""
        slug = self.get_model_slug(model_key)
        variation = VARIATION_BY_KEY.get(variation_key)
        if variation:
            return variation.handle(self.username, slug)
        return f"{self.username}/{slug}/{variation_key}"

    def get_model_dir(self, model_key: str) -> Path:
        """Resolve root folder in LemGendaryModels for model."""
        return self.hub_root / model_key

    def get_variation_dir(self, model_key: str, variation_key: str) -> Path:
        """Local source folder of a variation (model root for ``pytorch/default``)."""
        model_dir = self.get_model_dir(model_key)
        dist_dir = model_dir / "dist" / "kaggle"
        variation = VARIATION_BY_KEY.get(variation_key)
        if not variation:
            return dist_dir / variation_key

        if variation is PYTORCH_DEFAULT:
            # The checkpoints live in the model root; the push stages the isolated payload.
            return model_dir
        return dist_dir / _UPLOAD_SUBDIRS.get(variation.key, variation_key)

    def model_metadata(self, model_key: str) -> ModelMetadata:
        """The validated publish metadata of ``model_key`` (this hub's hand edits merged)."""
        return self.metadata_service.build(model_key, output_dir=self.get_model_dir(model_key))

    def variation_license(self, model_key: str, variation: str) -> str:
        """License of ``variation`` from the model metadata (passed to every upload, K-MDL-6)."""
        metadata = self.model_metadata(model_key)
        section = metadata.variations.get(variation)
        return section.licensing.license if section else metadata.model.publishing.license

    def ensure_model_metadata(self, model_key: str) -> Path:
        """Ensure both model_metadata.yaml and model-metadata.json are materialized.

        ``write_metadata`` validates ``model_key`` against the registry before it
        creates the model folder, so an unknown key leaves no directory behind.
        """
        model_dir = self.get_model_dir(model_key)
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
        facts: ReleaseFacts | None = None,
    ) -> VariationPushResult:
        """Validate and upload a single isolated model variation to Kaggle Models Hub.

        ``pytorch/default`` is staged first (hard links of ``checkpoints/*.pth``,
        ``README.md`` and ``metrics.csv`` only) and the staged copy is validated and uploaded,
        so the model folder's notebooks, ``dist/`` and exports never reach Kaggle.

        A payload whose files hash exactly like the last upload of this variation (recorded in
        ``.kaggle_sync.json``) is skipped: no new Hub version (K-MDL-5). Without
        ``version_notes`` the notes come from :func:`build_version_notes` (``facts`` when the
        caller already read them).
        """
        if variation not in VARIATION_BY_KEY:
            return VariationPushResult(
                variation=variation,
                handle="",
                success=False,
                error=(
                    f"Unsupported variation '{variation}'. Allowed: {list(VARIATION_BY_KEY.keys())}"
                ),
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
        notes = version_notes or build_version_notes(
            model_key,
            facts or release_facts(self.get_model_dir(model_key)),
            label=VARIATION_BY_KEY[variation].label,
        )
        with contextlib.ExitStack() as stack:
            upload_dir = variation_dir
            if variation == PYTORCH_DEFAULT.key:
                self.hub_root.mkdir(parents=True, exist_ok=True)
                upload_dir = Path(
                    stack.enter_context(
                        tempfile.TemporaryDirectory(
                            prefix=f".kaggle-push-{model_key}-", dir=self.hub_root
                        )
                    )
                )
                _stage_default_payload(variation_dir, upload_dir)
            return self._upload_checked(_Upload(model_key, variation, upload_dir, notes, dry_run))

    def _upload_checked(self, upload: _Upload) -> VariationPushResult:
        """Audit the payload, skip it when unchanged, else upload it (or report on dry run)."""
        model_key, variation = upload.model_key, upload.variation
        handle = self.get_handle(model_key, variation)
        blocked = self._payload_problem(
            model_key, variation, upload.payload_dir, handle, upload.dry_run
        )
        if blocked is not None:
            return blocked

        files = [p.name for p in upload.payload_dir.iterdir() if p.is_file() or p.is_dir()]
        hashes = payload_hashes(upload.payload_dir)
        manifest = SyncManifest(self.get_model_dir(model_key))
        if hashes and manifest.pushed_hashes(variation) == hashes:
            logger.info("Skipping %s: payload unchanged since the last upload", handle)
            return VariationPushResult(
                variation, handle, True, files, dry_run=upload.dry_run, skipped=True
            )
        if upload.dry_run:
            logger.info("[DRY-RUN] Verified Kaggle model variation payload %s", handle)
            return VariationPushResult(variation, handle, True, files, dry_run=True)

        error = self._upload(handle, upload, self.variation_license(model_key, variation))
        if error is not None:
            return VariationPushResult(variation, handle, False, files, error=error)
        manifest.record_push(variation, hashes, upload.notes)
        manifest.save()
        logger.info("[OK] Successfully uploaded %s to Kaggle Models Hub", handle)
        return VariationPushResult(variation, handle, True, files)

    @staticmethod
    def _upload(handle: str, upload: _Upload, license_name: str) -> str | None:
        """Upload the payload as a new version of ``handle``; the error text, or None."""
        try:
            kagglehub = importlib.import_module("kagglehub")
        except ImportError:
            return "kagglehub SDK is not installed in runtime environment."
        try:
            logger.info("Uploading %s to Kaggle Models Hub: %s ...", upload.variation, handle)
            kagglehub.model_upload(
                handle=handle,
                local_model_dir=str(upload.payload_dir),
                license_name=license_name,
                version_notes=upload.notes,
            )
        except kagglehub_errors() as exc:
            logger.error("Failed uploading %s to Kaggle Models Hub: %s", handle, exc)
            return str(exc)
        return None

    def _payload_problem(
        self, model_key: str, variation: str, variation_dir: Path, handle: str, dry_run: bool
    ) -> VariationPushResult | None:
        """Return a failed result when the variation folder is missing or not isolated."""
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
        return None

    def sync_kaggle_metadata(self, model_key: str, dry_run: bool = False) -> dict[str, Any]:
        """Synchronize model and variation metadata to Kaggle via REST API."""
        self.ensure_model_metadata(model_key)
        metadata = self.model_metadata(model_key)
        if dry_run:
            logger.info("[DRY-RUN] Validated metadata synchronization for %s", model_key)
            return {"status": "validated_dry_run", "model": metadata.model.slug}
        return self.api_sync_service.sync_metadata(metadata)

    def push_model(
        self,
        model_key: str,
        variation: str = ALL_VARIATIONS,
        version_notes: str | None = None,
        dry_run: bool = False,
        observer: PushObserver | None = None,
    ) -> ModelPushResult:
        """Publish ``variation`` (one key or ``"all"``) of a model: the one push policy.

        Used by the GUI job, the CLI and the notebook release, so all publish identically:

        1. validate the key and write ``model_metadata.yaml``; stage ``dist/kaggle`` (not on a
           dry run, K-MDL-8);
        2. create or update the model page (title, license) BEFORE the first upload; when that
           fails nothing is uploaded (K-MDL-6);
        3. upload each variation with its license and version notes (epoch, metrics, suite
           version), skipping payloads unchanged since the last upload (K-MDL-5);
        4. sync model and instance metadata again (instances created by step 3).
        """
        variations = selected_variations(variation)
        self.ensure_model_metadata(model_key)
        if not dry_run:
            staged = self.staging_service.stage_variations(
                model_key, base_dir=self.get_model_dir(model_key)
            )
            if observer is not None:
                observer.staged(model_key, [f.name for f in staged.staged_files.values()])
        result = ModelPushResult(
            model_key=model_key,
            success=True,
            isolation_report=self.validate_isolation(model_key),
        )
        if not dry_run and not self._prepare_model_page(model_key, result):
            return result

        facts = release_facts(self.get_model_dir(model_key))
        total = len(variations)
        for index, var in enumerate(variations, 1):
            if observer is not None:
                observer.variation_started(index, total, var, self.get_handle(model_key, var))
            var_res = self.push_variation(
                model_key, var, version_notes=version_notes, dry_run=dry_run, facts=facts
            )
            result.variations[var] = var_res
            if not var_res.success:
                result.success = False
                result.errors.append(f"[{var}] {var_res.error}")
            if observer is not None:
                observer.variation_done(index, total, var_res)

        self._finish_metadata(model_key, result, dry_run)
        return result

    def sync_model_page(self, model_key: str) -> dict[str, Any]:
        """Create or update the model page (title, subtitle, license) before uploads (K-MDL-6).

        ``result["model"]`` is None when Kaggle refused both the update and the create; the
        reasons are in ``result["errors"]``.
        """
        try:
            return self.api_sync_service.sync_model(self.model_metadata(model_key))
        except (*KAGGLE_API_ERRORS, KaggleAuthError, yaml.YAMLError) as exc:
            return {"model": None, "variations": {}, "errors": [str(exc)]}

    def _prepare_model_page(self, model_key: str, result: ModelPushResult) -> bool:
        """Sync the model page before any upload; False (and ``result`` failed) on error."""
        page = self.sync_model_page(model_key)
        if page.get("model") is not None:
            return True
        reasons = "; ".join(page.get("errors") or ["unknown error"])
        message = f"Model page metadata sync failed; nothing uploaded: {reasons}"
        logger.error("%s (model=%s)", message, model_key)
        result.success = False
        result.metadata_sync_result = page
        result.errors.append(message)
        return False

    def _finish_metadata(self, model_key: str, result: ModelPushResult, dry_run: bool) -> None:
        """Sync model and instance metadata after the uploads; record errors on ``result``."""
        try:
            sync_meta_res = self.sync_kaggle_metadata(model_key, dry_run=dry_run)
        except (*KAGGLE_API_ERRORS, KaggleAuthError, yaml.YAMLError) as meta_exc:
            err_str = f"Metadata synchronization error: {meta_exc}"
            logger.error(err_str)
            result.errors.append(err_str)
            return
        result.metadata_sync_result = sync_meta_res
        if sync_meta_res.get("errors"):
            result.errors.extend(sync_meta_res["errors"])

    def pull_destination(self, model_key: str, variation_key: str) -> Path:
        """Local folder a pulled variation is merged into.

        ``pytorch/default`` merges into the model root (``LemGendaryModels/<model>``); the
        other variations into their staged ``dist/kaggle`` folder.
        """
        if variation_key == PYTORCH_DEFAULT.key:
            return self.get_model_dir(model_key)
        return self.get_variation_dir(model_key, variation_key)

    def pull_variation(
        self,
        model_key: str,
        variation: str,
        force_download: bool = False,
        version: int | None = None,
    ) -> bool:
        """Download one variation into a temporary folder and merge it into the hub folder.

        Local files are never deleted. ``force_download`` overwrites only the files being
        pulled; without it, newer local checkpoints (by stored epoch) are kept. See
        :mod:`training.cloud.model_pull`. ``version`` pins the Hub version (default latest).
        """
        if variation not in VARIATION_BY_KEY:
            raise ValueError(
                f"Unknown variation '{variation}'. Allowed: {list(VARIATION_BY_KEY.keys())}"
            )

        handle = self.get_handle(model_key, variation)
        if version is not None:
            handle = f"{handle}/{int(version)}"
        dest_dir = self.pull_destination(model_key, variation)

        try:
            kagglehub = importlib.import_module("kagglehub")
        except ImportError as exc:
            raise RuntimeError("kagglehub SDK is not installed in runtime environment.") from exc

        self.hub_root.mkdir(parents=True, exist_ok=True)
        logger.info("Pulling Kaggle variation %s -> %s", handle, dest_dir)
        with tempfile.TemporaryDirectory(
            prefix=f".kaggle-pull-{model_key}-", dir=self.hub_root
        ) as tmp:
            downloaded = kagglehub.model_download(
                handle=handle, output_dir=str(Path(tmp) / "download")
            )
            if not downloaded or not Path(downloaded).is_dir():
                logger.warning("Kaggle returned no folder for %s", handle)
                return False
            report = merge_downloaded_payload(
                Path(downloaded),
                dest_dir,
                force=force_download,
                keep=is_default_payload_file if variation == PYTORCH_DEFAULT.key else None,
            )
        logger.info("Pulled %s: wrote %s; kept newer local %s", handle, report.written, report.kept)
        return True

    def pull_default_if_newer(self, model_key: str, versions: VersionLookup) -> int | None:
        """Pull ``pytorch/default`` only when the Hub has a version newer than the last pull.

        The last merged version is kept in the model folder's ``.kaggle_sync.json``
        (:class:`training.cloud.sync_manifest.SyncManifest`). Returns the version pulled, or
        None when the Hub has nothing newer (no download happens then).
        """
        key = PYTORCH_DEFAULT.key
        latest = versions(self.get_handle(model_key, key))
        manifest = SyncManifest(self.get_model_dir(model_key))
        known = manifest.pulled_version(key)
        if latest is None or (known is not None and latest <= known):
            return None
        if not self.pull_variation(model_key, key, version=latest):
            return None
        manifest.record_pull(key, latest)
        manifest.save()
        return latest

    def pull_model(self, model_key: str, force_download: bool = False) -> bool:
        """Download all 4 variations of a model from Kaggle Models Hub."""
        success = True
        # RuntimeError: raised by pull_variation when the kagglehub SDK is missing.
        pull_errors: tuple[type[Exception], ...] = (*kagglehub_errors(), RuntimeError)
        for var_key in VARIATION_BY_KEY:
            try:
                var_ok = self.pull_variation(model_key, var_key, force_download=force_download)
                if not var_ok:
                    success = False
            except pull_errors as exc:
                logger.warning("Failed pulling variation %s for %s: %s", var_key, model_key, exc)
                success = False
        return success
