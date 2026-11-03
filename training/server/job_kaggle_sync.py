"""Kaggle model push and pull job executors."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import yaml

from training.cloud.errors import KAGGLE_API_ERRORS, kagglehub_errors
from training.cloud.kaggle_model_sync import KaggleModelSyncManager, ModelPushResult
from training.cloud.variations import VARIATION_BY_KEY
from training.export.staging import VariationStagingService
from training.registry import get_model_registry

logger = logging.getLogger("lemtrain.server.jobs")


@dataclass(frozen=True)
class PushRequest:
    """One Kaggle model push: the model, its variation ("all" pushes every one) and notes."""

    model_key: str
    variation: str
    version_notes: str | None
    dry_run: bool


class KaggleSyncJobsMixin:
    """Kaggle model push and pull job executors."""

    def push_model_with_telemetry(
        self,
        job_id: str,
        sync_mgr: Any,
        request: PushRequest,
        *,
        cancel_event: threading.Event,
        prefix: str = "",
    ) -> Any:
        """Stage, then push each variation one by one, emitting progress lines and summary logs."""
        model_key, variation = request.model_key, request.variation
        version_notes, dry_run = request.version_notes, request.dry_run
        sync_mgr.ensure_model_metadata(model_key)
        staging_res = VariationStagingService().stage_variations(model_key)
        if staging_res.staged_files:
            staged_names = [f.name for f in staging_res.staged_files.values()]
            self._broadcast_log(
                job_id,
                (
                    f"{prefix}[STAGING] Staged {len(staged_names)} files for '{model_key}': "
                    f"{staged_names}"
                ),
            )

        variations = list(VARIATION_BY_KEY.keys()) if variation == "all" else [variation]
        result = ModelPushResult(
            model_key=model_key,
            success=True,
            isolation_report=sync_mgr.validate_isolation(model_key),
        )
        total = len(variations)

        # 1. Push all variations with progress
        for i, var in enumerate(variations, 1):
            if cancel_event.is_set():
                raise InterruptedError("Push cancelled")

            pct_start = int(((i - 1) / total) * 100)
            handle = sync_mgr.get_handle(model_key, var)
            # Format with [UPLOAD] keyword so progressParser.ts detects upload operation
            self._broadcast_log(
                job_id,
                (
                    f"{prefix}[UPLOAD] [Step {i}/{total}] PUSHING {model_key} [{var}] -> {handle} "
                    f"({pct_start}%)"
                ),
            )
            var_res = sync_mgr.push_variation(
                model_key=model_key,
                variation=var,
                version_notes=version_notes,
                dry_run=dry_run,
            )
            result.variations[var] = var_res
            pct_done = int((i / total) * 100)
            if var_res.success:
                self._broadcast_log(
                    job_id,
                    (
                        f"{prefix}[UPLOAD] [Step {i}/{total}] DONE {model_key} [{var}] "
                        f"files={var_res.files} ({pct_done}%)"
                    ),
                )
            else:
                result.success = False
                result.errors.append(f"[{var}] {var_res.error}")
                self._broadcast_log(job_id, f"{prefix}[ERROR] {model_key} [{var}]: {var_res.error}")

        # 2. Sync metadata via API unconditionally (publishes model card & instance docs)
        try:
            sync_meta_res = sync_mgr.sync_kaggle_metadata(model_key, dry_run=dry_run)
            result.metadata_sync_result = sync_meta_res
            if sync_meta_res.get("errors"):
                result.errors.extend(sync_meta_res["errors"])
        except (*KAGGLE_API_ERRORS, yaml.YAMLError) as meta_exc:
            err_str = f"Metadata synchronization error: {meta_exc}"
            logger.error("%s (model=%s)", err_str, model_key)
            result.errors.append(err_str)
            self._broadcast_log(job_id, f"{prefix}[ERROR] {model_key} metadata sync: {err_str}")

        # 3. Emit exact 2 clean summary log entries on success
        if result.success and (
            not result.metadata_sync_result or not result.metadata_sync_result.get("errors")
        ):
            self._broadcast_log(
                job_id,
                (
                    f"{prefix}[SUCCESS] Model {model_key} pushed successfully to Kaggle ({total} "
                    "variations)"
                ),
            )
            self._broadcast_log(
                job_id,
                f"{prefix}[SUCCESS] Kaggle metadata was successfully updated for {model_key}",
            )
        elif result.errors:
            self._broadcast_log(
                job_id,
                (
                    f"{prefix}[ERROR] Kaggle publish failed for {model_key}: "
                    f"{'; '.join(result.errors)}"
                ),
            )

        return result

    def execute_kaggle_push(
        self,
        job_id: str,
        model_key: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Push all 4 variations of a single model to Kaggle Models Hub with live telemetry."""
        variation = params.get("variation", "all")
        dry_run = params.get("dry_run", False)
        sync_mgr = KaggleModelSyncManager(username=params.get("username"))
        res = self.push_model_with_telemetry(
            job_id,
            sync_mgr,
            PushRequest(model_key, variation, params.get("version_notes"), dry_run),
            cancel_event=cancel_event,
        )
        if res.success and (
            not res.metadata_sync_result or not res.metadata_sync_result.get("errors")
        ):
            self.state.update_job_status(job_id=job_id, status="completed", metrics=res.to_dict())
        else:
            err = "; ".join(res.errors) if res.errors else "Kaggle push failed"
            self.state.update_job_status(
                job_id=job_id, status="failed", error_message=err, metrics=res.to_dict()
            )

    def execute_kaggle_push_all(
        self,
        job_id: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Execute fleet synchronization pushing all registered models to Kaggle Models Hub."""
        variation = params.get("variation", "all")
        version_notes = params.get("version_notes")
        dry_run = params.get("dry_run", False)

        reg = get_model_registry()
        models = reg.list_models()
        total_models = len(models)
        self._broadcast_log(
            job_id,
            f"[KAGGLE-PUSH-ALL] Commencing fleet push for {total_models} models "
            f"(Variation: '{variation}', DryRun: {dry_run})...",
        )

        sync_mgr = KaggleModelSyncManager()

        results: dict[str, Any] = {}
        successful_count = 0
        failed_count = 0

        for idx, model_spec in enumerate(models, 1):
            if cancel_event.is_set():
                self.state.update_job_status(
                    job_id=job_id,
                    status="cancelled",
                    error_message="Cancelled by user request",
                )
                self._broadcast_log(
                    job_id, f"[CANCEL] Fleet push cancelled before processing '{model_spec.key}'."
                )
                return

            model_key = model_spec.key
            fleet_pct = int(((idx - 1) / total_models) * 100)
            self._broadcast_log(
                job_id,
                (
                    f"[UPLOAD] [Step {idx}/{total_models}] FLEET {model_key} ({model_spec.name}) "
                    f"({fleet_pct}%)"
                ),
            )

            try:
                push_res = self.push_model_with_telemetry(
                    job_id,
                    sync_mgr,
                    PushRequest(model_key, variation, version_notes, dry_run),
                    cancel_event=cancel_event,
                    prefix="  ",
                )

                if push_res.success and (
                    not push_res.metadata_sync_result
                    or not push_res.metadata_sync_result.get("errors")
                ):
                    successful_count += 1
                else:
                    failed_count += 1

                results[model_key] = push_res.to_dict()

            except (*kagglehub_errors(), RuntimeError, yaml.YAMLError) as exc:
                # Per-model failure in a batch push: record it and continue with the next model.
                logger.warning("Kaggle push-all failed for %s: %s", model_key, exc)
                failed_count += 1
                self._broadcast_log(job_id, f"  [ERROR] Exception processing '{model_key}': {exc}")
                results[model_key] = {"success": False, "error": str(exc)}

        overall_status = "completed" if (successful_count > 0 or dry_run) else "failed"
        metrics: dict[str, Any] = {
            "total_models": total_models,
            "successful_count": successful_count,
            "failed_count": failed_count,
            "variation": variation,
            "dry_run": dry_run,
            "models": results,
        }
        self.state.update_job_status(job_id=job_id, status=overall_status, metrics=metrics)
        self._broadcast_log(
            job_id,
            f"[COMPLETED] Kaggle fleet push finished: {successful_count} succeeded, "
            f"{failed_count} skipped/failed out of {total_models} models (100%).",
        )

    def execute_kaggle_pull(
        self,
        job_id: str,
        model_key: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Download all 4 model variations from Kaggle Models Hub with live telemetry."""
        variation = params.get("variation", "all")
        force_download = params.get("force_download", False)

        sync_mgr = KaggleModelSyncManager(username=params.get("username"))
        vars_to_pull = list(VARIATION_BY_KEY.keys()) if variation == "all" else [variation]
        total = len(vars_to_pull)

        self._broadcast_log(
            job_id,
            f"[DOWNLOAD] Pulling {total} variation(s) for '{model_key}' from Kaggle Models Hub...",
        )

        success_count = 0
        errors: list[str] = []

        for i, var_key in enumerate(vars_to_pull, 1):
            if cancel_event.is_set():
                raise InterruptedError("Pull cancelled")

            pct_start = int(((i - 1) / total) * 100)
            handle = sync_mgr.get_handle(model_key, var_key)
            self._broadcast_log(
                job_id,
                (
                    f"[DOWNLOAD] [Step {i}/{total}] PULLING {model_key} [{var_key}] <- {handle} "
                    f"({pct_start}%)"
                ),
            )

            try:
                ok = sync_mgr.pull_variation(model_key, var_key, force_download=force_download)
                pct_done = int((i / total) * 100)
                if ok:
                    success_count += 1
                    target_dir = sync_mgr.get_variation_dir(model_key, var_key)
                    self._broadcast_log(
                        job_id,
                        (
                            f"[DOWNLOAD] [Step {i}/{total}] DONE {model_key} [{var_key}] -> "
                            f"{target_dir} ({pct_done}%)"
                        ),
                    )
                else:
                    errors.append(f"[{var_key}] Pull returned False")
                    self._broadcast_log(
                        job_id, f"[ERROR] {model_key} [{var_key}]: pull returned False"
                    )
            except (*kagglehub_errors(), RuntimeError) as exc:
                # RuntimeError: kagglehub SDK missing (raised by pull_variation).
                logger.warning("Kaggle pull failed for %s [%s]: %s", model_key, var_key, exc)
                errors.append(f"[{var_key}] {exc}")
                self._broadcast_log(job_id, f"[ERROR] {model_key} [{var_key}]: {exc}")

        if success_count == total:
            self.state.update_job_status(
                job_id=job_id,
                status="completed",
                metrics={"model_key": model_key, "variations_pulled": total},
            )
            self._broadcast_log(
                job_id,
                f"[SUCCESS] Model {model_key} pulled successfully from Kaggle ({total} variations)",
            )
        else:
            err_summary = "; ".join(errors)
            self.state.update_job_status(
                job_id=job_id,
                status="failed",
                error_message=err_summary,
                metrics={"model_key": model_key, "success_count": success_count, "errors": errors},
            )
            self._broadcast_log(
                job_id,
                (
                    f"[ERROR] Pull incomplete for {model_key}: {success_count}/{total} succeeded. "
                    f"{err_summary}"
                ),
            )
