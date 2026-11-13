"""Kaggle model push and pull job executors."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from training.cloud.errors import kagglehub_errors
from training.cloud.kaggle_model_sync import KaggleModelSyncManager, VariationPushResult
from training.cloud.variations import selected_variations
from training.registry import get_model_registry
from training.server.state import ServerState

logger = logging.getLogger("lemtrain.server.jobs")


@dataclass(frozen=True)
class PushRequest:
    """One Kaggle model push: the model, its variation ("all" pushes every one) and notes."""

    model_key: str
    variation: str
    version_notes: str | None
    dry_run: bool


@dataclass(frozen=True)
class _PushRun:
    """One telemetry-reported model push: the job, the sync manager and the log prefix."""

    job_id: str
    sync_mgr: Any
    request: PushRequest
    cancel_event: threading.Event
    prefix: str


@dataclass(frozen=True)
class _PullRun:
    """One telemetry-reported model pull."""

    job_id: str
    sync_mgr: Any
    model_key: str
    force_download: bool
    total: int


def _push_succeeded(result: Any) -> bool:
    """True when every variation pushed and the metadata sync reported no errors."""
    return bool(result.success) and (
        not result.metadata_sync_result or not result.metadata_sync_result.get("errors")
    )


class _JobPushObserver:
    """Turns push steps into job log lines (``[UPLOAD]`` lines feed the GUI progress parser)."""

    def __init__(self, broadcast: Callable[[str, str], None], run: _PushRun) -> None:
        self._broadcast = broadcast
        self._run = run

    def staged(self, model_key: str, files: list[str]) -> None:
        """Log the staged file names."""
        if files:
            self._broadcast(
                self._run.job_id,
                f"{self._run.prefix}[STAGING] Staged {len(files)} files for '{model_key}': {files}",
            )

    def variation_started(self, index: int, total: int, variation: str, handle: str) -> None:
        """Log the upload start; stop the push when the job was cancelled."""
        if self._run.cancel_event.is_set():
            raise InterruptedError("Push cancelled")
        pct = int(((index - 1) / total) * 100)
        self._broadcast(
            self._run.job_id,
            f"{self._run.prefix}[UPLOAD] [Step {index}/{total}] PUSHING "
            f"{self._run.request.model_key} [{variation}] -> {handle} ({pct}%)",
        )

    def variation_done(self, index: int, total: int, result: VariationPushResult) -> None:
        """Log the variation outcome (uploaded, unchanged or failed)."""
        model_key, prefix = self._run.request.model_key, self._run.prefix
        if not result.success:
            self._broadcast(
                self._run.job_id,
                f"{prefix}[ERROR] {model_key} [{result.variation}]: {result.error}",
            )
            return
        state = "UNCHANGED (no new version)" if result.skipped else f"DONE files={result.files}"
        self._broadcast(
            self._run.job_id,
            f"{prefix}[UPLOAD] [Step {index}/{total}] {state} {model_key} [{result.variation}] "
            f"({int((index / total) * 100)}%)",
        )


class KaggleSyncJobsMixin:
    """Kaggle model push and pull job executors."""

    # Provided by ``JobManager``.
    project_root: Path
    state: ServerState
    _broadcast_log: Callable[[str, str], None]

    def push_model_with_telemetry(
        self,
        job_id: str,
        sync_mgr: Any,
        request: PushRequest,
        *,
        cancel_event: threading.Event,
        prefix: str = "",
    ) -> Any:
        """Run the one push policy (``push_model``), emitting progress lines and summary logs."""
        run = _PushRun(job_id, sync_mgr, request, cancel_event, prefix)
        result = sync_mgr.push_model(
            model_key=request.model_key,
            variation=request.variation,
            version_notes=request.version_notes,
            dry_run=request.dry_run,
            observer=_JobPushObserver(self._broadcast_log, run),
        )
        self._log_push_summary(run, result, len(result.variations))
        return result

    def _log_push_summary(self, run: _PushRun, result: Any, total: int) -> None:
        model_key, prefix = run.request.model_key, run.prefix
        if _push_succeeded(result):
            self._broadcast_log(
                run.job_id,
                (
                    f"{prefix}[SUCCESS] Model {model_key} pushed successfully to Kaggle ({total} "
                    "variations)"
                ),
            )
            self._broadcast_log(
                run.job_id,
                f"{prefix}[SUCCESS] Kaggle metadata was successfully updated for {model_key}",
            )
        elif result.errors:
            self._broadcast_log(
                run.job_id,
                (
                    f"{prefix}[ERROR] Kaggle publish failed for {model_key}: "
                    f"{'; '.join(result.errors)}"
                ),
            )

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
        if _push_succeeded(res):
            self.state.update_job_status(job_id=job_id, status="completed", metrics=res.to_dict())
        else:
            err = "; ".join(res.errors) if res.errors else "Kaggle push failed"
            self.state.update_job_status(
                job_id=job_id, status="failed", error_message=err, metrics=res.to_dict()
            )

    def _push_fleet_model(
        self, job_id: str, sync_mgr: Any, request: PushRequest, cancel_event: threading.Event
    ) -> tuple[bool, dict[str, Any]]:
        """Push one fleet model; return whether it succeeded and its result record."""
        try:
            push_res = self.push_model_with_telemetry(
                job_id,
                sync_mgr,
                request,
                cancel_event=cancel_event,
                prefix="  ",
            )
            return _push_succeeded(push_res), push_res.to_dict()
        except (*kagglehub_errors(), RuntimeError, yaml.YAMLError) as exc:
            # Per-model failure in a batch push: record it and continue with the next model.
            logger.warning("Kaggle push-all failed for %s: %s", request.model_key, exc)
            self._broadcast_log(
                job_id, f"  [ERROR] Exception processing '{request.model_key}': {exc}"
            )
            return False, {"success": False, "error": str(exc)}

    def _push_fleet(
        self,
        job_id: str,
        models: list[Any],
        template: PushRequest,
        cancel_event: threading.Event,
    ) -> tuple[dict[str, Any], list[bool]] | None:
        """Push every fleet model; return results and outcomes, or None when cancelled."""
        sync_mgr = KaggleModelSyncManager()
        total_models = len(models)
        results: dict[str, Any] = {}
        outcomes: list[bool] = []

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
                return None

            model_key = model_spec.key
            fleet_pct = int(((idx - 1) / total_models) * 100)
            self._broadcast_log(
                job_id,
                (
                    f"[UPLOAD] [Step {idx}/{total_models}] FLEET {model_key} ({model_spec.name}) "
                    f"({fleet_pct}%)"
                ),
            )
            ok, results[model_key] = self._push_fleet_model(
                job_id, sync_mgr, replace(template, model_key=model_key), cancel_event
            )
            outcomes.append(ok)
        return results, outcomes

    def execute_kaggle_push_all(
        self,
        job_id: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Execute fleet synchronization pushing all registered models to Kaggle Models Hub."""
        template = PushRequest(
            "",
            params.get("variation", "all"),
            params.get("version_notes"),
            params.get("dry_run", False),
        )

        models = get_model_registry().list_models()
        total_models = len(models)
        self._broadcast_log(
            job_id,
            f"[KAGGLE-PUSH-ALL] Commencing fleet push for {total_models} models "
            f"(Variation: '{template.variation}', DryRun: {template.dry_run})...",
        )

        fleet = self._push_fleet(job_id, models, template, cancel_event)
        if fleet is None:
            return
        results, outcomes = fleet
        successful_count = sum(outcomes)
        failed_count = len(outcomes) - successful_count

        metrics: dict[str, Any] = {
            "total_models": total_models,
            "successful_count": successful_count,
            "failed_count": failed_count,
            "variation": template.variation,
            "dry_run": template.dry_run,
            "models": results,
        }
        self.state.update_job_status(
            job_id=job_id,
            status="completed" if (successful_count > 0 or template.dry_run) else "failed",
            metrics=metrics,
        )
        self._broadcast_log(
            job_id,
            f"[COMPLETED] Kaggle fleet push finished: {successful_count} succeeded, "
            f"{failed_count} skipped/failed out of {total_models} models (100%).",
        )

    def _pull_one(self, run: _PullRun, i: int, var_key: str) -> str | None:
        """Pull one variation; return its error, or None when it was pulled."""
        model_key, total = run.model_key, run.total
        pct_start = int(((i - 1) / total) * 100)
        handle = run.sync_mgr.get_handle(model_key, var_key)
        self._broadcast_log(
            run.job_id,
            (
                f"[DOWNLOAD] [Step {i}/{total}] PULLING {model_key} [{var_key}] <- {handle} "
                f"({pct_start}%)"
            ),
        )

        try:
            ok = run.sync_mgr.pull_variation(model_key, var_key, force_download=run.force_download)
            pct_done = int((i / total) * 100)
            if ok:
                target_dir = run.sync_mgr.pull_destination(model_key, var_key)
                self._broadcast_log(
                    run.job_id,
                    (
                        f"[DOWNLOAD] [Step {i}/{total}] DONE {model_key} [{var_key}] -> "
                        f"{target_dir} ({pct_done}%)"
                    ),
                )
                return None
            self._broadcast_log(run.job_id, f"[ERROR] {model_key} [{var_key}]: pull returned False")
            return f"[{var_key}] Pull returned False"
        except (*kagglehub_errors(), RuntimeError) as exc:
            # RuntimeError: kagglehub SDK missing (raised by pull_variation).
            logger.warning("Kaggle pull failed for %s [%s]: %s", model_key, var_key, exc)
            self._broadcast_log(run.job_id, f"[ERROR] {model_key} [{var_key}]: {exc}")
            return f"[{var_key}] {exc}"

    def _finish_pull(self, run: _PullRun, success_count: int, errors: list[str]) -> None:
        job_id, model_key, total = run.job_id, run.model_key, run.total
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

    def execute_kaggle_pull(
        self,
        job_id: str,
        model_key: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Download all 4 model variations from Kaggle Models Hub with live telemetry."""
        variation = params.get("variation", "all")
        sync_mgr = KaggleModelSyncManager(username=params.get("username"))
        vars_to_pull = selected_variations(variation)
        run = _PullRun(
            job_id, sync_mgr, model_key, params.get("force_download", False), len(vars_to_pull)
        )

        self._broadcast_log(
            job_id,
            f"[DOWNLOAD] Pulling {run.total} variation(s) for '{model_key}' "
            "from Kaggle Models Hub...",
        )

        success_count = 0
        errors: list[str] = []
        for i, var_key in enumerate(vars_to_pull, 1):
            if cancel_event.is_set():
                raise InterruptedError("Pull cancelled")
            error = self._pull_one(run, i, var_key)
            if error is None:
                success_count += 1
            else:
                errors.append(error)

        self._finish_pull(run, success_count, errors)
