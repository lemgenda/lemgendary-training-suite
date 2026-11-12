"""Kaggle kernel training and monitoring job executors.

``kaggle_train`` builds the kernel bundle (the generated training notebook plus
``kernel-metadata.json``, :mod:`training.cloud.kernel_bundle`), pushes it, watches the session
(:mod:`training.cloud.kernel_watch`) and, when it completes, merges the kernel output's
checkpoints and metrics into ``LemGendaryModels/<model>``. ``kaggle_monitor`` watches an
existing kernel without pushing.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.cloud.credentials import resolve_kaggle_credentials
from training.cloud.errors import KAGGLE_API_ERRORS, kagglehub_errors
from training.cloud.hub_versions import kaggle_version_lookup
from training.cloud.kaggle_client import KaggleAuthError, authenticated_kaggle_api
from training.cloud.kaggle_model_sync import KaggleModelSyncManager
from training.cloud.kernel_bundle import BundleRequest, model_kernel_handle
from training.cloud.kernel_launch import launch_kernel
from training.cloud.kernel_output import retrieve_kernel_output
from training.cloud.kernel_status import CANCELLED_STATUSES, COMPLETE, QUEUED, PollBackoff
from training.cloud.kernel_watch import KernelWatcher, WatchHooks, WatchResult
from training.cloud.session_deadline import load_session_config
from training.server.state import ServerState

logger = logging.getLogger("lemtrain.server.jobs")

DEFAULT_POLL_SECONDS = 30
"""First status poll interval; it backs off to :data:`MAX_POLL_SECONDS` while unchanged."""
MAX_POLL_SECONDS = 300


def _pull_errors() -> tuple[type[Exception], ...]:
    # RuntimeError: kagglehub SDK missing (raised by pull_variation).
    return (RuntimeError, *kagglehub_errors())


def _authenticated_api(user: str, key: str, failure_hint: str) -> Any:
    """Return a Kaggle API authenticated with the resolved ``user``/``key``.

    :class:`KaggleAuthError` (a ``RuntimeError``) fails the job instead of the SDK's
    ``exit(1)`` silently killing the job thread.
    """
    try:
        return authenticated_kaggle_api(user, key)
    except KaggleAuthError as e:
        raise KaggleAuthError(f"{e}{failure_hint}", e.http_status) from e


@dataclass(frozen=True)
class _KernelWatch:
    """One job's watch of one Kaggle kernel."""

    job_id: str
    model_key: str
    user: str
    api: Any
    kernel_slug: str
    auto_pull: bool


class KaggleRunJobsMixin:
    """Kaggle kernel training and monitoring job executors."""

    # Provided by ``JobManager``.
    project_root: Path
    state: ServerState
    _broadcast_log: Callable[[str, str], None]

    def _hub_sync(self, user: str) -> KaggleModelSyncManager:
        return KaggleModelSyncManager(username=user)

    def _pull_newer(self, watch: _KernelWatch) -> None:
        """Pull ``pytorch/default`` when the Hub has a newer version than the last pull."""
        try:
            version = self._hub_sync(watch.user).pull_default_if_newer(
                watch.model_key, kaggle_version_lookup(watch.api)
            )
        except _pull_errors() as exc:
            self._broadcast_log(watch.job_id, f"[WARN] Auto-pull notice: {exc}")
            return
        if version is not None:
            self._broadcast_log(
                watch.job_id,
                f"[AUTO-PULL] Hub version {version} of {watch.model_key} merged locally.",
            )

    def _watch(
        self, watch: _KernelWatch, params: dict[str, Any], cancel_event: threading.Event
    ) -> WatchResult:
        poll = max(5, int(params.get("poll_interval") or DEFAULT_POLL_SECONDS))
        hooks = WatchHooks(
            emit=lambda line: self._broadcast_log(watch.job_id, line),
            cancelled=cancel_event.is_set,
            sleep=cancel_event.wait,
            on_poll=(lambda: self._pull_newer(watch))
            if watch.auto_pull and watch.model_key
            else None,
        )
        backoff = PollBackoff(initial=poll, maximum=max(poll, MAX_POLL_SECONDS))
        return KernelWatcher(watch.api, watch.kernel_slug, backoff, hooks).watch(QUEUED)

    def _retrieve_output(self, watch: _KernelWatch) -> bool:
        """Merge the kernel output's checkpoints and metrics; True after a deadline stop."""
        hub_root = self._hub_sync(watch.user).hub_root
        try:
            output = retrieve_kernel_output(watch.api, watch.kernel_slug, watch.model_key, hub_root)
        except KAGGLE_API_ERRORS as exc:
            self._broadcast_log(watch.job_id, f"[WARN] Kernel output retrieval notice: {exc}")
            return False
        self._broadcast_log(
            watch.job_id,
            f"[SYNC] Kernel output merged into LemGendaryModels/{watch.model_key}: "
            f"{len(output.report.written)} written, {len(output.report.kept)} newer local "
            f"kept (deadline stop: {output.deadline_stop}).",
        )
        if watch.auto_pull:
            self._pull_newer(watch)
        return output.deadline_stop

    def _fail(self, watch: _KernelWatch, result: WatchResult) -> None:
        """Record a kernel that ended cancelled or with an error."""
        status = result.status or ""
        metrics = {"kaggle_status": status, "kernel_slug": watch.kernel_slug}
        if status in CANCELLED_STATUSES:
            self.state.update_job_status(
                job_id=watch.job_id,
                status="cancelled",
                error_message=f"Kaggle kernel was cancelled ({status})",
                metrics=metrics,
            )
            self._broadcast_log(watch.job_id, f"[CANCEL] Kaggle kernel cancelled: {status}")
            return
        detail = f": {result.failure}" if result.failure else ""
        self.state.update_job_status(
            job_id=watch.job_id,
            status="failed",
            error_message=f"Kaggle kernel ended with status {status}{detail}",
            metrics=metrics,
        )
        self._broadcast_log(watch.job_id, f"[ERROR] Kaggle kernel ended: {status}{detail}")

    def _complete(self, watch: _KernelWatch, extra: dict[str, Any] | None = None) -> None:
        """Record a completed kernel."""
        metrics = {"kaggle_status": COMPLETE, "kernel_slug": watch.kernel_slug, **(extra or {})}
        self.state.update_job_status(job_id=watch.job_id, status="completed", metrics=metrics)
        self._broadcast_log(watch.job_id, f"[SUCCESS] Kaggle kernel completed: {watch.kernel_slug}")

    def _detach(self, watch: _KernelWatch) -> None:
        self.state.update_job_status(
            job_id=watch.job_id, status="cancelled", error_message="Cancelled by user request"
        )
        self._broadcast_log(
            watch.job_id,
            f"[CANCEL] Telemetry detached from {watch.kernel_slug}; the kernel may keep "
            "running on Kaggle.",
        )

    def execute_kaggle_training(
        self,
        job_id: str,
        model_key: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Push the model's kernel bundle, watch the session and merge its output."""
        user, key = resolve_kaggle_credentials()
        accelerator = params.get("accelerator")
        self._broadcast_log(
            job_id, f"[KAGGLE] Authenticating as '{user}' (accelerator: {accelerator or 'default'})"
        )
        api = _authenticated_api(user, key, ". Please configure the Kaggle API token.")
        chain = load_session_config(self.project_root).with_chain(
            params.get("auto_resume"), params.get("max_resumes")
        )
        resumes = 0
        while True:
            watch = self._launch(job_id, model_key, user, api, params)
            result = self._watch(watch, params, cancel_event)
            if result.status is None:
                self._detach(watch)
                return
            if result.status != COMPLETE:
                self._fail(watch, result)
                return
            deadline_stop = self._retrieve_output(watch)
            if not (deadline_stop and chain.auto_resume and resumes < chain.max_resumes):
                self._complete(watch, {"deadline_stop": deadline_stop, "resumes": resumes})
                return
            resumes += 1
            self._broadcast_log(
                job_id,
                f"[CHAIN] Session stopped at its deadline; pushing resume session "
                f"{resumes}/{chain.max_resumes} (auto_resume).",
            )

    def _launch(
        self, job_id: str, model_key: str, user: str, api: Any, params: dict[str, Any]
    ) -> _KernelWatch:
        """Push the model's kernel bundle and return the watch of the new session."""
        launch = launch_kernel(
            api,
            BundleRequest(model_key, kernel_owner=user, accelerator_id=params.get("accelerator")),
        )
        self._broadcast_log(
            job_id,
            f"[KAGGLE] Kernel pushed: {launch.handle} (version {launch.version_number}, "
            f"machine shape {launch.metadata.get('machine_shape', 'CPU')}, "
            f"datasets {launch.metadata['dataset_sources']}, "
            f"models {launch.metadata['model_sources']})",
        )
        return _KernelWatch(
            job_id, model_key, user, api, launch.handle, bool(params.get("auto_pull", True))
        )

    def execute_kaggle_monitor(
        self,
        job_id: str,
        model_key: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Watch an existing Kaggle kernel without pushing."""
        user, key = resolve_kaggle_credentials()
        kernel_slug = params.get("kernel_slug") or model_kernel_handle(user, model_key)
        self._broadcast_log(job_id, f"[KAGGLE] Connecting to kernel '{kernel_slug}'...")
        api = _authenticated_api(user, key, "")
        watch = _KernelWatch(
            job_id,
            params.get("model") or "",
            user,
            api,
            kernel_slug,
            bool(params.get("auto_pull", False)),
        )
        result = self._watch(watch, params, cancel_event)
        if result.status is None:
            self._detach(watch)
            return
        if result.status != COMPLETE:
            self._fail(watch, result)
            return
        if watch.auto_pull and watch.model_key:
            self._retrieve_output(watch)
        self._complete(watch)
