"""Kaggle kernel training and monitoring job executors."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi

from training.cloud.credentials import resolve_kaggle_credentials
from training.cloud.errors import KAGGLE_API_ERRORS
from training.cloud.kaggle_hub import KaggleHubManager
from training.kaggle_accounts import authenticate_kaggle_user
from training.kaggle_cloud_manager import pull_kaggle_artifacts
from training.kaggle_stream import fetch_new_log_lines, is_epoch_milestone
from training.server.state import ServerState

logger = logging.getLogger("lemtrain.server.jobs")


# pull_kaggle_artifacts reports most failures itself; RuntimeError covers a missing kagglehub SDK.
_PULL_ERRORS: tuple[type[Exception], ...] = (RuntimeError, *KAGGLE_API_ERRORS)

_TERMINAL_STATUSES = ("COMPLETE", "ERROR", "CANCELACK")


def _authenticated_api(user: str, key: str, failure_hint: str) -> Any:
    """Return a Kaggle API for ``user``/``key``, else for the ambient credentials."""
    api = authenticate_kaggle_user(user, key) if (user and key) else None
    if api:
        return api
    try:
        api = KaggleApi()
        api.authenticate()
    except KAGGLE_API_ERRORS as e:
        raise RuntimeError(f"Kaggle authentication failed: {e}{failure_hint}") from e
    return api


@dataclass(frozen=True)
class _KernelWatch:
    """One job's telemetry stream of one Kaggle kernel."""

    job_id: str
    model_key: str
    user: str
    api: Any
    kernel_slug: str
    auto_pull: bool
    milestone_message: str


class KaggleRunJobsMixin:
    """Kaggle kernel training and monitoring job executors."""

    # Provided by ``JobManager``.
    project_root: Path
    state: ServerState
    _broadcast_log: Callable[[str, str], None]

    def _push_kernel(self, job_id: str, api: Any, kernel_dir: Path, kernel_slug: str) -> None:
        """Push the kernel bundle with the Python SDK, falling back to the Kaggle CLI."""
        try:
            api.kernels_push(str(kernel_dir))
            self._broadcast_log(job_id, f"[KAGGLE] Kernel successfully pushed: {kernel_slug}")
        except KAGGLE_API_ERRORS as sdk_err:
            self._broadcast_log(
                job_id, f"[WARN] Python SDK push notice: {sdk_err}. Trying CLI fallback..."
            )
            cflags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                if sys.platform == "win32"
                else 0
            )
            res = subprocess.run(
                ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
                capture_output=True,
                check=False,
                text=True,
                timeout=60,
                creationflags=cflags,
            )
            if res.returncode == 0:
                self._broadcast_log(job_id, f"[KAGGLE] CLI push succeeded: {res.stdout.strip()}")
            else:
                raise RuntimeError(f"CLI push failed: {res.stderr.strip()}") from sdk_err

    def _poll_kernel_status(self, watch: _KernelWatch, last_status: str) -> str:
        """Return the kernel's cloud status (``last_status`` when the poll fails)."""
        try:
            st = watch.api.kernels_status(watch.kernel_slug)
            status_val = str(getattr(st, "status", st)).rsplit(".", maxsplit=1)[-1].upper()
            if status_val != last_status:
                self._broadcast_log(
                    watch.job_id,
                    f"[{time.strftime('%H:%M:%S')}] [KAGGLE STATUS] Cloud status: {status_val}",
                )
        except KAGGLE_API_ERRORS as exc:
            logger.debug(
                "Kaggle status poll failed for %s; keeping %s: %s",
                watch.kernel_slug,
                last_status,
                exc,
            )
            return last_status
        return status_val

    def _pull_artifacts(self, watch: _KernelWatch, notice: str) -> bool:
        """Pull the model artifacts; on failure log ``notice: <error>`` and return False."""
        try:
            pull_kaggle_artifacts(watch.model_key, username=watch.user)
        except _PULL_ERRORS as pull_err:
            self._broadcast_log(watch.job_id, f"{notice}: {pull_err}")
            return False
        return True

    def _stream_kernel_logs(self, watch: _KernelWatch, printed_lines_count: int) -> int:
        """Broadcast new kernel log lines; return the new printed line count."""
        try:
            new_lines, printed_lines_count = fetch_new_log_lines(
                watch.api, watch.kernel_slug, printed_lines_count
            )
            for line in new_lines:
                self._broadcast_log(watch.job_id, line)
                if watch.auto_pull and watch.model_key and is_epoch_milestone(line):
                    self._broadcast_log(watch.job_id, watch.milestone_message)
                    self._pull_artifacts(watch, "[WARN] Auto-pull notice")
        except KAGGLE_API_ERRORS as exc:
            logger.debug(
                "Kaggle log poll failed for %s; retrying next interval: %s", watch.kernel_slug, exc
            )
        return printed_lines_count

    def _watch_kernel(
        self,
        watch: _KernelWatch,
        last_status: str,
        poll_interval: Any,
        cancel_event: threading.Event,
    ) -> str | None:
        """Stream the kernel until a terminal status (returned) or cancellation (None)."""
        printed_lines_count = 0
        while not cancel_event.is_set():
            last_status = self._poll_kernel_status(watch, last_status)
            printed_lines_count = self._stream_kernel_logs(watch, printed_lines_count)
            if last_status in _TERMINAL_STATUSES:
                return last_status
            time.sleep(poll_interval)
        return None

    def _finish_kaggle_training(self, watch: _KernelWatch, status_val: str) -> None:
        """Record the end of a Kaggle training job (pulling final artifacts on success)."""
        job_id, model_key = watch.job_id, watch.model_key
        if status_val != "COMPLETE":
            self.state.update_job_status(
                job_id=job_id,
                status="failed",
                error_message=f"Kaggle kernel halted with status: {status_val}",
            )
            self._broadcast_log(job_id, f"[ERROR] Kaggle cloud job ended with status: {status_val}")
            return
        if watch.auto_pull:
            self._broadcast_log(
                job_id,
                f"[SYNC] Cloud execution complete. Pulling final artifacts for {model_key}...",
            )
            if self._pull_artifacts(watch, "[WARN] Artifact pull notice"):
                self._broadcast_log(
                    job_id, f"[SUCCESS] Final artifacts saved to LemGendaryModels/{model_key}!"
                )
        self.state.update_job_status(
            job_id=job_id,
            status="completed",
            metrics={"kaggle_status": "COMPLETE", "kernel_slug": watch.kernel_slug},
        )
        self._broadcast_log(
            job_id, f"[SUCCESS] Kaggle training completed successfully for {model_key}!"
        )

    def execute_kaggle_training(
        self,
        job_id: str,
        model_key: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Launch training on Kaggle Cloud, stream telemetry, and auto-pull checkpoints."""
        gpu = params.get("gpu", "T4")
        auto_pull = params.get("auto_pull", True)

        user, key = resolve_kaggle_credentials(
            override_user=params.get("username"), override_key=params.get("key")
        )
        self._broadcast_log(
            job_id, f"[KAGGLE] Authenticating Kaggle API for user: '{user}' (Accelerator: {gpu})..."
        )
        api = _authenticated_api(user, key, ". Please configure Kaggle username and API key.")

        mgr = KaggleHubManager(username=user, key=key)
        kernel_slug = mgr.get_kernel_slug(model_key)
        self._broadcast_log(
            job_id, f"[KAGGLE] Packaging standalone kernel bundle for {model_key}..."
        )
        kernel_dir = mgr.create_cloud_kernel_bundle(model_key, gpu=gpu)
        self._broadcast_log(
            job_id, f"[KAGGLE] Deploying kernel bundle {kernel_slug} to Kaggle Cloud..."
        )
        self._push_kernel(job_id, api, kernel_dir, kernel_slug)

        self._broadcast_log(
            job_id,
            f"[KAGGLE] Telemetry connected. Monitoring {kernel_slug} (Auto-pull: {auto_pull})...",
        )
        watch = _KernelWatch(
            job_id,
            model_key,
            user,
            api,
            kernel_slug,
            auto_pull,
            f"[AUTO-PULL] Epoch milestone detected in logs. Syncing artifacts for {model_key}...",
        )
        status_val = self._watch_kernel(
            watch, "QUEUED", params.get("poll_interval", 5), cancel_event
        )
        if status_val is not None:
            self._finish_kaggle_training(watch, status_val)
            return

        if cancel_event.is_set():
            self.state.update_job_status(
                job_id=job_id, status="cancelled", error_message="Cancelled by user request"
            )
            self._broadcast_log(
                job_id,
                (
                    f"[CANCEL] Telemetry stream detached for {kernel_slug}. Remote kernel may "
                    "continue running on Kaggle."
                ),
            )

    def _finish_kaggle_monitor(self, watch: _KernelWatch, status_val: str) -> None:
        """Record the end of a Kaggle monitor job (pulling final artifacts when asked)."""
        job_id, model_key = watch.job_id, watch.model_key
        if status_val == "COMPLETE" and watch.auto_pull and model_key:
            self._broadcast_log(
                job_id,
                f"[SYNC] Cloud run completed. Pulling final artifacts for {model_key}...",
            )
            if self._pull_artifacts(watch, "[WARN] Pull notice"):
                self._broadcast_log(
                    job_id,
                    f"[SUCCESS] Final artifacts saved to LemGendaryModels/{model_key}!",
                )

        self.state.update_job_status(
            job_id=job_id,
            status="completed" if status_val == "COMPLETE" else "failed",
            metrics={"kaggle_status": status_val, "kernel_slug": watch.kernel_slug},
        )
        self._broadcast_log(job_id, f"[TERMINAL] Job ended with status: {status_val}")

    def execute_kaggle_monitor(
        self,
        job_id: str,
        model_key: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Stream live logs from an existing or active Kaggle kernel without pushing."""
        kernel_slug = params.get("kernel_slug")

        user, key = resolve_kaggle_credentials(
            override_user=params.get("username"), override_key=params.get("key")
        )
        if not kernel_slug:
            mgr = KaggleHubManager(username=user)
            kernel_slug = mgr.get_kernel_slug(model_key)

        self._broadcast_log(
            job_id, f"[KAGGLE] Connecting to telemetry stream for '{kernel_slug}'..."
        )
        api = _authenticated_api(user, key, "")

        watch = _KernelWatch(
            job_id,
            model_key,
            user,
            api,
            kernel_slug,
            params.get("auto_pull", False),
            f"[AUTO-PULL] Checkpoint milestone detected. Syncing {model_key}...",
        )
        status_val = self._watch_kernel(
            watch, "UNKNOWN", params.get("poll_interval", 5), cancel_event
        )
        if status_val is not None:
            self._finish_kaggle_monitor(watch, status_val)
            return

        if cancel_event.is_set():
            self.state.update_job_status(
                job_id=job_id, status="cancelled", error_message="Cancelled by user request"
            )
            self._broadcast_log(job_id, "[CANCEL] Disconnected from telemetry stream.")
