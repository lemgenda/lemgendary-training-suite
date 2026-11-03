"""Kaggle kernel training and monitoring job executors."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi

from training.cloud.credentials import resolve_kaggle_credentials
from training.cloud.errors import KAGGLE_API_ERRORS
from training.cloud.kaggle_hub import KaggleHubManager
from training.kaggle_accounts import authenticate_kaggle_user
from training.kaggle_cloud_manager import pull_kaggle_artifacts
from training.kaggle_stream import fetch_new_log_lines, is_epoch_milestone

logger = logging.getLogger("lemtrain.server.jobs")


# pull_kaggle_artifacts reports most failures itself; RuntimeError covers a missing kagglehub SDK.
_PULL_ERRORS: tuple[type[Exception], ...] = (RuntimeError, *KAGGLE_API_ERRORS)


class KaggleRunJobsMixin:
    """Kaggle kernel training and monitoring job executors."""

    def execute_kaggle_training(
        self,
        job_id: str,
        model_key: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Launch training on Kaggle Cloud, stream telemetry, and auto-pull checkpoints."""
        username_override = params.get("username")
        key_override = params.get("key")
        gpu = params.get("gpu", "T4")
        auto_pull = params.get("auto_pull", True)
        poll_interval = params.get("poll_interval", 5)

        user, key = resolve_kaggle_credentials(
            override_user=username_override, override_key=key_override
        )
        self._broadcast_log(
            job_id, f"[KAGGLE] Authenticating Kaggle API for user: '{user}' (Accelerator: {gpu})..."
        )

        api = authenticate_kaggle_user(user, key) if (user and key) else None
        if not api:
            try:
                api = KaggleApi()
                api.authenticate()
            except KAGGLE_API_ERRORS as e:
                raise RuntimeError(
                    f"Kaggle authentication failed: {e}. Please configure Kaggle username and "
                    "API key."
                ) from e

        mgr = KaggleHubManager(username=user, key=key)
        kernel_slug = mgr.get_kernel_slug(model_key)
        self._broadcast_log(
            job_id, f"[KAGGLE] Packaging standalone kernel bundle for {model_key}..."
        )
        kernel_dir = mgr.create_cloud_kernel_bundle(model_key, gpu=gpu)
        self._broadcast_log(
            job_id, f"[KAGGLE] Deploying kernel bundle {kernel_slug} to Kaggle Cloud..."
        )

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

        self._broadcast_log(
            job_id,
            f"[KAGGLE] Telemetry connected. Monitoring {kernel_slug} (Auto-pull: {auto_pull})...",
        )

        printed_lines_count = 0
        last_status = "QUEUED"

        while not cancel_event.is_set():
            try:
                st = api.kernels_status(kernel_slug)
                status_val = str(getattr(st, "status", st)).rsplit(".", maxsplit=1)[-1].upper()
                if status_val != last_status:
                    self._broadcast_log(
                        job_id,
                        f"[{time.strftime('%H:%M:%S')}] [KAGGLE STATUS] Cloud status: {status_val}",
                    )
                    last_status = status_val
            except KAGGLE_API_ERRORS as exc:
                logger.debug(
                    "Kaggle status poll failed for %s; keeping %s: %s",
                    kernel_slug,
                    last_status,
                    exc,
                )
                status_val = last_status

            try:
                new_lines, printed_lines_count = fetch_new_log_lines(
                    api, kernel_slug, printed_lines_count
                )
                for line in new_lines:
                    self._broadcast_log(job_id, line)
                    if auto_pull and model_key and is_epoch_milestone(line):
                        self._broadcast_log(
                            job_id,
                            f"[AUTO-PULL] Epoch milestone detected in logs. "
                            f"Syncing artifacts for {model_key}...",
                        )
                        try:
                            pull_kaggle_artifacts(model_key, username=user)
                        except _PULL_ERRORS as pull_err:
                            self._broadcast_log(job_id, f"[WARN] Auto-pull notice: {pull_err}")
            except KAGGLE_API_ERRORS as exc:
                logger.debug(
                    "Kaggle log poll failed for %s; retrying next interval: %s", kernel_slug, exc
                )

            if status_val in ["COMPLETE", "ERROR", "CANCELACK"]:
                if status_val == "COMPLETE":
                    if auto_pull:
                        self._broadcast_log(
                            job_id,
                            (
                                "[SYNC] Cloud execution complete. Pulling final artifacts for "
                                f"{model_key}..."
                            ),
                        )
                        try:
                            pull_kaggle_artifacts(model_key, username=user)
                            self._broadcast_log(
                                job_id,
                                f"[SUCCESS] Final artifacts saved to LemGendaryModels/{model_key}!",
                            )
                        except _PULL_ERRORS as pe:
                            self._broadcast_log(job_id, f"[WARN] Artifact pull notice: {pe}")
                    self.state.update_job_status(
                        job_id=job_id,
                        status="completed",
                        metrics={"kaggle_status": "COMPLETE", "kernel_slug": kernel_slug},
                    )
                    self._broadcast_log(
                        job_id, f"[SUCCESS] Kaggle training completed successfully for {model_key}!"
                    )
                else:
                    self.state.update_job_status(
                        job_id=job_id,
                        status="failed",
                        error_message=f"Kaggle kernel halted with status: {status_val}",
                    )
                    self._broadcast_log(
                        job_id, f"[ERROR] Kaggle cloud job ended with status: {status_val}"
                    )
                return

            time.sleep(poll_interval)

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

    def execute_kaggle_monitor(
        self,
        job_id: str,
        model_key: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Stream live logs from an existing or active Kaggle kernel without pushing."""
        username_override = params.get("username")
        key_override = params.get("key")
        kernel_slug = params.get("kernel_slug")
        auto_pull = params.get("auto_pull", False)
        poll_interval = params.get("poll_interval", 5)

        user, key = resolve_kaggle_credentials(
            override_user=username_override, override_key=key_override
        )
        if not kernel_slug:
            mgr = KaggleHubManager(username=user)
            kernel_slug = mgr.get_kernel_slug(model_key)

        self._broadcast_log(
            job_id, f"[KAGGLE] Connecting to telemetry stream for '{kernel_slug}'..."
        )
        api = authenticate_kaggle_user(user, key) if (user and key) else None
        if not api:
            try:
                api = KaggleApi()
                api.authenticate()
            except KAGGLE_API_ERRORS as e:
                raise RuntimeError(f"Kaggle authentication failed: {e}") from e

        printed_lines_count = 0
        last_status = "UNKNOWN"

        while not cancel_event.is_set():
            try:
                st = api.kernels_status(kernel_slug)
                status_val = str(getattr(st, "status", st)).rsplit(".", maxsplit=1)[-1].upper()
                if status_val != last_status:
                    self._broadcast_log(
                        job_id,
                        f"[{time.strftime('%H:%M:%S')}] [KAGGLE STATUS] Cloud status: {status_val}",
                    )
                    last_status = status_val
            except KAGGLE_API_ERRORS as exc:
                logger.debug(
                    "Kaggle status poll failed for %s; keeping %s: %s",
                    kernel_slug,
                    last_status,
                    exc,
                )
                status_val = last_status

            try:
                new_lines, printed_lines_count = fetch_new_log_lines(
                    api, kernel_slug, printed_lines_count
                )
                for line in new_lines:
                    self._broadcast_log(job_id, line)
                    if auto_pull and model_key and is_epoch_milestone(line):
                        self._broadcast_log(
                            job_id,
                            f"[AUTO-PULL] Checkpoint milestone detected. Syncing {model_key}...",
                        )
                        try:
                            pull_kaggle_artifacts(model_key, username=user)
                        except _PULL_ERRORS as pull_err:
                            self._broadcast_log(job_id, f"[WARN] Auto-pull notice: {pull_err}")
            except KAGGLE_API_ERRORS as exc:
                logger.debug(
                    "Kaggle log poll failed for %s; retrying next interval: %s", kernel_slug, exc
                )

            if status_val in ["COMPLETE", "ERROR", "CANCELACK"]:
                if status_val == "COMPLETE" and auto_pull and model_key:
                    self._broadcast_log(
                        job_id,
                        f"[SYNC] Cloud run completed. Pulling final artifacts for {model_key}...",
                    )
                    try:
                        pull_kaggle_artifacts(model_key, username=user)
                        self._broadcast_log(
                            job_id,
                            f"[SUCCESS] Final artifacts saved to LemGendaryModels/{model_key}!",
                        )
                    except _PULL_ERRORS as pe:
                        self._broadcast_log(job_id, f"[WARN] Pull notice: {pe}")

                self.state.update_job_status(
                    job_id=job_id,
                    status="completed" if status_val == "COMPLETE" else "failed",
                    metrics={"kaggle_status": status_val, "kernel_slug": kernel_slug},
                )
                self._broadcast_log(job_id, f"[TERMINAL] Job ended with status: {status_val}")
                return

            time.sleep(poll_interval)

        if cancel_event.is_set():
            self.state.update_job_status(
                job_id=job_id, status="cancelled", error_message="Cancelled by user request"
            )
            self._broadcast_log(job_id, "[CANCEL] Disconnected from telemetry stream.")
