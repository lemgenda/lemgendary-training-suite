"""Asynchronous Job Execution Manager and WebSocket Log Broadcaster.

Manages background thread pool, dispatches in-process jobs via training.services,
captures stdout/logging, and broadcasts events to WebSocket clients.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Callable
import uuid

from training.server.state import ServerState
from training.services.eval_service import EvaluationService
from training.services.export_service import ExportService
from training.services.training_service import TrainingService
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.server.jobs")

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CLEAN_CONTROL_RE = re.compile(r"^\s*\[K\s*")
PROGRESS_RE = re.compile(
    r"(\d+%\s*[━█─\-=|]+|\b\d+/\d+\b\s+.*\b(it/s|s/it|B/s|KB/s|MB/s|GB/s)\b|\d+:\d+<\d+:\d+|\d+%\s*\|.*\|\s*\d+/\d+|\[BATCH\s+\d+/\d+\])"
)


def _clean_console_line(text: str) -> str:
    if not text:
        return ""
    if "\r" in text:
        parts = [p for p in text.split("\r") if p.strip()]
        if parts:
            text = parts[-1]
    text = ANSI_ESCAPE_RE.sub("", text)
    text = CLEAN_CONTROL_RE.sub("", text)
    return text.strip()


def _is_progress_line(text: str) -> bool:
    return bool(PROGRESS_RE.search(text))


class _GlobalOutputTee:
    """Process-wide stream tee that routes thread-specific stdout/stderr lines to active job loggers."""

    def __init__(self, original_stream: Any) -> None:
        self.original_stream = original_stream
        self._handlers: dict[int, Callable[[str], None]] = {}
        self._lock = threading.Lock()
        self._buffers: dict[int, str] = {}

    def register_thread(self, thread_id: int, on_line: Callable[[str], None]) -> None:
        with self._lock:
            self._handlers[thread_id] = on_line
            self._buffers[thread_id] = ""

    def unregister_thread(self, thread_id: int) -> None:
        with self._lock:
            buf = self._buffers.pop(thread_id, "").strip()
            handler = self._handlers.pop(thread_id, None)
            if buf and handler:
                try:
                    handler(buf)
                except Exception:
                    pass

    def write(self, s: str) -> int:
        try:
            self.original_stream.write(s)
            self.original_stream.flush()
        except Exception:
            pass

        tid = threading.get_ident()
        with self._lock:
            handler = self._handlers.get(tid)
            if handler:
                buf = self._buffers.get(tid, "") + s
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    clean = _clean_console_line(line)
                    if clean:
                        try:
                            handler(clean)
                        except Exception:
                            pass
                self._buffers[tid] = buf
        return len(s)

    def flush(self) -> None:
        try:
            self.original_stream.flush()
        except Exception:
            pass
        tid = threading.get_ident()
        with self._lock:
            handler = self._handlers.get(tid)
            if handler:
                buf = self._buffers.get(tid, "")
                clean = _clean_console_line(buf)
                if clean:
                    try:
                        handler(clean)
                    except Exception:
                        pass
                    self._buffers[tid] = ""

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        if hasattr(self.original_stream, "fileno"):
            return self.original_stream.fileno()
        raise OSError("fileno not supported")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original_stream, name)


_stdout_tee: _GlobalOutputTee | None = None
_stderr_tee: _GlobalOutputTee | None = None


def _ensure_output_tees() -> tuple[_GlobalOutputTee, _GlobalOutputTee]:
    global _stdout_tee, _stderr_tee
    if _stdout_tee is None:
        _stdout_tee = _GlobalOutputTee(sys.stdout)
        sys.stdout = _stdout_tee  # type: ignore[assignment]
    if _stderr_tee is None:
        _stderr_tee = _GlobalOutputTee(sys.stderr)
        sys.stderr = _stderr_tee  # type: ignore[assignment]
    return _stdout_tee, _stderr_tee


class JobManager:
    """Coordinates asynchronous execution of training, evaluation, and export tasks."""

    def __init__(self, state: ServerState | None = None, max_workers: int = 2) -> None:
        self.project_root = get_project_root()
        self.state = state or ServerState(project_root=self.project_root)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="lemtrain-worker")
        self._cancellation_events: dict[str, threading.Event] = {}
        self._job_subscribers: dict[str, set[asyncio.Queue[str]]] = {}
        self._global_subscribers: set[asyncio.Queue[str]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._stdout_tee, self._stderr_tee = _ensure_output_tees()
        self._last_progress_broadcast: dict[str, float] = {}

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register server event loop for threadsafe WebSocket broadcasting."""
        self._loop = loop

    def submit_job(
        self,
        job_type: str,
        model_key: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Enqueue task for background execution."""
        clean_uuid = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"job_{timestamp}_{clean_uuid}"

        job_params = params or {}
        self.state.insert_job(job_id=job_id, job_type=job_type, model_key=model_key, params=job_params)

        cancel_event = threading.Event()
        with self._lock:
            self._cancellation_events[job_id] = cancel_event

        self.executor.submit(self._run_job_worker, job_id, job_type, model_key, job_params, cancel_event)
        return job_id

    def cancel_job(self, job_id: str) -> bool:
        """Signal job to abort execution."""
        with self._lock:
            cancel_event = self._cancellation_events.get(job_id)
            if cancel_event:
                cancel_event.set()

        job = self.state.get_job(job_id)
        if not job:
            return False

        if job["status"] in ("pending", "running"):
            self.state.update_job_status(job_id=job_id, status="cancelled", error_message="Cancelled by user request")
            self._broadcast_log(job_id, f"[STATUS] Job {job_id} was cancelled.")
            return True

        return False

    def subscribe_job_logs(self, job_id: str) -> asyncio.Queue[str]:
        """Subscribe to real-time log feed for specific job."""
        q: asyncio.Queue[str] = asyncio.Queue()
        with self._lock:
            if job_id not in self._job_subscribers:
                self._job_subscribers[job_id] = set()
            self._job_subscribers[job_id].add(q)
        return q

    def unsubscribe_job_logs(self, job_id: str, q: asyncio.Queue[str]) -> None:
        """Unsubscribe from job log feed."""
        with self._lock:
            if job_id in self._job_subscribers:
                self._job_subscribers[job_id].discard(q)
                if not self._job_subscribers[job_id]:
                    del self._job_subscribers[job_id]

    def subscribe_global_logs(self) -> asyncio.Queue[str]:
        """Subscribe to global event log feed."""
        q: asyncio.Queue[str] = asyncio.Queue()
        with self._lock:
            self._global_subscribers.add(q)
        return q

    def unsubscribe_global_logs(self, q: asyncio.Queue[str]) -> None:
        """Unsubscribe from global event log feed."""
        with self._lock:
            self._global_subscribers.discard(q)

    def _broadcast_log(self, job_id: str, message: str) -> None:
        """Stream log line to file, job subscribers, and global subscribers."""
        timestamp = datetime.now().isoformat()
        is_progress = _is_progress_line(message)

        # Rate-limit high-frequency progress bar updates (max 10 Hz) to avoid overwhelming UI/WebSocket
        if is_progress:
            now = time.monotonic()
            last_broadcast = self._last_progress_broadcast.get(job_id, 0.0)
            if now - last_broadcast < 0.1:
                return
            self._last_progress_broadcast[job_id] = now

        # Classify status for compatible GUI telemetry display
        status = "running"
        msg_lower = message.lower()
        if "[error]" in msg_lower or "failed" in msg_lower or "traceback" in msg_lower:
            status = "error"
        elif "[success]" in msg_lower or "complete" in msg_lower:
            status = "success"
        elif "[progress]" in msg_lower:
            status = "running"
        elif "[warn" in msg_lower:
            status = "warning"

        # Determine contextual job type and step name for telemetry isolation
        job = self.state.get_job(job_id)
        job_type = job.get("job_type", "train") if job else "train"
        step_name = "Cloud Training" if job_type in ("kaggle_train", "kaggle_monitor", "cloud_push") else "Training"

        payload_dict = {
            "timestamp": timestamp,
            "job_id": job_id,
            "job_type": job_type,
            "message": message,
            "step_name": step_name,
            "step_number": 0,
            "status": status,
            "is_progress": is_progress,
        }
        log_payload = json.dumps(payload_dict)

        # Append to log file
        log_file = self.state.logs_dir / f"{job_id}.log"
        try:
            self.state.logs_dir.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")
        except OSError as e:
            logger.warning("Failed writing to log file %s: %s", log_file, e)

        # Threadsafe push to asyncio queues
        if self._loop and self._loop.is_running():
            with self._lock:
                target_queues = list(self._job_subscribers.get(job_id, set())) + list(self._global_subscribers)
            for q in target_queues:
                self._loop.call_soon_threadsafe(q.put_nowait, log_payload)

    def _run_job_worker(
        self,
        job_id: str,
        job_type: str,
        model_key: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Execute job logic within background thread."""
        self.state.update_job_status(job_id=job_id, status="running")
        self._broadcast_log(job_id, f"[START] Commencing execution of {job_type} on {model_key}...")

        tid = threading.get_ident()
        def _job_log_filter(line: str) -> None:
            if job_type in ("kaggle_push", "kaggle_push_all"):
                # Suppress noisy library lines; keep structured steps, errors, and success summaries
                low = line.lower()
                if (
                    low.startswith("uploading model https://")
                    or "does not exist or access is forbidden" in low
                    or "creating or handling model" in low
                    or low.startswith("failed uploading ")
                ):
                    return
            self._broadcast_log(job_id, line)

        if self._stdout_tee:
            self._stdout_tee.register_thread(tid, _job_log_filter)
        if self._stderr_tee:
            self._stderr_tee.register_thread(tid, _job_log_filter)

        try:
            if cancel_event.is_set():
                self.state.update_job_status(job_id=job_id, status="cancelled", error_message="Cancelled before start")
                self._broadcast_log(job_id, "[CANCEL] Job aborted prior to dispatch.")
                return

            if job_type == "train":
                self._execute_training(job_id, model_key, params, cancel_event)
            elif job_type == "eval":
                self._execute_evaluation(job_id, model_key, params)
            elif job_type == "export":
                self._execute_export(job_id, model_key, params)
            elif job_type == "kaggle_train":
                self._execute_kaggle_training(job_id, model_key, params, cancel_event)
            elif job_type == "kaggle_monitor":
                self._execute_kaggle_monitor(job_id, model_key, params, cancel_event)
            elif job_type in ("test_all", "unit_test_all"):
                self._execute_test_all(job_id, params, cancel_event)
            elif job_type == "kaggle_push_all":
                self._execute_kaggle_push_all(job_id, params, cancel_event)
            elif job_type == "kaggle_push":
                self._execute_kaggle_push(job_id, model_key, params, cancel_event)
            elif job_type == "kaggle_pull":
                self._execute_kaggle_pull(job_id, model_key, params, cancel_event)
            else:
                raise ValueError(f"Unknown job_type: '{job_type}'")

        except (InterruptedError, KeyboardInterrupt):
            logger.info("Job %s was cancelled by user request.", job_id)
            self.state.update_job_status(job_id=job_id, status="cancelled", error_message="Cancelled by user request")
            self._broadcast_log(job_id, f"[CANCEL] Job {job_id} halted by cancellation signal.")
        except Exception as exc:
            logger.exception("Job %s encountered unexpected failure: %s", job_id, exc)
            self.state.update_job_status(job_id=job_id, status="failed", error_message=str(exc))
            self._broadcast_log(job_id, f"[ERROR] Execution failed: {exc}")
        finally:
            if self._stdout_tee:
                self._stdout_tee.unregister_thread(tid)
            if self._stderr_tee:
                self._stderr_tee.unregister_thread(tid)
            with self._lock:
                self._cancellation_events.pop(job_id, None)

    def _execute_training(
        self,
        job_id: str,
        model_key: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Execute training service task."""
        service = TrainingService(project_root=self.project_root)

        def epoch_callback(epoch: int, metrics: dict[str, float]) -> None:
            if cancel_event.is_set():
                raise InterruptedError(f"Job {job_id} cancelled by user.")
            metrics_summary = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            self._broadcast_log(job_id, f"[PROGRESS] Epoch {epoch} completed: {metrics_summary}")

        summary = service.train(
            model_key=model_key,
            epochs=params.get("epochs"),
            batch_size=params.get("batch_size"),
            lr=params.get("learning_rate") or params.get("lr"),
            preset=params.get("preset"),
            env=params.get("env", "local"),
            clean=params.get("clean", False),
            auto_sync=params.get("auto_sync", False),
            parallel=params.get("parallel", "auto"),
            resolution=params.get("resolution"),
            enable_sawtooth=params.get("enable_sawtooth", True),
            on_epoch_end=epoch_callback,
            cancel_check=cancel_event.is_set,
        )

        metrics: dict[str, Any] = {
            "final_epoch": summary.final_epoch,
            "total_time": summary.total_time,
            "status": summary.status,
            "best_metrics": summary.best_metrics,
        }

        if summary.status == "cancelled" or cancel_event.is_set():
            self.state.update_job_status(job_id=job_id, status="cancelled", error_message="Cancelled by user request")
            self._broadcast_log(job_id, f"[CANCEL] Training halted for '{summary.model_name}' by user request.")
            return

        self.state.update_job_status(job_id=job_id, status="completed", metrics=metrics)
        self._broadcast_log(job_id, f"[SUCCESS] Training complete for '{summary.model_name}'. Status: {summary.status}")

    def _execute_evaluation(
        self,
        job_id: str,
        model_key: str,
        params: dict[str, Any],
    ) -> None:
        """Execute evaluation service task."""
        service = EvaluationService(project_root=self.project_root)
        result = service.evaluate(
            model_key=model_key,
            checkpoint_path=params.get("checkpoint_path"),
            batch_size=params.get("batch_size"),
            env=params.get("env", "local"),
        )

        self.state.update_job_status(job_id=job_id, status="completed", metrics=result.get("metrics", {}))
        self._broadcast_log(job_id, f"[SUCCESS] Evaluation complete. Metrics: {result.get('metrics', {})}")

    def _execute_export(
        self,
        job_id: str,
        model_key: str,
        params: dict[str, Any],
    ) -> None:
        """Execute export service task."""
        service = ExportService(project_root=self.project_root)
        exported = service.export(
            model_key=model_key,
            checkpoint_path=params.get("checkpoint_path"),
            output_dir=params.get("output_dir"),
            targets=params.get("targets"),
        )

        metrics: dict[str, Any] = {"exported_artifacts": exported}
        self.state.update_job_status(job_id=job_id, status="completed", metrics=metrics)
        self._broadcast_log(job_id, f"[SUCCESS] Export complete: {list(exported.keys())}")

    def _execute_kaggle_training(
        self,
        job_id: str,
        model_key: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Launch training on Kaggle Cloud, stream telemetry, and auto-pull checkpoints."""
        import subprocess
        from training.cloud.credentials import resolve_kaggle_credentials
        from training.cloud.kaggle_hub import KaggleHubManager
        from training.kaggle_cloud_manager import pull_kaggle_artifacts
        from training.kaggle_monitor import authenticate_kaggle_user, parse_kernel_logs

        username_override = params.get("username")
        key_override = params.get("key")
        gpu = params.get("gpu", "T4")
        auto_pull = params.get("auto_pull", True)
        poll_interval = params.get("poll_interval", 5)

        user, key = resolve_kaggle_credentials(override_user=username_override, override_key=key_override)
        self._broadcast_log(job_id, f"[KAGGLE] Authenticating Kaggle API for user: '{user}' (Accelerator: {gpu})...")

        api = authenticate_kaggle_user(user, key) if (user and key) else None
        if not api:
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
            except Exception as e:
                raise RuntimeError(f"Kaggle authentication failed: {e}. Please configure Kaggle username and API key.")

        mgr = KaggleHubManager(username=user, key=key)
        kernel_slug = mgr.get_kernel_slug(model_key)
        self._broadcast_log(job_id, f"[KAGGLE] Packaging standalone kernel bundle for {model_key}...")
        kernel_dir = mgr.create_cloud_kernel_bundle(model_key, gpu=gpu)
        self._broadcast_log(job_id, f"[KAGGLE] Deploying kernel bundle {kernel_slug} to Kaggle Cloud...")

        try:
            api.kernels_push(str(kernel_dir))
            self._broadcast_log(job_id, f"[KAGGLE] Kernel successfully pushed: {kernel_slug}")
        except Exception as sdk_err:
            self._broadcast_log(job_id, f"[WARN] Python SDK push notice: {sdk_err}. Trying CLI fallback...")
            cflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if sys.platform == "win32" else 0
            res = subprocess.run(
                ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
                capture_output=True,
                text=True,
                timeout=60,
                creationflags=cflags,
            )
            if res.returncode == 0:
                self._broadcast_log(job_id, f"[KAGGLE] CLI push succeeded: {res.stdout.strip()}")
            else:
                raise RuntimeError(f"CLI push failed: {res.stderr.strip()}")

        self._broadcast_log(job_id, f"[KAGGLE] Telemetry connected. Monitoring {kernel_slug} (Auto-pull: {auto_pull})...")

        printed_lines_count = 0
        last_status = "QUEUED"

        while not cancel_event.is_set():
            try:
                st = api.kernels_status(kernel_slug)
                status_val = str(getattr(st, "status", st)).rsplit(".", maxsplit=1)[-1].upper()
                if status_val != last_status:
                    self._broadcast_log(job_id, f"[{time.strftime('%H:%M:%S')}] [KAGGLE STATUS] Cloud status: {status_val}")
                    last_status = status_val
            except Exception:
                status_val = last_status

            try:
                raw_logs = api.kernels_logs(kernel_slug)
                log_text = parse_kernel_logs(raw_logs)
                all_lines = log_text.splitlines()
                if len(all_lines) > printed_lines_count:
                    new_lines = all_lines[printed_lines_count:]
                    for line in new_lines:
                        self._broadcast_log(job_id, line)
                        if auto_pull and model_key:
                            epoch_signal = re.search(r"epoch\s+\d+\s+complete|saving.*checkpoint|model version.*committed|\[sota\]", line, re.IGNORECASE)
                            if epoch_signal:
                                self._broadcast_log(job_id, f"[AUTO-PULL] Epoch milestone detected in logs. Syncing artifacts for {model_key}...")
                                try:
                                    pull_kaggle_artifacts(model_key, username=user)
                                except Exception as pull_err:
                                    self._broadcast_log(job_id, f"[WARN] Auto-pull notice: {pull_err}")
                    printed_lines_count = len(all_lines)
            except Exception:
                pass

            if status_val in ["COMPLETE", "ERROR", "CANCELACK"]:
                if status_val == "COMPLETE":
                    if auto_pull:
                        self._broadcast_log(job_id, f"[SYNC] Cloud execution complete. Pulling final artifacts for {model_key}...")
                        try:
                            pull_kaggle_artifacts(model_key, username=user)
                            self._broadcast_log(job_id, f"[SUCCESS] Final artifacts saved to LemGendaryModels/{model_key}!")
                        except Exception as pe:
                            self._broadcast_log(job_id, f"[WARN] Artifact pull notice: {pe}")
                    self.state.update_job_status(job_id=job_id, status="completed", metrics={"kaggle_status": "COMPLETE", "kernel_slug": kernel_slug})
                    self._broadcast_log(job_id, f"[SUCCESS] Kaggle training completed successfully for {model_key}!")
                else:
                    self.state.update_job_status(job_id=job_id, status="failed", error_message=f"Kaggle kernel halted with status: {status_val}")
                    self._broadcast_log(job_id, f"[ERROR] Kaggle cloud job ended with status: {status_val}")
                return

            time.sleep(poll_interval)

        if cancel_event.is_set():
            self.state.update_job_status(job_id=job_id, status="cancelled", error_message="Cancelled by user request")
            self._broadcast_log(job_id, f"[CANCEL] Telemetry stream detached for {kernel_slug}. Remote kernel may continue running on Kaggle.")

    def _execute_kaggle_monitor(
        self,
        job_id: str,
        model_key: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Stream live logs from an existing or active Kaggle kernel without pushing."""
        from training.cloud.credentials import resolve_kaggle_credentials
        from training.cloud.kaggle_hub import KaggleHubManager
        from training.kaggle_cloud_manager import pull_kaggle_artifacts
        from training.kaggle_monitor import authenticate_kaggle_user, parse_kernel_logs

        username_override = params.get("username")
        key_override = params.get("key")
        kernel_slug = params.get("kernel_slug")
        auto_pull = params.get("auto_pull", False)
        poll_interval = params.get("poll_interval", 5)

        user, key = resolve_kaggle_credentials(override_user=username_override, override_key=key_override)
        if not kernel_slug:
            mgr = KaggleHubManager(username=user)
            kernel_slug = mgr.get_kernel_slug(model_key)

        self._broadcast_log(job_id, f"[KAGGLE] Connecting to telemetry stream for '{kernel_slug}'...")
        api = authenticate_kaggle_user(user, key) if (user and key) else None
        if not api:
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
            except Exception as e:
                raise RuntimeError(f"Kaggle authentication failed: {e}")

        printed_lines_count = 0
        last_status = "UNKNOWN"

        while not cancel_event.is_set():
            try:
                st = api.kernels_status(kernel_slug)
                status_val = str(getattr(st, "status", st)).rsplit(".", maxsplit=1)[-1].upper()
                if status_val != last_status:
                    self._broadcast_log(job_id, f"[{time.strftime('%H:%M:%S')}] [KAGGLE STATUS] Cloud status: {status_val}")
                    last_status = status_val
            except Exception:
                status_val = last_status

            try:
                raw_logs = api.kernels_logs(kernel_slug)
                log_text = parse_kernel_logs(raw_logs)
                all_lines = log_text.splitlines()
                if len(all_lines) > printed_lines_count:
                    new_lines = all_lines[printed_lines_count:]
                    for line in new_lines:
                        self._broadcast_log(job_id, line)
                        if auto_pull and model_key:
                            epoch_signal = re.search(r"epoch\s+\d+\s+complete|saving.*checkpoint|model version.*committed|\[sota\]", line, re.IGNORECASE)
                            if epoch_signal:
                                self._broadcast_log(job_id, f"[AUTO-PULL] Checkpoint milestone detected. Syncing {model_key}...")
                                try:
                                    pull_kaggle_artifacts(model_key, username=user)
                                except Exception as pull_err:
                                    self._broadcast_log(job_id, f"[WARN] Auto-pull notice: {pull_err}")
                    printed_lines_count = len(all_lines)
            except Exception:
                pass

            if status_val in ["COMPLETE", "ERROR", "CANCELACK"]:
                if status_val == "COMPLETE" and auto_pull and model_key:
                    self._broadcast_log(job_id, f"[SYNC] Cloud run completed. Pulling final artifacts for {model_key}...")
                    try:
                        pull_kaggle_artifacts(model_key, username=user)
                        self._broadcast_log(job_id, f"[SUCCESS] Final artifacts saved to LemGendaryModels/{model_key}!")
                    except Exception as pe:
                        self._broadcast_log(job_id, f"[WARN] Pull notice: {pe}")

                self.state.update_job_status(job_id=job_id, status="completed" if status_val == "COMPLETE" else "failed", metrics={"kaggle_status": status_val, "kernel_slug": kernel_slug})
                self._broadcast_log(job_id, f"[TERMINAL] Job ended with status: {status_val}")
                return

            time.sleep(poll_interval)

        if cancel_event.is_set():
            self.state.update_job_status(job_id=job_id, status="cancelled", error_message="Cancelled by user request")
            self._broadcast_log(job_id, f"[CANCEL] Disconnected from telemetry stream.")

    def _execute_test_all(
        self,
        job_id: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Execute fleet diagnostic pass across all models using tools/train_all.py."""
        import subprocess
        import sys

        train_all_script = self.project_root / "tools" / "train_all.py"
        epochs = str(params.get("epochs", 1))
        env = str(params.get("env", "local"))
        cmd = [sys.executable, str(train_all_script), "--epochs", epochs, "--yes", "--env", env]
        if params.get("force", False):
            cmd.append("--force")

        self._broadcast_log(job_id, f"[COMMAND] Launching fleet diagnostic: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            if proc.stdout:
                for line in proc.stdout:
                    if cancel_event.is_set():
                        if os.name == "nt":
                            cflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                            subprocess.run(
                                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                capture_output=True,
                                check=False,
                                creationflags=cflags,
                            )
                        else:
                            proc.terminate()
                        break
                    clean_line = _clean_console_line(line)
                    if clean_line:
                        self._broadcast_log(job_id, clean_line)
            proc.wait()
        finally:
            if cancel_event.is_set():
                try:
                    if os.name == "nt":
                        cflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            capture_output=True,
                            check=False,
                            creationflags=cflags,
                        )
                    else:
                        proc.kill()
                except Exception:
                    pass

        if cancel_event.is_set():
            self.state.update_job_status(job_id=job_id, status="cancelled", error_message="Cancelled by user request")
            self._broadcast_log(job_id, "[CANCEL] Fleet diagnostic pass halted by cancellation.")
            return

        if proc.returncode == 0:
            self.state.update_job_status(
                job_id=job_id,
                status="completed",
                metrics={"returncode": 0, "epochs": int(epochs)},
            )
            self._broadcast_log(job_id, "[SUCCESS] Fleet diagnostic pass completed successfully.")
        else:
            self.state.update_job_status(
                job_id=job_id,
                status="failed",
                error_message=f"Exit code: {proc.returncode}",
            )
            self._broadcast_log(job_id, f"[ERROR] Fleet diagnostic pass failed with return code {proc.returncode}.")

    def _push_model_with_telemetry(
        self,
        job_id: str,
        sync_mgr: Any,
        model_key: str,
        variation: str,
        version_notes: str | None,
        dry_run: bool,
        cancel_event: threading.Event,
        prefix: str = "",
    ) -> Any:
        """Stage, then push each variation one by one, emitting progress lines and summary logs."""
        from training.cloud.variations import VARIATION_BY_KEY, VARIATIONS
        from training.export.staging import VariationStagingService

        sync_mgr.ensure_model_metadata(model_key)
        staging_res = VariationStagingService().stage_variations(model_key)
        if staging_res.staged_files:
            staged_names = [f.name for f in staging_res.staged_files.values()]
            self._broadcast_log(job_id, f"{prefix}[STAGING] Staged {len(staged_names)} files for '{model_key}': {staged_names}")

        variations = list(VARIATION_BY_KEY.keys()) if variation == "all" else [variation]
        from training.cloud.kaggle_model_sync import ModelPushResult
        result = ModelPushResult(model_key=model_key, success=True, isolation_report=sync_mgr.validate_isolation(model_key))
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
                f"{prefix}[UPLOAD] [Step {i}/{total}] PUSHING {model_key} [{var}] -> {handle} ({pct_start}%)",
            )
            var_res = sync_mgr.push_variation(model_key=model_key, variation=var, version_notes=version_notes, dry_run=dry_run)
            result.variations[var] = var_res
            pct_done = int((i / total) * 100)
            if var_res.success:
                self._broadcast_log(
                    job_id,
                    f"{prefix}[UPLOAD] [Step {i}/{total}] DONE {model_key} [{var}] files={var_res.files} ({pct_done}%)",
                )
            else:
                result.success = False
                result.errors.append(f"[{var}] {var_res.error}")
                self._broadcast_log(job_id, f"{prefix}[ERROR] {model_key} [{var}]: {var_res.error}")

        # 2. Sync metadata via API if variations succeeded
        if result.success or dry_run:
            try:
                sync_meta_res = sync_mgr.sync_kaggle_metadata(model_key, dry_run=dry_run)
                result.metadata_sync_result = sync_meta_res
                if sync_meta_res.get("errors"):
                    result.errors.extend(sync_meta_res["errors"])
            except Exception as meta_exc:
                err_str = f"Metadata synchronization error: {meta_exc}"
                result.errors.append(err_str)
                self._broadcast_log(job_id, f"{prefix}[ERROR] {model_key} metadata sync: {err_str}")

        # 3. Emit exact 2 clean summary log entries on success
        if result.success and (not result.metadata_sync_result or not result.metadata_sync_result.get("errors")):
            self._broadcast_log(job_id, f"{prefix}[SUCCESS] Model {model_key} pushed successfully to Kaggle ({total} variations)")
            self._broadcast_log(job_id, f"{prefix}[SUCCESS] Kaggle metadata was successfully updated for {model_key}")
        elif result.errors:
            self._broadcast_log(job_id, f"{prefix}[ERROR] Kaggle publish failed for {model_key}: {'; '.join(result.errors)}")

        return result

    def _execute_kaggle_push(
        self,
        job_id: str,
        model_key: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Push all 4 variations of a single model to Kaggle Models Hub with live telemetry."""
        from training.cloud.kaggle_model_sync import KaggleModelSyncManager

        variation = params.get("variation", "all")
        dry_run = params.get("dry_run", False)
        sync_mgr = KaggleModelSyncManager(username=params.get("username"))
        res = self._push_model_with_telemetry(
            job_id, sync_mgr, model_key, variation, params.get("version_notes"), dry_run, cancel_event
        )
        if res.success and (not res.metadata_sync_result or not res.metadata_sync_result.get("errors")):
            self.state.update_job_status(job_id=job_id, status="completed", metrics=res.to_dict())
        else:
            err = "; ".join(res.errors) if res.errors else "Kaggle push failed"
            self.state.update_job_status(job_id=job_id, status="failed", error_message=err, metrics=res.to_dict())

    def _execute_kaggle_push_all(
        self,
        job_id: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Execute fleet synchronization pushing all registered models to Kaggle Models Hub."""
        from training.cloud.kaggle_model_sync import KaggleModelSyncManager
        from training.registry import get_model_registry

        variation = params.get("variation", "all")
        version_notes = params.get("version_notes")
        dry_run = params.get("dry_run", False)

        reg = get_model_registry()
        models = reg.list_models()
        total_models = len(models)
        self._broadcast_log(
            job_id,
            f"[KAGGLE-PUSH-ALL] Commencing fleet push for {total_models} models (Variation: '{variation}', DryRun: {dry_run})...",
        )

        sync_mgr = KaggleModelSyncManager()

        results: dict[str, Any] = {}
        successful_count = 0
        failed_count = 0

        for idx, model_spec in enumerate(models, 1):
            if cancel_event.is_set():
                self.state.update_job_status(job_id=job_id, status="cancelled", error_message="Cancelled by user request")
                self._broadcast_log(job_id, f"[CANCEL] Fleet push cancelled before processing '{model_spec.key}'.")
                return

            model_key = model_spec.key
            fleet_pct = int(((idx - 1) / total_models) * 100)
            self._broadcast_log(
                job_id,
                f"[UPLOAD] [Step {idx}/{total_models}] FLEET {model_key} ({model_spec.name}) ({fleet_pct}%)",
            )

            try:
                push_res = self._push_model_with_telemetry(
                    job_id, sync_mgr, model_key, variation, version_notes, dry_run, cancel_event, prefix="  "
                )

                if push_res.success and (not push_res.metadata_sync_result or not push_res.metadata_sync_result.get("errors")):
                    successful_count += 1
                else:
                    failed_count += 1

                results[model_key] = push_res.to_dict()

            except Exception as exc:
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
            f"[COMPLETED] Kaggle fleet push finished: {successful_count} succeeded, {failed_count} skipped/failed out of {total_models} models (100%).",
        )

    def _execute_kaggle_pull(
        self,
        job_id: str,
        model_key: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Download all 4 model variations from Kaggle Models Hub with live telemetry."""
        from training.cloud.kaggle_model_sync import KaggleModelSyncManager
        from training.cloud.variations import VARIATION_BY_KEY

        variation = params.get("variation", "all")
        force_download = params.get("force_download", False)

        sync_mgr = KaggleModelSyncManager(username=params.get("username"))
        vars_to_pull = list(VARIATION_BY_KEY.keys()) if variation == "all" else [variation]
        total = len(vars_to_pull)

        self._broadcast_log(job_id, f"[DOWNLOAD] Pulling {total} variation(s) for '{model_key}' from Kaggle Models Hub...")

        success_count = 0
        errors: list[str] = []

        for i, var_key in enumerate(vars_to_pull, 1):
            if cancel_event.is_set():
                raise InterruptedError("Pull cancelled")

            pct_start = int(((i - 1) / total) * 100)
            handle = sync_mgr.get_handle(model_key, var_key)
            self._broadcast_log(
                job_id,
                f"[DOWNLOAD] [Step {i}/{total}] PULLING {model_key} [{var_key}] <- {handle} ({pct_start}%)",
            )

            try:
                ok = sync_mgr.pull_variation(model_key, var_key, force_download=force_download)
                pct_done = int((i / total) * 100)
                if ok:
                    success_count += 1
                    target_dir = sync_mgr.get_variation_dir(model_key, var_key)
                    self._broadcast_log(
                        job_id,
                        f"[DOWNLOAD] [Step {i}/{total}] DONE {model_key} [{var_key}] -> {target_dir} ({pct_done}%)",
                    )
                else:
                    errors.append(f"[{var_key}] Pull returned False")
                    self._broadcast_log(job_id, f"[ERROR] {model_key} [{var_key}]: pull returned False")
            except Exception as exc:
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
                f"[ERROR] Pull incomplete for {model_key}: {success_count}/{total} succeeded. {err_summary}",
            )

    def shutdown(self) -> None:
        """Gracefully terminate thread pool executors."""
        self.executor.shutdown(wait=False, cancel_futures=True)
