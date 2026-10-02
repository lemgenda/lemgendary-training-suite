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
from pathlib import Path
import re
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
    r"(\d+%\s*[━█─\-=|]+|\b\d+/\d+\b\s+.*\b(it/s|s/it|B/s|KB/s|MB/s|GB/s)\b|\d+:\d+<\d+:\d+|\d+%\s*\|.*\|\s*\d+/\d+)"
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

        payload_dict = {
            "timestamp": timestamp,
            "job_id": job_id,
            "message": message,
            "step_name": "Training",
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
        if self._stdout_tee:
            self._stdout_tee.register_thread(tid, lambda line: self._broadcast_log(job_id, line))
        if self._stderr_tee:
            self._stderr_tee.register_thread(tid, lambda line: self._broadcast_log(job_id, line))

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

    def shutdown(self) -> None:
        """Gracefully terminate thread pool executors."""
        self.executor.shutdown(wait=False, cancel_futures=True)
