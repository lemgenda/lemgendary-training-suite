"""Asynchronous Job Execution Manager and WebSocket Log Broadcaster.

Manages background thread pool, dispatches in-process jobs via training.services,
captures stdout/logging, and broadcasts events to WebSocket clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from training.server.job_kaggle_run import KaggleRunJobsMixin
from training.server.job_kaggle_sync import KaggleSyncJobsMixin
from training.server.job_local import LocalJobsMixin
from training.server.job_output import (
    _ensure_output_tees,
    _is_progress_line,
)
from training.server.state import ServerState
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.server.jobs")


def _is_kaggle_noise(line: str) -> bool:
    """Noisy kagglehub lines hidden from push logs (steps, errors and summaries stay)."""
    low = line.lower()
    return (
        low.startswith(("uploading model https://", "failed uploading "))
        or "does not exist or access is forbidden" in low
        or "creating or handling model" in low
    )


def _log_status(message: str) -> str:
    """GUI telemetry status of a log line: error, success, warning or running."""
    msg_lower = message.lower()
    if "[error]" in msg_lower or "failed" in msg_lower or "traceback" in msg_lower:
        return "error"
    if "[success]" in msg_lower or "complete" in msg_lower:
        return "success"
    if "[progress]" in msg_lower:
        return "running"
    if "[warn" in msg_lower:
        return "warning"
    return "running"


class JobManager(LocalJobsMixin, KaggleRunJobsMixin, KaggleSyncJobsMixin):
    """Coordinates asynchronous execution of training, evaluation, and export tasks."""

    def __init__(self, state: ServerState | None = None, max_workers: int = 2) -> None:
        self.project_root = get_project_root()
        self.state = state or ServerState(project_root=self.project_root)
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="lemtrain-worker"
        )
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
        self.state.insert_job(
            job_id=job_id, job_type=job_type, model_key=model_key, params=job_params
        )

        cancel_event = threading.Event()
        with self._lock:
            self._cancellation_events[job_id] = cancel_event

        self.executor.submit(
            self._run_job_worker, job_id, job_type, model_key, job_params, cancel_event
        )
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
            self.state.update_job_status(
                job_id=job_id, status="cancelled", error_message="Cancelled by user request"
            )
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

    def _append_log_file(self, job_id: str, timestamp: str, message: str) -> None:
        """Append ``[timestamp] message`` to the job's log file (failures are logged)."""
        log_file = self.state.logs_dir / f"{job_id}.log"
        try:
            self.state.logs_dir.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")
        except OSError as e:
            logger.warning("Failed writing to log file %s: %s", log_file, e)

    def _broadcast_log(self, job_id: str, message: str) -> None:
        """Stream log line to file, job subscribers, and global subscribers."""
        timestamp = datetime.now().isoformat()
        is_progress = _is_progress_line(message)

        # Rate-limit progress bar updates (max 10 Hz) to avoid overwhelming the UI/WebSocket
        if is_progress:
            now = time.monotonic()
            last_broadcast = self._last_progress_broadcast.get(job_id, 0.0)
            if now - last_broadcast < 0.1:
                return
            self._last_progress_broadcast[job_id] = now

        # Classify status for compatible GUI telemetry display
        status = _log_status(message)

        # Determine contextual job type and step name for telemetry isolation
        job = self.state.get_job(job_id)
        job_type = job.get("job_type", "train") if job else "train"
        step_name = (
            "Cloud Training"
            if job_type in ("kaggle_train", "kaggle_monitor", "cloud_push")
            else "Training"
        )

        log_payload = json.dumps(
            {
                "timestamp": timestamp,
                "job_id": job_id,
                "job_type": job_type,
                "message": message,
                "step_name": step_name,
                "step_number": 0,
                "status": status,
                "is_progress": is_progress,
            }
        )

        # Append to log file
        self._append_log_file(job_id, timestamp, message)

        # Threadsafe push to asyncio queues
        if self._loop and self._loop.is_running():
            with self._lock:
                target_queues = list(self._job_subscribers.get(job_id, set())) + list(
                    self._global_subscribers
                )
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
            if job_type in ("kaggle_push", "kaggle_push_all") and _is_kaggle_noise(line):
                return
            self._broadcast_log(job_id, line)

        if self._stdout_tee:
            self._stdout_tee.register_thread(tid, _job_log_filter)
        if self._stderr_tee:
            self._stderr_tee.register_thread(tid, _job_log_filter)

        try:
            if cancel_event.is_set():
                self.state.update_job_status(
                    job_id=job_id, status="cancelled", error_message="Cancelled before start"
                )
                self._broadcast_log(job_id, "[CANCEL] Job aborted prior to dispatch.")
                return

            self._dispatch_job(job_id, job_type, model_key, params, cancel_event)

        except (InterruptedError, KeyboardInterrupt):
            logger.info("Job %s was cancelled by user request.", job_id)
            self.state.update_job_status(
                job_id=job_id, status="cancelled", error_message="Cancelled by user request"
            )
            self._broadcast_log(job_id, f"[CANCEL] Job {job_id} halted by cancellation signal.")
        except Exception as exc:
            # Record the failure on the job, then re-raise so it also lands on the executor future.
            logger.exception("Job %s encountered unexpected failure", job_id)
            self.state.update_job_status(job_id=job_id, status="failed", error_message=str(exc))
            self._broadcast_log(job_id, f"[ERROR] Execution failed: {exc}")
            raise
        finally:
            if self._stdout_tee:
                self._stdout_tee.unregister_thread(tid)
            if self._stderr_tee:
                self._stderr_tee.unregister_thread(tid)
            with self._lock:
                self._cancellation_events.pop(job_id, None)

    def _dispatch_job(
        self,
        job_id: str,
        job_type: str,
        model_key: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Run the ``execute_*`` handler for ``job_type``; ValueError when it is unknown."""
        handlers: dict[str, Callable[[], None]] = {
            "train": lambda: self.execute_training(job_id, model_key, params, cancel_event),
            "eval": lambda: self.execute_evaluation(job_id, model_key, params),
            "export": lambda: self.execute_export(job_id, model_key, params),
            "kaggle_train": lambda: self.execute_kaggle_training(
                job_id, model_key, params, cancel_event
            ),
            "kaggle_monitor": lambda: self.execute_kaggle_monitor(
                job_id, model_key, params, cancel_event
            ),
            "test_all": lambda: self.execute_test_all(job_id, params, cancel_event),
            "unit_test_all": lambda: self.execute_test_all(job_id, params, cancel_event),
            "kaggle_push_all": lambda: self.execute_kaggle_push_all(job_id, params, cancel_event),
            "kaggle_push": lambda: self.execute_kaggle_push(
                job_id, model_key, params, cancel_event
            ),
            "kaggle_pull": lambda: self.execute_kaggle_pull(
                job_id, model_key, params, cancel_event
            ),
        }
        handler = handlers.get(job_type)
        if handler is None:
            raise ValueError(f"Unknown job_type: '{job_type}'")
        handler()

    def shutdown(self) -> None:
        """Gracefully terminate thread pool executors."""
        self.executor.shutdown(wait=False, cancel_futures=True)
