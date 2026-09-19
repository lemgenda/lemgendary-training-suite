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
        log_payload = json.dumps({"timestamp": timestamp, "job_id": job_id, "message": message})

        # Append to log file
        log_file = self.state.logs_dir / f"{job_id}.log"
        try:
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

        except Exception as exc:
            logger.exception("Job %s encountered unexpected failure: %s", job_id, exc)
            self.state.update_job_status(job_id=job_id, status="failed", error_message=str(exc))
            self._broadcast_log(job_id, f"[ERROR] Execution failed: {exc}")
        finally:
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
            on_epoch_end=epoch_callback,
        )

        metrics: dict[str, Any] = {
            "final_epoch": summary.final_epoch,
            "total_time": summary.total_time,
            "status": summary.status,
            "best_metrics": summary.best_metrics,
        }

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
