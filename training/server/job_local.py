"""In-process training, evaluation, export and test-all job executors."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from typing import Any

from training.server.job_output import (
    _clean_console_line,
)
from training.services.eval_service import EvaluationService
from training.services.export_service import ExportService
from training.services.training_service import TrainingService

logger = logging.getLogger("lemtrain.server.jobs")


class LocalJobsMixin:
    """In-process training, evaluation, export and test-all job executors."""

    def execute_training(
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
            self.state.update_job_status(
                job_id=job_id, status="cancelled", error_message="Cancelled by user request"
            )
            self._broadcast_log(
                job_id, f"[CANCEL] Training halted for '{summary.model_name}' by user request."
            )
            return

        self.state.update_job_status(job_id=job_id, status="completed", metrics=metrics)
        if summary.status == "completed":
            self._broadcast_log(
                job_id,
                f"[SUCCESS] Training fully complete for '{summary.model_name}' "
                f"(Status: {summary.status}, Epochs: {summary.final_epoch}).",
            )
        else:
            self._broadcast_log(
                job_id,
                f"[INFO] Training run finished for '{summary.model_name}' "
                f"(Status: {summary.status}, Epochs: {summary.final_epoch}). "
                "Model requires further epochs/curriculum to reach full completion.",
            )

    def execute_evaluation(
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

        self.state.update_job_status(
            job_id=job_id, status="completed", metrics=result.get("metrics", {})
        )
        self._broadcast_log(
            job_id, f"[SUCCESS] Evaluation complete. Metrics: {result.get('metrics', {})}"
        )

    def execute_export(
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

    def execute_test_all(
        self,
        job_id: str,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Execute fleet diagnostic pass across all models using tools/train_all.py."""
        train_all_script = self.project_root / "tools" / "train_all.py"
        epochs = str(params.get("epochs", 1))
        env = str(params.get("env", "local"))
        cmd = [sys.executable, str(train_all_script), "--epochs", epochs, "--yes", "--env", env]
        if params.get("force", False):
            cmd.append("--force")

        self._broadcast_log(job_id, f"[COMMAND] Launching fleet diagnostic: {' '.join(cmd)}")

        with subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc:
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
                    except (OSError, subprocess.SubprocessError) as exc:
                        logger.warning(
                            "Failed to terminate test process %s after cancellation: %s",
                            proc.pid,
                            exc,
                        )

        if cancel_event.is_set():
            self.state.update_job_status(
                job_id=job_id, status="cancelled", error_message="Cancelled by user request"
            )
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
            self._broadcast_log(
                job_id, f"[ERROR] Fleet diagnostic pass failed with return code {proc.returncode}."
            )
