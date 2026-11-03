"""Service-layer calls shared by the ``lemtrain`` model and LLM commands."""

from __future__ import annotations

from typing import Any

import typer

from training.cli.server_commands import is_sidecar_online, sidecar_port
from training.services.export_service import ExportService
from training.services.training_service import TrainingService, TrainRequest
from training.training.summary import TrainingSummary


def announce_sidecar() -> None:
    """Tell the operator when the sidecar daemon is already listening."""
    if is_sidecar_online():
        typer.echo(f"[INFO] Detected active LemGendary Sidecar Daemon on port {sidecar_port()}.")


def run_training(model_key: str, **options: Any) -> TrainingSummary:
    """Train ``model_key`` in-process through ``TrainingService``."""
    return TrainingService().train(TrainRequest(model_key=model_key, **options))


def export_targets(
    model_key: str, checkpoint: str | None, output_dir: str | None, targets: list[str] | None
) -> dict[str, str]:
    """Export ``model_key`` to ``targets`` in-process through ``ExportService``."""
    return ExportService().export(
        model_key=model_key,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        targets=targets,
    )
