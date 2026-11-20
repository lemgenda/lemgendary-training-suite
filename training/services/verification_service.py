"""Model Verification Service for LemGendary Model Training Suite.

Orchestrates verification contracts across models and emits machine-readable reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from training.utils.paths import get_project_root
from training.verification.contracts import ModelVerificationReport
from training.verification.engine import ModelVerificationEngine


class VerificationService:
    """Service executing model family verification pipelines."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or get_project_root()
        self.engine = ModelVerificationEngine(self.project_root)

    def verify_model(
        self,
        model_key: str,
        checkpoint_path: Path | str | None = None,
        output_report_path: Path | str | None = None,
        model: nn.Module | None = None,
        dummy_input: torch.Tensor | None = None,
        metrics: dict[str, float] | None = None,
    ) -> ModelVerificationReport:
        """Run complete verification contract on model and optionally emit JSON report."""
        report = self.engine.run_full_verification(
            model_key=model_key,
            checkpoint_path=checkpoint_path,
            model=model,
            dummy_input=dummy_input,
            metrics=metrics,
        )

        if output_report_path is not None:
            out_path = Path(output_report_path).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)

        return report
