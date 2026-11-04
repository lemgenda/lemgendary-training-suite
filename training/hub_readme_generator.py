"""Hub-level README dashboard generator for the LemGendaryModels repository.

Summarises every registered model's training status (SOTA / in progress / not started).
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, NamedTuple

import yaml

from training.telemetry import TelemetryEngine

logger = logging.getLogger("lemtrain.docs.hub_readme_generator")

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)


_HUB_CONTRACT_LINES: tuple[str, ...] = (
    "",
    "---",
    "",
    "## Canonical Directory Topology & Storage Invariants",
    "",
    "Every model repository in `LemGendaryModels/` adheres to a strict,",
    "standardized directory structure:",
    "",
    "```text",
    "LemGendaryModels/<model_key>/",
    "├── README.md               # Model Card & SOTA benchmark targets",
    "├── metrics.csv             # Training and validation loss curves",
    "├── model_metadata.yaml     # Kaggle Models manifest (Public, CC BY-NC 4.0)",
    "├── checkpoints/            # Best and milestone checkpoint vaults (*.pth)",
    "├── notebooks/              # Standardized Kaggle & Colab notebooks",
    "├── dist/                   # Isolated production distributions",
    "│   └── kaggle/",
    "│       ├── pytorch/        # Variation 1: Standalone PyTorch binary (*.pt)",
    "│       ├── onnx_fp32/      # Variation 2: ONNX FP32 (*_FP32.onnx*)",
    "│       └── onnx_fp16/      # Variation 3: Edge ONNX FP16 (*.onnx)",
    "└── visualizations/         # Training curves and diagnostics",
    "```",
    "",
    "## Strict 3-Variation Isolation Contract",
    "",
    "All model distributions published to Kaggle Models Hub must satisfy strict",
    "binary isolation:",
    "",
    "1. **Variation 1 (`pytorch`)**: Exclusively standalone PyTorch weights",
    "   (`<ModelStem>.pt`).",
    "2. **Variation 2 (`onnx_fp32`)**: Exclusively standard precision ONNX model",
    "   (`<ModelStem>_FP32.onnx`) and sidecar tensor data",
    "   (`<ModelStem>_FP32.onnx.data`).",
    "3. **Variation 3 (`onnx_fp16`)**: Exclusively half-precision embedded ONNX",
    "   model (`<ModelStem>.onnx`) with zero external tensor references.",
    "",
    "This contract is strictly validated prior to upload via",
    "`StrictIsolationValidator` to ensure zero cross-contamination.",
    "",
)
"""Static directory-topology and variation-isolation sections of the dashboard."""


@dataclass
class _Progress:
    """Last ``metrics.csv`` row of a model (defaults when the file is absent)."""

    epoch: int = 0
    data: float = 1.0
    res: float = 0.0
    qs: float = 0.0
    phase: int = 1
    fold: int = 1


class _InProgress(NamedTuple):
    """Dashboard row of a model that has trained at least one epoch."""

    name: str
    completeness: float
    epoch: int
    data: float
    res: float
    task_type: str
    phase: int
    fold: int


def _max_resolution(model_info: dict[str, Any]) -> int:
    """Validation resolution, else the top of the resolution ladder, else 256."""
    max_res = model_info.get("val_resolution", 0)
    if not max_res:
        opt = model_info.get("optimization", {})
        ladder = opt.get("res_ladder", [])
        if ladder:
            max_res = max(ladder)
    if not max_res:
        max_res = 256  # Fallback
    return max_res


def _update_from_row(progress: _Progress, last_row: dict[str, Any], task_type: str) -> None:
    """Copy the progress columns of ``last_row`` (in column order; may raise ``ValueError``)."""
    epoch_val = last_row.get("Epoch")
    progress.epoch = int(epoch_val) if epoch_val is not None else 0
    data_val = last_row.get("Data")
    progress.data = float(data_val) if data_val is not None else 1.0
    res_val = last_row.get("Res")
    progress.res = float(res_val) if res_val is not None else float(progress.res)
    qs_val = last_row.get("Quality_Score")
    progress.qs = float(qs_val) if qs_val is not None else 0.0

    if task_type == "forex":
        phase_val = last_row.get("Phase")
        progress.phase = int(phase_val) if phase_val is not None else 1
        fold_val = last_row.get("Fold")
        progress.fold = int(fold_val) if fold_val is not None else 1


def _read_progress(metrics_csv: str, max_res: int, task_type: str, model_key: str) -> _Progress:
    """Parse the last row of ``metrics_csv``; columns read before a bad cell are kept."""
    progress = _Progress(res=float(max_res))
    if os.path.exists(metrics_csv):
        try:
            with open(metrics_csv, encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
                if reader:
                    _update_from_row(progress, reader[-1], task_type)
        except (OSError, UnicodeDecodeError, csv.Error, ValueError) as e:
            # ValueError: non-numeric Epoch/Data/Res/Quality_Score/Phase/Fold cells.
            logger.warning("Failed parsing metrics %s for %s: %s", metrics_csv, model_key, e)
    return progress


def _completeness(
    progress: _Progress,
    model_info: dict[str, Any],
    max_res: int,
    engine: TelemetryEngine,
) -> tuple[float, bool]:
    """Return ``(completeness percent, ladder passed)`` for a model's progress."""
    if progress.epoch <= 1:
        return 0.0, False
    task_type = model_info.get("dataset_type", "restoration")
    sota_targets = model_info.get("sota_targets", {})
    target_qs = 0.0
    if sota_targets:
        target_qs, _ = engine.compute_quality_score(sota_targets, sota_targets, task_type)

    if target_qs > 0:
        clamped_cqs = max(0.0, progress.qs)
        qs_pct = min(1.0, clamped_cqs / target_qs)
    else:
        qs_pct = min(1.0, progress.epoch / 100.0)  # Fallback heuristic

    if task_type == "forex":
        phase_pct = min(1.0, progress.phase / 4.0)
        fold_pct = min(1.0, progress.fold / 6.0)
        return (phase_pct + fold_pct + qs_pct) / 3.0 * 100.0, progress.fold >= 6
    data_pct = min(1.0, progress.data)
    res_pct = min(1.0, progress.res / max_res) if max_res > 0 else 1.0
    return (data_pct + res_pct + qs_pct) / 3.0 * 100.0, progress.res >= max_res


class _Dashboard:
    """Models grouped by training status."""

    def __init__(self) -> None:
        self.completed: list[str] = []
        self.in_progress: list[_InProgress] = []
        self.not_started: list[str] = []

    def add(
        self, model_key: str, model_info: dict[str, Any], hub_dir: str, engine: TelemetryEngine
    ) -> None:
        """Classify one registry model as SOTA, in progress or not started."""
        model_name = model_info.get("name", model_key)
        task_type = model_info.get("dataset_type", "restoration")
        max_res = _max_resolution(model_info)
        metrics_csv = os.path.join(hub_dir, model_key, "metrics.csv")
        progress = _read_progress(metrics_csv, max_res, task_type, model_key)
        completeness, ladder_passed = _completeness(progress, model_info, max_res, engine)

        # Invariant: Training should NEVER be declared complete if it didn't reach 100% data
        # fraction on max resolution and hasn't trained for at least 20 epochs
        is_sota = (
            progress.epoch >= 20
            and progress.data >= 0.999
            and ladder_passed
            and completeness >= 99.99
        )
        if is_sota:
            self.completed.append(model_name)
        elif progress.epoch >= 1:
            self.in_progress.append(
                _InProgress(
                    model_name,
                    completeness,
                    progress.epoch,
                    progress.data,
                    progress.res,
                    task_type,
                    progress.phase,
                    progress.fold,
                )
            )
        else:
            self.not_started.append(model_name)

    def markdown_lines(self) -> list[str]:
        """Dashboard Markdown, one list item per line."""
        lines = [
            "# LemGendary AI Training Matrix",
            "",
            (
                "Auto-generated live dashboard | Last Updated: "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ),
            "",
            "## Completed (SOTA Targets Achieved)",
            "",
            "Models that have fully satisfied their strict target benchmarks",
            "and exported their architecture for production deployment.",
            "",
        ]
        lines.extend(self._completed_lines())
        lines.extend(
            [
                "",
                "## In Progress (Training)",
                "",
                "Models actively scaling the resolution ladder and optimizing",
                "weights across dataset manifolds.",
                "",
            ]
        )
        lines.extend(self._in_progress_lines())
        lines.extend(
            [
                "",
                "## Not Started",
                "",
                "Registered matrix targets awaiting cluster allocation.",
                "",
            ]
        )
        if self.not_started:
            lines.extend(f"- {m}" for m in sorted(self.not_started))
        else:
            lines.append("- *All models have been deployed.*")
        lines.extend(_HUB_CONTRACT_LINES)
        return lines

    def _completed_lines(self) -> list[str]:
        if not self.completed:
            return ["- *No models have achieved SOTA targets yet.*"]
        out: list[str] = []
        for m in sorted(self.completed):
            out.append(f"- {m}")
            out.append("  ![SOTA](https://img.shields.io/badge/Status-SOTA-brightgreen)")
        return out

    def _in_progress_lines(self) -> list[str]:
        if not self.in_progress:
            return ["- *No models currently in active training.*"]
        lines = [
            "| Model | Completeness | Epochs | Data / Phase | Res / Fold |",
            "| --- | --- | --- | --- | --- |",
        ]
        # Sort descending by completeness
        self.in_progress.sort(key=lambda x: x[1], reverse=True)
        for m, pct, ep, df, res, t_type, ph, fd in self.in_progress:
            disp_m = m.split(" (", 1)[0] if len(m) > 42 and " (" in m else m
            if t_type == "forex":
                lines.append(f"| {disp_m} | **{pct:.1f}%** | {ep} | Phase {ph} | Fold {fd} |")
            else:
                lines.append(
                    f"| {disp_m} | **{pct:.1f}%** | {ep} | {df * 100:.0f}% | {int(res)}px |"
                )
        return lines


def generate_hub_readme(project_root: str = workspace_root) -> None:
    """Write LemGendaryModels/README.md summarising training status of every model."""
    yaml_path = os.path.join(project_root, "unified_models_v2.yaml")
    hub_dir = os.path.join(os.path.dirname(project_root), "LemGendaryModels")
    readme_path = os.path.join(hub_dir, "README.md")

    with open(yaml_path, encoding="utf-8") as f:
        unified_models: dict[str, Any] = yaml.safe_load(f)

    engine = TelemetryEngine(".", ".")
    dashboard = _Dashboard()
    for model_key, model_info in unified_models.items():
        if model_key != "_registry_metadata":
            dashboard.add(model_key, model_info, hub_dir, engine)

    # Save
    os.makedirs(hub_dir, exist_ok=True)
    with open(readme_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(dashboard.markdown_lines()))

    print(f"[HUB SYNC] Dashboard successfully updated at: {readme_path}")


if __name__ == "__main__":
    generate_hub_readme()
