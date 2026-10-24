from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any

import yaml

from training.telemetry import TelemetryEngine

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)


def generate_hub_readme(project_root: str = workspace_root) -> None:
    yaml_path = os.path.join(project_root, "unified_models_v2.yaml")
    hub_dir = os.path.join(os.path.dirname(project_root), "LemGendaryModels")
    readme_path = os.path.join(hub_dir, "README.md")

    with open(yaml_path, encoding="utf-8") as f:
        unified_models: dict[str, Any] = yaml.safe_load(f)

    engine = TelemetryEngine(".", ".")

    completed: list[str] = []
    in_progress: list[tuple[str, float, int, float, float, str, int, int]] = []
    not_started: list[str] = []

    for model_key, model_info in unified_models.items():
        if model_key == "_registry_metadata":
            continue

        model_name = model_info.get("name", model_key)
        task_type = model_info.get("dataset_type", "restoration")
        sota_targets = model_info.get("sota_targets", {})

        # Max resolution
        max_res = model_info.get("val_resolution", 0)
        if not max_res:
            opt = model_info.get("optimization", {})
            ladder = opt.get("res_ladder", [])
            if ladder:
                max_res = max(ladder)
        if not max_res:
            max_res = 256  # Fallback

        model_dir = os.path.join(hub_dir, model_key)
        metrics_csv = os.path.join(model_dir, "metrics.csv")

        # 1. Parse Metrics Progress
        epoch = 0
        current_data = 1.0
        current_res = float(max_res)
        current_qs = 0.0

        current_phase = 1
        current_fold = 1

        if os.path.exists(metrics_csv):
            try:
                with open(metrics_csv, encoding="utf-8") as f:
                    reader = list(csv.DictReader(f))
                    if reader:
                        last_row = reader[-1]
                        epoch_val = last_row.get("Epoch")
                        epoch = int(epoch_val) if epoch_val is not None else 0
                        data_val = last_row.get("Data")
                        current_data = float(data_val) if data_val is not None else 1.0
                        res_val = last_row.get("Res")
                        current_res = float(res_val) if res_val is not None else float(current_res)
                        qs_val = last_row.get("Quality_Score")
                        current_qs = float(qs_val) if qs_val is not None else 0.0

                        if task_type == "forex":
                            phase_val = last_row.get("Phase")
                            current_phase = int(phase_val) if phase_val is not None else 1
                            fold_val = last_row.get("Fold")
                            current_fold = int(fold_val) if fold_val is not None else 1
            except Exception as e:
                print(f"[REMEDY] Exception encountered while parsing metrics: {e}")

        completeness = 0.0
        ladder_passed = False

        if epoch > 1:
            target_qs = 0.0
            if sota_targets:
                target_qs, _ = engine.compute_quality_score(sota_targets, sota_targets, task_type)

            if target_qs > 0:
                clamped_cqs = max(0.0, current_qs)
                qs_pct = min(1.0, clamped_cqs / target_qs)
            else:
                qs_pct = min(1.0, epoch / 100.0)  # Fallback heuristic

            if task_type == "forex":
                phase_pct = min(1.0, current_phase / 4.0)
                fold_pct = min(1.0, current_fold / 6.0)
                completeness = (phase_pct + fold_pct + qs_pct) / 3.0 * 100.0
                ladder_passed = current_fold >= 6
            else:
                data_pct = min(1.0, current_data)
                res_pct = min(1.0, current_res / max_res) if max_res > 0 else 1.0
                completeness = (data_pct + res_pct + qs_pct) / 3.0 * 100.0
                ladder_passed = current_res >= max_res

        # 2. Check if SOTA achieved
        # Invariant: Training should NEVER be declared complete if it didn't reach 100% data fraction
        # on max resolution and hasn't trained for at least 30 epochs
        is_sota = (
            epoch >= 30
            and current_data >= 0.999
            and ladder_passed
            and completeness >= 99.99
        )

        if is_sota:
            completed.append(model_name)
        elif epoch >= 1:
            in_progress.append((
                model_name,
                completeness,
                epoch,
                current_data,
                current_res,
                task_type,
                current_phase,
                current_fold,
            ))
        else:
            not_started.append(model_name)

    # Build Markdown Dashboard
    lines = [
        "# LemGendary AI Training Matrix",
        "",
        f"Auto-generated live dashboard | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Completed (SOTA Targets Achieved)",
        "",
        (
            "Models that have fully satisfied their strict target benchmarks "
            "and exported their architecture for production deployment."
        ),
        "",
    ]

    if completed:
        for m in sorted(completed):
            lines.append(f"- {m} ![SOTA](https://img.shields.io/badge/Status-SOTA-brightgreen)")
    else:
        lines.append("- *No models have achieved SOTA targets yet.*")

    lines.extend([
        "",
        "## In Progress (Training)",
        "",
        "Models actively scaling the resolution ladder and optimizing weights across dataset manifolds.",
        "",
    ])

    if in_progress:
        lines.append("| Model | Completeness | Epochs | Data / Phase | Res / Fold |")
        lines.append("| --- | --- | --- | --- | --- |")
        # Sort descending by completeness
        in_progress.sort(key=lambda x: x[1], reverse=True)
        for m, pct, ep, df, res, t_type, ph, fd in in_progress:
            if t_type == "forex":
                lines.append(f"| {m} | **{pct:.1f}%** | {ep} | Phase {ph} | Fold {fd} |")
            else:
                lines.append(f"| {m} | **{pct:.1f}%** | {ep} | {df*100:.0f}% | {int(res)}px |")
    else:
        lines.append("- *No models currently in active training.*")

    lines.extend([
        "",
        "## Not Started",
        "",
        "Registered matrix targets awaiting cluster allocation.",
        "",
    ])

    if not_started:
        for m in sorted(not_started):
            lines.append(f"- {m}")
    else:
        lines.append("- *All models have been deployed.*")

    lines.extend([
        "",
        "---",
        "",
        "## Canonical Directory Topology & Storage Invariants",
        "",
        "Every model repository in `LemGendaryModels/` adheres to a strict, standardized directory structure:",
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
        "│       ├── onnx_fp32/      # Variation 2: ONNX FP32 with external data (*_FP32.onnx*)",
        "│       └── onnx_fp16/      # Variation 3: Edge-optimized ONNX FP16 (*.onnx)",
        "└── visualizations/         # Training curves, confusion matrices, and diagnostics",
        "```",
        "",
        "## Strict 3-Variation Isolation Contract",
        "",
        "All model distributions published to Kaggle Models Hub must satisfy strict binary isolation:",
        "",
        "1. **Variation 1 (`pytorch`)**: Exclusively standalone PyTorch weights (`<ModelStem>.pt`).",
        (
            "2. **Variation 2 (`onnx_fp32`)**: Exclusively standard precision ONNX model (`<ModelStem>_FP32.onnx`)"
            " and sidecar tensor data (`<ModelStem>_FP32.onnx.data`)."
        ),
        (
            "3. **Variation 3 (`onnx_fp16`)**: Exclusively half-precision embedded ONNX model (`<ModelStem>.onnx`)"
            " with zero external tensor references."
        ),
        "",
        (
            "This contract is strictly validated prior to upload via `StrictIsolationValidator`"
            " to ensure zero cross-contamination."
        ),
        "",
    ])

    # Save
    os.makedirs(hub_dir, exist_ok=True)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[HUB SYNC] Dashboard successfully updated at: {readme_path}")


if __name__ == "__main__":
    generate_hub_readme()
