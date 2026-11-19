"""Post-training target audit and diagnostic operator guidance.

Implements Pathology 16 (Universal Post-Training Target Audit & Interactive Guidance):
when a training run finishes its epoch budget, this utility audits achieved metrics
against the configured SOTA targets and outputs clear benchmark diagnostics.
"""

from __future__ import annotations

import logging
from typing import Any

from training.governance.metrics import DEFAULT_METRIC_DIRECTIONS, is_lower_better

logger = logging.getLogger("lemtrain.target_audit")


def _is_metric_higher_better(name: str) -> bool:
    """Return True if higher values indicate improvement."""
    clean = name.lower().replace(" ", "_")
    if clean in DEFAULT_METRIC_DIRECTIONS:
        return DEFAULT_METRIC_DIRECTIONS[clean]
    return not is_lower_better(clean)


def audit_final_targets(
    achieved_metrics: dict[str, float],
    sota_targets: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compare achieved metrics against target thresholds.

    Returns a dict mapping metric names to:
        - achieved: float value
        - target: float value
        - higher_is_better: bool
        - met: bool
        - gap: float (positive indicates deficit)
    """
    results: dict[str, dict[str, Any]] = {}
    if not isinstance(sota_targets, dict):
        return results

    for key, target_val in sota_targets.items():
        if target_val is None:
            continue
        try:
            target_f = float(target_val)
        except (ValueError, TypeError):
            continue

        achieved = achieved_metrics.get(key)
        if achieved is None:
            # Check for prefixed variants, e.g. test_psnr
            for k, v in achieved_metrics.items():
                if k.endswith(f"_{key}"):
                    achieved = v
                    break

        higher_is_better = _is_metric_higher_better(key)
        if achieved is not None:
            achieved_f = float(achieved)
            if higher_is_better:
                met = achieved_f >= target_f
                gap = max(0.0, target_f - achieved_f)
            else:
                met = achieved_f <= target_f
                gap = max(0.0, achieved_f - target_f)
        else:
            achieved_f = None
            met = False
            gap = target_f

        results[key] = {
            "achieved": achieved_f,
            "target": target_f,
            "higher_is_better": higher_is_better,
            "met": met,
            "gap": gap,
        }
    return results


def format_post_training_audit(
    model_name: str,
    final_epoch: int,
    status: str,
    achieved_metrics: dict[str, float],
    sota_targets: dict[str, Any],
) -> str:
    """Format structured benchmark comparison table and diagnostic advice."""
    audit_results = audit_final_targets(achieved_metrics, sota_targets)
    lines: list[str] = [
        "",
        "============================================================",
        f"POST-TRAINING TARGET AUDIT: {model_name} (Epoch {final_epoch})",
        f"Execution Status: {status.upper()}",
        "------------------------------------------------------------",
    ]

    all_met = True
    if audit_results:
        lines.append(f"{'Metric':<16} {'Achieved':<12} {'Target':<12} {'Status':<10}")
        lines.append("-" * 52)
        for metric, data in audit_results.items():
            ach_str = f"{data['achieved']:.4f}" if data["achieved"] is not None else "N/A"
            tgt_str = f"{data['target']:.4f}"
            status_str = "[PASS]" if data["met"] else "[DEFICIT]"
            if not data["met"]:
                all_met = False
            lines.append(f"{metric:<16} {ach_str:<12} {tgt_str:<12} {status_str:<10}")
        lines.append("-" * 52)
    else:
        lines.append("No explicit sota_targets configured for this model.")

    if all_met and audit_results:
        lines.append("[VERDICT] All SOTA benchmarks satisfied. Model is production-ready.")
    else:
        lines.append("[DIAGNOSTIC GUIDANCE]")
        lines.append("  1. Escalate to Kaggle Cloud Hub for high-VRAM / larger batch training.")
        lines.append("  2. Resume locally with '--epochs N' to extend training budget.")
        lines.append("  3. Deploy targeted metric loss scaling in unified_models_v2.yaml.")
    lines.append("============================================================")
    lines.append("")
    return "\n".join(lines)
