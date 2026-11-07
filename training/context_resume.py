"""Checkpoint resume for ``build_training_context``.

Selects the most advanced checkpoint, restores model, optimizer, scheduler and governor
state, reconciles the curriculum with ``metrics.csv`` and resynchronizes the train loader
with the restored resolution, data fraction and batch size.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.checkpoint import (
    CheckpointRecoveryEngine,
    parse_resume_state,
    safe_load_checkpoint,
)
from training.checkpoint.manager import CHECKPOINT_LOAD_ERRORS
from training.context_data import RunSpec, rebuild_resumed_train_loader
from training.fid_reeval.ledger import filter_metrics_row_for_fid_provenance
from training.governance.governor_setup import configured_data_fraction
from training.hardware.probe import VramProbe, audit_hardware_vram
from training.utils.error_types import STATE_RESTORE_ERRORS

logger = logging.getLogger("lemtrain.core_loop")


@dataclass
class TrainState:
    """Objects whose state a checkpoint restores."""

    raw_model: Any
    model: Any
    optimizer: Any
    scheduler: Any
    governor: Any
    sota_tracker: Any


@dataclass
class LoaderState:
    """The train dataset and loader that a resume may resize and rebuild."""

    train_ds: Any
    train_loader: Any
    batch_size: int
    sample_fraction: float


def select_resume_candidate(spec: RunSpec) -> Path | None:
    """Return the discovered checkpoint with the highest epoch (newest on ties)."""
    recovery_engine = CheckpointRecoveryEngine(
        workspace_root=spec.project_root.parent,
        project_root=spec.project_root,
        env=spec.args.env,
    )
    candidate_roots = recovery_engine.find_candidate_roots(spec.model_key, config=spec.config)
    discovered = recovery_engine.discover_checkpoints(candidate_roots, spec.model_key)
    best_candidate: Path | None = None
    best_cand_epoch = -1
    best_cand_mtime = 0.0

    for cand_key in ("latest", "best", "progress", "latest_epoch"):
        cand_p = discovered.get(cand_key)
        if not (cand_p and cand_p.exists()):
            continue
        try:
            c_data = safe_load_checkpoint(cand_p, map_location="cpu")
            c_ep = int(c_data.get("epoch", 0)) if isinstance(c_data, dict) else 0
            c_mt = cand_p.stat().st_mtime
            if c_ep > best_cand_epoch or (c_ep == best_cand_epoch and c_mt > best_cand_mtime):
                best_candidate = cand_p
                best_cand_epoch = c_ep
                best_cand_mtime = c_mt
        except CHECKPOINT_LOAD_ERRORS as cand_err:
            logger.warning("Checkpoint candidate %s could not be inspected: %s", cand_p, cand_err)
            if best_candidate is None:
                best_candidate = cand_p
    return best_candidate


def _restore_weights_and_optimizer(spec: RunSpec, loaded: dict, train: TrainState) -> None:
    """Load model weights, then optimizer and scheduler state (unless ``--reset-scheduler``)."""
    model_sd = (
        loaded.get("model_state") or loaded.get("model_state_dict") or loaded.get("state_dict")
    )
    if isinstance(model_sd, dict):
        train.raw_model.load_state_dict(model_sd, strict=False)
    opt_state = loaded.get("optimizer_state") or loaded.get("optimizer_state_dict")
    if isinstance(opt_state, dict):
        try:
            train.optimizer.load_state_dict(opt_state)
        except STATE_RESTORE_ERRORS as opt_err:
            logger.warning("Optimizer state could not be loaded cleanly: %s", opt_err)
    sched_state = loaded.get("scheduler_state") or loaded.get("scheduler_state_dict")
    if getattr(spec.args, "reset_scheduler", False):
        logger.info(
            "--reset-scheduler set: ignoring checkpoint scheduler state "
            "and starting a fresh LR curve."
        )
    elif isinstance(sched_state, dict) and train.scheduler is not None:
        try:
            train.scheduler.load_state_dict(sched_state)
        except STATE_RESTORE_ERRORS as sched_err:
            logger.warning("Scheduler state could not be loaded cleanly: %s", sched_err)


def calibrate_quality_anchor(spec: RunSpec, governor: Any) -> None:
    """Reset the artificial pre-fix quality anchor of synthesized and face models.

    ``governor.quality_anchor_recalibration`` in ``config.yaml`` holds the limits: a restored
    ``best_quality`` above ``above`` is set to ``reset_to``. Without that section nothing changes.
    """
    limits = spec.config.get("governor", {}).get("quality_anchor_recalibration") or {}
    if "above" not in limits or "reset_to" not in limits:
        return
    above, reset_to = float(limits["above"]), float(limits["reset_to"])
    if (
        spec.model_info.get("needs_synthesis") or spec.model_info.get("dataset_type") == "face"
    ) and float(getattr(governor, "best_quality", 0.0) or 0.0) > above:
        logger.info(
            "[CALIBRATION] Calibrating artificial quality anchor (%.2f -> %.2f)",
            float(governor.best_quality),
            reset_to,
        )
        governor.best_quality = reset_to


def reconcile_metrics_csv(governor: Any, metrics_csv: Path | None, sample_fraction: float) -> None:
    """Raise the governor's data fraction and best quality to what ``metrics.csv`` records."""
    if metrics_csv is None or not metrics_csv.exists():
        return
    try:
        with open(metrics_csv, "r", encoding="utf-8") as mf:
            rows = list(csv.DictReader(mf))
    except OSError as csv_err:
        logger.debug("Could not reconcile metrics.csv on resume: %s", csv_err)
        return
    if not rows:
        return
    curr_res_val = int(getattr(governor, "current_res", 256))
    max_frac_at_res = float(getattr(governor, "current_fraction", sample_fraction))
    max_quality_csv = float(getattr(governor, "best_quality", 0.0))
    for raw_row in rows:
        r = filter_metrics_row_for_fid_provenance(raw_row, metrics_csv.parent)
        try:
            r_res = int(float(r.get("Res") or curr_res_val))
            r_frac = float(r.get("Data") or 0.0)
            r_qual = float(r.get("Quality_Score") or 0.0)
            if r_res == curr_res_val and r_frac > max_frac_at_res:
                max_frac_at_res = r_frac
            max_quality_csv = max(max_quality_csv, r_qual)
        except (ValueError, TypeError) as row_err:
            logger.debug("Skipping malformed metrics.csv row %r: %s", r, row_err)
    if max_frac_at_res > float(getattr(governor, "current_fraction", 0.0)):
        governor.current_fraction = min(1.0, max_frac_at_res)
    if max_quality_csv > float(getattr(governor, "best_quality", 0.0)):
        governor.best_quality = max_quality_csv


def _resync_train_loader(spec: RunSpec, train: TrainState, data: LoaderState) -> None:
    """Resize the train dataset to the restored rung and rebuild its loader."""
    governor = train.governor
    rec_res = getattr(governor, "current_res", None)
    rec_frac = getattr(governor, "current_fraction", None)
    if spec.is_llm:
        rec_frac = configured_data_fraction(spec.model_info) or 1.0
        governor.current_fraction = rec_frac
    if rec_frac is None and rec_res is None:
        return
    train_ds = data.train_ds
    if hasattr(train_ds, "update_strategy"):
        train_ds.update_strategy(size=rec_res, fraction=rec_frac)
    elif hasattr(train_ds, "sample_fraction") and rec_frac is not None:
        train_ds.sample_fraction = rec_frac
    # Re-audit the hardware safety limit for the restored resolution.
    hw_safe_batch = audit_hardware_vram(
        model_key=spec.model_key,
        model_info=spec.model_info,
        config=spec.config,
        model=train.model,
        probe=VramProbe(
            device=spec.device,
            res_override=rec_res,
            mode="train",
            sample_fraction=rec_frac if rec_frac is not None else 1.0,
            fold=getattr(spec.args, "fold", None),
            pairs=getattr(spec.args, "pairs", None) or spec.model_info.get("pairs"),
        ),
    )
    restored_batch = int(getattr(governor, "current_batch", data.batch_size))
    curr_batch = min(restored_batch, hw_safe_batch)
    if curr_batch != restored_batch:
        logger.warning(
            "[RESUME CLAMP] Clamped restored batch %d -> %d to prevent "
            "OOM on current hardware at res %s",
            restored_batch,
            curr_batch,
            rec_res,
        )
        if hasattr(governor, "update_batch_and_accumulation"):
            governor.update_batch_and_accumulation(curr_batch)

    data.train_loader = rebuild_resumed_train_loader(spec, train_ds, data.train_loader, curr_batch)
    total_ds_samples = len(getattr(train_ds, "all_samples", [])) or len(train_ds)
    shown_frac = rec_frac if rec_frac is not None else 1.0
    if os.environ.get("RANK", "0") == "0" and spec.is_llm:
        ctx_len = rec_res or spec.model_info.get("input_size") or 2048
        print(
            f"[CURRICULUM] Active LLM instruction manifold: {len(train_ds)} samples "
            f"({shown_frac * 100:.0f}% data fraction) | Context: {ctx_len} tokens | "
            f"Batch: {curr_batch} | Total Batches: {len(data.train_loader)}",
            flush=True,
        )
    elif os.environ.get("RANK", "0") == "0":
        print(
            "[CURRICULUM] Active training manifold: "
            f"{len(train_ds)}/{total_ds_samples} samples "
            f"({shown_frac * 100:.0f}% data fraction) | "
            f"Res: {rec_res}px | Batch: {curr_batch} | "
            f"Total Batches: {len(data.train_loader)}",
            flush=True,
        )
    logger.info(
        "Synchronized training dataset and loader with recovered governor state: res=%s, frac=%.2f",
        rec_res,
        shown_frac,
    )


def _restore_governor(
    spec: RunSpec, loaded: dict, train: TrainState, data: LoaderState, metrics_csv: Path | None
) -> None:
    """Restore the governor's curriculum state and resynchronize the train loader with it."""
    governor = train.governor
    gov_state = dict(loaded.get("governor_state") or {})
    ckpt_epoch = int(loaded.get("epoch", 0) or 0)
    if ckpt_epoch > 0:
        gov_state["epoch_count"] = max(int(gov_state.get("epoch_count", 0) or 0), ckpt_epoch)
    if not hasattr(governor, "load_state"):
        return
    try:
        governor.load_state(gov_state)
        calibrate_quality_anchor(spec, governor)
        reconcile_metrics_csv(governor, metrics_csv, data.sample_fraction)
        _resync_train_loader(spec, train, data)
    except STATE_RESTORE_ERRORS as gov_err:
        logger.warning("Governor state could not be loaded cleanly: %s", gov_err)


def _restore_sota_best(loaded: dict, train: TrainState) -> None:
    """Carry the checkpoint's best quality score into the SOTA tracker."""
    if not hasattr(train.sota_tracker, "best_quality"):
        return
    saved_best = (
        loaded.get("best_quality_score")
        or loaded.get("best_quality")
        or getattr(train.governor, "best_quality", None)
    )
    if saved_best is None:
        return
    try:
        train.sota_tracker.best_quality = max(
            float(train.sota_tracker.best_quality), float(saved_best)
        )
    except (ValueError, TypeError) as best_err:
        logger.warning("Ignoring non-numeric saved best quality %r: %s", saved_best, best_err)


def resume_from_checkpoint(
    spec: RunSpec, train: TrainState, data: LoaderState, metrics_csv: Path | None
):
    """Restore the best checkpoint into ``train`` and ``data``; return its resume state.

    Returns ``None`` when there is no usable checkpoint. ``data.train_loader`` is replaced
    when the restored curriculum changes the dataset or batch size.
    """
    ckpt_candidate = select_resume_candidate(spec)
    if not (ckpt_candidate and ckpt_candidate.exists()):
        return None
    loaded = safe_load_checkpoint(ckpt_candidate, map_location=spec.device)
    if not loaded:
        return None
    _restore_weights_and_optimizer(spec, loaded, train)
    _restore_governor(spec, loaded, train, data, metrics_csv)
    _restore_sota_best(loaded, train)
    return parse_resume_state(loaded)
