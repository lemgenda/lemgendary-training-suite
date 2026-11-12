"""Per-model statistics of the Desktop GUI model catalog.

:func:`model_with_stats` builds one :class:`ModelWithStats` entry from the
registry entry, the checkpoints and the ``metrics.csv`` of a model.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from training.cloud.variations import ONNX_FP16, ONNX_FP32, PYTORCH_FP32, variation_ready
from training.export.validation import resolve_model_stem
from training.fid_reeval.ledger import filter_metrics_row_for_fid_provenance
from training.governance.curriculum import (
    DEFAULT_FOREX_LADDER,
    DEFAULT_SPATIAL_LADDER,
    is_forex_model,
)
from training.governance.metrics import is_lower_better
from training.server.routes.gui_schemas import ModelWithStats
from training.server.routes.gui_status import RungTracker, training_verdict

logger = logging.getLogger("lemtrain.server.routes.gui")

# Comprehensive Metric Registry mapping YAML keys to CSV columns, directionality, and labels
# yaml_key: (csv_col, lower_is_better, display_label, is_forex)
METRIC_REGISTRY: dict[str, tuple[str, bool, str, bool]] = {
    # Vision restoration & quality
    "psnr": ("PSNR", False, "PSNR (dB)", False),
    "ssim": ("SSIM", False, "SSIM", False),
    "lpips": ("LPIPS", True, "LPIPS", False),
    "fid": ("FID", True, "FID", False),
    "mae": ("MAE", True, "MAE", False),
    "srcc": ("SRCC", False, "SRCC", False),
    "plcc": ("PLCC", False, "PLCC", False),
    "accuracy": ("Accuracy", False, "Accuracy", False),
    "rank_margin": ("Rank_Margin", True, "Rank Margin", False),
    # Vision detection / segmentation
    "miou": ("mIoU", False, "mIoU", False),
    "map50": ("mAP50", False, "mAP50", False),
    "map50_95": ("mAP50-95", False, "mAP50-95", False),
    "map_easy": ("mAP_Easy", False, "mAP Easy", False),
    "map_medium": ("mAP_Medium", False, "mAP Medium", False),
    "map_hard": ("mAP_Hard", False, "mAP Hard", False),
    # Forex Trading Telemetry
    "dir_acc": ("DirAcc", False, "Dir Acc (%)", True),
    "win_rate": ("WinRate", False, "Win Rate (%)", True),
    "profit_factor": ("ProfitFactor", False, "Profit Factor", True),
    "sharpe_ratio": ("Sharpe", False, "Sharpe Ratio", True),
    "sortino_ratio": ("Sortino", False, "Sortino Ratio", True),
    "max_drawdown": ("MaxDD", True, "Max DD (%)", True),
    "tp_mae": ("TP_MAE", True, "TP MAE", True),
    "sl_mae": ("SL_MAE", True, "SL MAE", True),
    # LLM / Causal Assistant Benchmarks
    "eval_loss": ("Eval_Loss", True, "Eval Loss", False),
    "perplexity": ("Perplexity", True, "Perplexity", False),
    "token_accuracy": ("Token_Accuracy", False, "Token Accuracy (%)", False),
}

# Headline metric priority: (yaml_key, label, lower_is_better)
_FOREX_PRIORITY: tuple[tuple[str, str, bool], ...] = (
    ("dir_acc", "Dir Acc (%)", False),
    ("win_rate", "Win Rate (%)", False),
    ("profit_factor", "Profit Factor", False),
    ("sharpe_ratio", "Sharpe Ratio", False),
    ("sortino_ratio", "Sortino Ratio", False),
    ("max_drawdown", "Max DD (%)", True),
    ("tp_mae", "TP MAE", True),
    ("sl_mae", "SL MAE", True),
)
_VISION_PRIORITY: tuple[tuple[str, str, bool], ...] = (
    ("psnr", "PSNR (dB)", False),
    ("srcc", "SRCC", False),
    ("accuracy", "Accuracy", False),
    ("map50", "mAP50", False),
    ("miou", "mIoU", False),
    ("mae", "MAE", True),
    ("auc", "AUC-ROC", False),
    ("ssim", "SSIM", False),
    ("map_easy", "mAP Easy", False),
    ("eval_loss", "Eval Loss", True),
    ("perplexity", "Perplexity", True),
)

_FOREX_LADDER: tuple[int, ...] = DEFAULT_FOREX_LADDER
_SPATIAL_LADDER: tuple[int, ...] = DEFAULT_SPATIAL_LADDER

_EPOCH_COLS = ("Epoch", "epoch")
_RES_COLS = ("Res", "res", "resolution", "Resolution", "imgsz")
_FOLD_COLS = ("Fold", "fold")
_DATA_COLS = ("Data", "data", "sample_fraction", "Data_Fraction")


def _as_float(value: Any) -> float | None:
    """Return ``value`` as a float, or ``None`` when it is not numeric."""
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _ModelKind(NamedTuple):
    """Domain of a model: forex, LLM, or (neither) vision."""

    is_forex: bool
    is_llm: bool


class _Ladder(NamedTuple):
    """Training ladder of a model: its kind, its steps and the top step."""

    ladder_type: str
    steps: list[int]
    target_res: int


class _Primary(NamedTuple):
    """Headline SOTA metric of a model."""

    key: str | None
    metric_name: str
    target: Any
    lower_is_better: bool


class _Progress(NamedTuple):
    """Ladder and data-fraction pillars of a model."""

    active_res: Any
    ladder_passed: bool
    data_fraction_passed: bool
    data_fraction: float


@dataclass
class _CsvStats:
    """What a ``metrics.csv`` replay tells about a model."""

    epochs: int = 0
    max_res: int = 0
    latest_res: int | None = None
    max_fold: int = 0
    latest_fold: int | None = None
    history: dict[str, list[float]] = field(default_factory=dict)


def _model_kind(model_key: str, info: dict[str, Any]) -> _ModelKind:
    is_forex = is_forex_model(model_key, info)
    is_llm = (
        info.get("category") == "llm"
        or info.get("dataset_type") in ("text_instruction", "llm")
        or "llm" in model_key.lower()
    )
    return _ModelKind(is_forex, is_llm)


def _configured_ladder(info: dict[str, Any]) -> list[int] | None:
    opt_cfg = info.get("optimization")
    if (
        isinstance(opt_cfg, dict)
        and "res_ladder" in opt_cfg
        and isinstance(opt_cfg["res_ladder"], list)
    ):
        return opt_cfg["res_ladder"]
    return None


def _spatial_steps(info: dict[str, Any]) -> list[int]:
    configured = _configured_ladder(info)
    if configured is not None:
        return configured
    if info.get("input_size"):
        inp = info["input_size"]
        if isinstance(inp, (list, tuple)) and len(inp) >= 2:
            return [inp[-1]]
        if isinstance(inp, int):
            return [inp]
        return list(_SPATIAL_LADDER)
    if info.get("resolution"):
        return [info["resolution"]]
    return list(_SPATIAL_LADDER)


def _ladder(info: dict[str, Any], kind: _ModelKind) -> _Ladder:
    if kind.is_forex:
        configured = _configured_ladder(info)
        ladder_type = "timeframe"
        steps = configured if configured is not None else list(_FOREX_LADDER)
    elif kind.is_llm:
        ladder_type = "context_tokens"
        ctx_len = info.get("curriculum", {}).get("max_seq_length") or info.get("input_size") or 2048
        steps = [int(ctx_len)]
    else:
        ladder_type = "spatial"
        steps = _spatial_steps(info)
    # Target resolution from spatial ladder or model metadata
    target_res: int = (
        max(steps) if steps else (info.get("resolution") or (512 if not kind.is_forex else 1440))
    )
    return _Ladder(ladder_type, steps, target_res)


def _primary_metric(sota: Any, is_forex: bool) -> _Primary:
    if not (isinstance(sota, dict) and sota):
        return _Primary(None, "Metric", None, False)
    priority = _FOREX_PRIORITY if is_forex else _VISION_PRIORITY
    match = next((p for p in priority if p[0] in sota), None)
    if match is not None and sota[match[0]] is not None:
        key, label, is_low = match
        return _Primary(key, label, sota[key], is_low)
    first_k = next(iter(sota))
    return _Primary(
        first_k,
        first_k.upper(),
        sota[first_k],
        is_lower_better(first_k),
    )


def _choose_csv(candidates: list[Path]) -> Path | None:
    chosen: Path | None = None
    for cand in candidates:
        if (
            cand.exists()
            and cand.stat().st_size > 0
            and (chosen is None or cand.stat().st_mtime > chosen.stat().st_mtime)
        ):
            chosen = cand
    return chosen


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first truthy ``row`` value among ``keys``, else ``None``."""
    return next((row.get(k) for k in keys if row.get(k)), None)


def _read_epoch(stats: _CsvStats, row: dict[str, Any], path: Path) -> None:
    ep_val = _first_value(row, _EPOCH_COLS)
    if not ep_val:
        return
    try:
        stats.epochs = max(stats.epochs, int(ep_val))
    except ValueError:
        logger.debug("Non-integer Epoch %r in %s", ep_val, path)


def _read_res(stats: _CsvStats, row: dict[str, Any], path: Path) -> int | None:
    res_val = _first_value(row, _RES_COLS)
    if not res_val:
        return None
    try:
        res_num = int(float(res_val))
    except ValueError:
        logger.debug("Non-numeric resolution %r in %s", res_val, path)
        return None
    stats.latest_res = res_num
    stats.max_res = max(stats.max_res, res_num)
    return res_num


def _read_fold(stats: _CsvStats, row: dict[str, Any], path: Path) -> None:
    fold_val = _first_value(row, _FOLD_COLS)
    if not fold_val:
        return
    try:
        fold_num = int(fold_val)
    except ValueError:
        logger.debug("Non-integer Fold %r in %s", fold_val, path)
        return
    stats.latest_fold = fold_num
    stats.max_fold = max(stats.max_fold, fold_num)


def _read_fraction(row: dict[str, Any], path: Path) -> float | None:
    data_val = _first_value(row, _DATA_COLS)
    if not data_val:
        return None
    try:
        d_num = float(data_val.replace("%", "").strip())
    except ValueError:
        logger.debug("Non-numeric data fraction %r in %s", data_val, path)
        return None
    return d_num / 100.0 if d_num > 1.0 else d_num


def _read_metrics(
    history: dict[str, list[float]], row: dict[str, Any], is_forex: bool, path: Path
) -> None:
    """Collect all recognized metrics for the model domain."""
    for ykey, (col_name, _is_lower, _lbl, is_fx_col) in METRIC_REGISTRY.items():
        if is_forex != is_fx_col or not row.get(col_name):
            continue
        try:
            c_val = float(row[col_name])
        except (ValueError, TypeError):
            logger.debug("Non-numeric %s value %r in %s", col_name, row.get(col_name), path)
            continue
        history.setdefault(ykey, []).append(c_val)


def _read_csv_stats(
    chosen: Path | None,
    is_forex: bool,
    model_key: str,
    hub_dir: Path | None = None,
) -> tuple[_CsvStats, RungTracker]:
    stats = _CsvStats()
    rungs = RungTracker(str(chosen))
    if chosen is None:
        return stats, rungs
    provenance_hub = hub_dir if hub_dir is not None else chosen.parent
    try:
        with open(chosen, encoding="utf-8", errors="ignore") as f:
            for raw_row in csv.DictReader(f):
                row = filter_metrics_row_for_fid_provenance(raw_row, provenance_hub)
                _read_epoch(stats, row, chosen)
                row_res = _read_res(stats, row, chosen)
                _read_fold(stats, row, chosen)
                rungs.add(row_res, _read_fraction(row, chosen))
                _read_metrics(stats.history, row, is_forex, chosen)
    except (OSError, csv.Error) as exc:
        logger.warning("Failed reading metrics CSV %s for %s: %s", chosen, model_key, exc)
    return stats, rungs


def _has_weights(model_key: str, info: dict[str, Any], root: Path, hub_dir: Path) -> bool:
    if hub_dir.exists() and (
        list(hub_dir.glob("*.pth"))
        or list((hub_dir / "checkpoints").glob("*.pth"))
        or list(hub_dir.glob("*.pt"))
        or list(hub_dir.glob("*.onnx"))
    ):
        return True
    ckpt_filename = info.get("checkpoint")
    if ckpt_filename and any(
        cand.exists()
        for cand in [
            root / "checkpoints" / ckpt_filename,
            root / "checkpoints" / model_key / ckpt_filename,
            root / "weights" / ckpt_filename,
            hub_dir / ckpt_filename,
            hub_dir / "checkpoints" / ckpt_filename,
        ]
    ):
        return True
    return (root / "checkpoints" / f"{model_key}.pt").exists()


def _has_checkpoint(
    model_key: str,
    info: dict[str, Any],
    root: Path,
    hub_dir: Path,
    model_ckpts: list[dict[str, Any]],
) -> bool:
    weights_exist = _has_weights(model_key, info, root, hub_dir)
    best_ckpt = next((c for c in model_ckpts if c.get("is_best")), None)
    return best_ckpt is not None or len(model_ckpts) > 0 or weights_exist


def _readiness_fields(model_key: str, hub_dir: Path) -> dict[str, bool]:
    """Per-format flags: whether the model hub holds each exported binary."""
    stem = resolve_model_stem(model_key)
    return {
        "pytorch_ready": variation_ready(hub_dir, PYTORCH_FP32, stem),
        "onnx_fp32_ready": variation_ready(hub_dir, ONNX_FP32, stem),
        "onnx_fp16_ready": variation_ready(hub_dir, ONNX_FP16, stem),
    }


def _metric_meta(ykey: str) -> tuple[bool, str]:
    """Return ``(lower_is_better, label)`` of one SOTA target key."""
    reg_entry = METRIC_REGISTRY.get(ykey)
    if reg_entry:
        _col_name, is_low, lbl, _ = reg_entry
        return is_low, lbl
    is_low = is_lower_better(ykey)
    return is_low, ykey.replace("_", " ").title()


def _sota_detail(
    ykey: str, tgt_val: Any, history: dict[str, list[float]], model_key: str
) -> dict[str, Any]:
    is_low, lbl = _metric_meta(ykey)
    achieved_vals = history.get(ykey, [])
    achieved: float | None = None
    passed = False
    if achieved_vals:
        achieved = min(achieved_vals) if is_low else max(achieved_vals)
        try:
            tgt_num = float(tgt_val)
            passed = (achieved <= tgt_num) if is_low else (achieved >= tgt_num)
        except (ValueError, TypeError):
            logger.debug("Non-numeric SOTA target %r for %s.%s", tgt_val, model_key, ykey)
    return {
        "key": ykey,
        "label": lbl,
        "target": tgt_val,
        "achieved": round(achieved, 4) if achieved is not None else None,
        "lower_is_better": is_low,
        "passed": passed,
    }


def _best_from_sota(s: dict[str, Any]) -> dict[str, Any]:
    # Only measured values count; a missing measurement is never replaced by its target.
    b_val = s["achieved"]
    b_num = _as_float(b_val)
    tgt_num = _as_float(s["target"])
    passed_sota = (
        b_num is not None
        and tgt_num is not None
        and ((b_num <= tgt_num) if s["lower_is_better"] else (b_num >= tgt_num))
    )
    return {
        "key": s["key"],
        "label": s["label"],
        "value": b_val,
        "target": s["target"],
        "lower_is_better": s["lower_is_better"],
        "passed": s["passed"] or passed_sota,
    }


def _best_metrics_details(
    sota_details: list[dict[str, Any]],
    history: dict[str, list[float]],
    is_forex: bool,
) -> list[dict[str, Any]]:
    """Best achieved metrics across SOTA targets and metrics history."""
    details = [_best_from_sota(s) for s in sota_details]
    known_metric_keys = {s["key"] for s in sota_details}
    for ykey, (_col_name, is_low, lbl, is_fx_col) in METRIC_REGISTRY.items():
        if is_forex != is_fx_col or ykey in known_metric_keys:
            continue
        vals = history.get(ykey)
        if vals:
            ach = min(vals) if is_low else max(vals)
            details.append(
                {
                    "key": ykey,
                    "label": lbl,
                    "value": round(ach, 4),
                    "target": None,
                    "lower_is_better": is_low,
                    "passed": False,  # no target, so nothing to pass
                }
            )
    return details


def _headline_metric(
    primary: _Primary, history: dict[str, list[float]], sota_details: list[dict[str, Any]]
) -> Any:
    if primary.key and primary.key in history:
        p_vals = history[primary.key]
        return round(min(p_vals) if primary.lower_is_better else max(p_vals), 2)
    if sota_details and sota_details[0]["achieved"] is not None:
        return round(sota_details[0]["achieved"], 2)
    return None  # never the target when nothing was measured


def _sota_fields(
    model_key: str,
    info: dict[str, Any],
    is_forex: bool,
    history: dict[str, list[float]],
    has_checkpoint: bool,
) -> dict[str, Any]:
    """SOTA payload fields: ALL target metrics defined for the model must be met."""
    sota = info.get("sota_targets", {})
    primary = _primary_metric(sota, is_forex)
    details = (
        [_sota_detail(k, v, history, model_key) for k, v in sota.items()]
        if isinstance(sota, dict) and sota
        else []
    )
    total = len(sota) if isinstance(sota, dict) else 0
    met = sum(1 for d in details if d["passed"])
    reached = met == total if total > 0 else has_checkpoint
    return {
        "best_metric": _headline_metric(primary, history, details),
        "metric_name": primary.metric_name,
        "sota_target": primary.target,
        "sota_targets_total": total,
        "sota_targets_met": met,
        "sota_all_met": reached,
        "sota_details": details,
        "best_metrics_details": _best_metrics_details(details, history, is_forex),
        "sota_reached": reached,
    }


def _spatial_progress(
    stats: _CsvStats, rungs: RungTracker, ladder: _Ladder, completed_epochs: int
) -> _Progress:
    if stats.latest_res is not None:
        active_res = stats.latest_res
    elif stats.max_res > 0:
        active_res = stats.max_res
    elif ladder.steps:
        active_res = ladder.steps[0]
    else:
        active_res = ladder.target_res
    target_res = ladder.target_res
    ladder_passed = (
        (stats.max_res >= target_res) if target_res is not None else (completed_epochs >= 20)
    )
    data_fraction_passed = ladder_passed and rungs.full_data_at_top(ladder.steps or [target_res])
    # Data fraction of the rung the run last trained at (ratchet pair rule:
    # a fraction from a lower resolution step does not count at the top).
    data_fraction = rungs.current.fraction if rungs.current is not None else 0.0
    return _Progress(active_res, ladder_passed, data_fraction_passed, data_fraction)


def _progress(
    kind: _ModelKind,
    stats: _CsvStats,
    rungs: RungTracker,
    ladder: _Ladder,
    completed_epochs: int,
) -> _Progress:
    if kind.is_forex:
        ladder_default = ladder.steps[0] if ladder.steps else 1
        active_fold = (
            stats.latest_fold
            if stats.latest_fold is not None
            else (stats.max_fold if stats.max_fold > 0 else ladder_default)
        )
        ladder_passed = (stats.max_fold >= 6) or (stats.max_res >= 1440)
        return _Progress(active_fold, ladder_passed, True, 1.0)
    if kind.is_llm:
        return _Progress(ladder.target_res, True, True, 1.0)
    return _spatial_progress(stats, rungs, ladder, completed_epochs)


def _ladder_fields(
    kind: _ModelKind, ladder: _Ladder, stats: _CsvStats, progress: _Progress
) -> dict[str, Any]:
    is_forex = kind.is_forex
    active_res = progress.active_res
    max_res_completed = (
        stats.max_res
        if stats.max_res > 0
        else (stats.max_fold if is_forex and stats.max_fold > 0 else active_res)
    )
    return {
        "spatial_ladder": ladder.steps,
        "ladder_type": ladder.ladder_type,
        "is_forex": is_forex,
        "is_llm": kind.is_llm,
        "ladder_passed": progress.ladder_passed,
        "active_res": active_res,
        "max_res_completed": max_res_completed,
        "target_res": ladder.target_res,
        "active_fold": active_res if is_forex else None,
        "total_folds": 6 if is_forex else None,
        "fold_years_label": (
            f"{2019}-{2019 + min(active_res, 6)} -> {2019 + min(active_res, 6) + 1}"
            if is_forex and isinstance(active_res, int) and active_res >= 1
            else None
        ),
        "data_fraction_completed": 1.0 if is_forex else round(progress.data_fraction, 2),
        "data_fraction_passed": True if is_forex else progress.data_fraction_passed,
    }


def _latest_checkpoint(model_ckpts: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(model_ckpts, key=lambda c: c.get("epoch") or 0) if model_ckpts else None


def _completed_epochs(stats: _CsvStats, model_ckpts: list[dict[str, Any]]) -> int:
    latest_ckpt = _latest_checkpoint(model_ckpts)
    return max(
        stats.epochs,
        latest_ckpt["epoch"] if latest_ckpt and latest_ckpt.get("epoch") else 0,
    )


def _checkpoint_fields(model_ckpts: list[dict[str, Any]], has_checkpoint: bool) -> dict[str, Any]:
    best_ckpt = next((c for c in model_ckpts if c.get("is_best")), None)
    latest_ckpt = _latest_checkpoint(model_ckpts)
    return {
        "checkpoints_count": len(model_ckpts),
        "has_best_checkpoint": has_checkpoint,
        "best_checkpoint_size_mb": best_ckpt["size_mb"] if best_ckpt else None,
        "best_checkpoint_epoch": best_ckpt["epoch"] if best_ckpt else None,
        "latest_checkpoint_epoch": latest_ckpt["epoch"] if latest_ckpt else None,
        "checkpoint_exists": has_checkpoint,
    }


def _training_defaults(model_key: str, info: dict[str, Any], is_forex: bool) -> dict[str, Any]:
    is_yolo = model_key == "yolov8n"
    default_lr = 0.01 if is_yolo else (0.0001 if is_forex else 0.0002)
    default_bs = 16 if is_yolo else (128 if is_forex else 8)
    default_ep = 300 if is_yolo else (50 if is_forex else 30)
    return {
        "learning_rate": (
            float(info["learning_rate"])
            if isinstance(info.get("learning_rate"), (int, float))
            else default_lr
        ),
        "batch_size": (
            int(info["batch_size"])
            if isinstance(info.get("batch_size"), (int, float))
            else default_bs
        ),
        "default_epochs": int(info.get("epochs") or default_ep),
    }


def _identity_fields(model_key: str, info: dict[str, Any]) -> dict[str, Any]:
    category = info.get("category")
    dataset_type = info.get("dataset_type")
    task_type = category or (dataset_type if isinstance(dataset_type, str) else None)
    return {
        "model_key": model_key,
        "name": info.get("name", model_key),
        "category": info.get("category", "general"),
        "class_name": info.get("class_name"),
        "resolution": info.get("resolution"),
        "key": model_key,
        "display_name": info.get("name", model_key),
        "architecture": info.get("architecture_type") or info.get("class_name") or "PyTorch Model",
        "task_type": task_type or "general",
        "canonical_format": info.get("canonical_format", "webdataset"),
        "parameters_m": _as_float(info.get("parameters_m")),
        "preferred_parallel": info.get("preferred_parallel", "single"),
        "status": info.get("status", "VALIDATED"),
        "authoritative_status": info.get("status", "VALIDATED"),
        "description": info.get("description", "").strip(),
    }


def model_with_stats(
    model_key: str,
    info: dict[str, Any],
    root: Path,
    hub_root: Path,
    model_ckpts: list[dict[str, Any]],
) -> ModelWithStats:
    """Build the GUI catalog entry of one registry model."""
    kind = _model_kind(model_key, info)
    ladder = _ladder(info, kind)
    hub_dir = hub_root / model_key
    stats, rungs = _read_csv_stats(
        _choose_csv([hub_dir / "metrics.csv", root / "checkpoints" / model_key / "metrics.csv"]),
        kind.is_forex,
        model_key,
        hub_dir=hub_dir,
    )
    has_checkpoint = _has_checkpoint(model_key, info, root, hub_dir, model_ckpts)
    sota = _sota_fields(model_key, info, kind.is_forex, stats.history, has_checkpoint)
    completed_epochs = _completed_epochs(stats, model_ckpts)
    progress = _progress(kind, stats, rungs, ladder, completed_epochs)
    verdict = training_verdict(
        epochs_completed=completed_epochs,
        has_checkpoint=has_checkpoint,
        sota_reached=sota["sota_reached"],
        ladder_passed=progress.ladder_passed,
        data_fraction_passed=progress.data_fraction_passed,
    )
    return ModelWithStats.model_validate(
        {
            **_identity_fields(model_key, info),
            **_checkpoint_fields(model_ckpts, has_checkpoint),
            **_readiness_fields(model_key, hub_dir),
            **_ladder_fields(kind, ladder, stats, progress),
            **sota,
            **_training_defaults(model_key, info, kind.is_forex),
            "epochs_completed": completed_epochs,
            "is_fully_trained": verdict.is_fully_trained,
            "training_status": verdict.training_status,
        }
    )
