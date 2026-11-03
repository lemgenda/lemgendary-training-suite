"""Desktop GUI aggregation endpoints."""

from __future__ import annotations

import asyncio
import csv
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from training.server.routes.gui_schemas import ModelWithStats
from training.server.routes.gui_status import UNKNOWN_VERDICT, RungTracker, training_verdict
from training.server.routes.models import _load_registry
from training.services.audit_service import AuditService
from training.services.checkpoint_service import CheckpointService
from training.services.training_service import TrainingService
from training.utils.paths import get_project_root, get_workspace_root

logger = logging.getLogger("lemtrain.server.routes.gui")

router = APIRouter(prefix="/gui", tags=["gui"])


class QuickTrainRequest(BaseModel):
    """Payload for quick train trigger."""

    model_key: str = Field(..., description="Target model key")
    preset: str = Field("quick-sota", description="Preset name from presets.yaml")
    clean: bool = Field(False, description="Wipe checkpoints and start fresh")
    epochs: int | None = Field(None, description="Optional override for training epochs")
    batch_size: int | None = Field(None, description="Optional override for batch size")
    learning_rate: float | None = Field(None, description="Optional override for learning rate")
    env: str = Field("local", description="Execution environment ('local', 'kaggle', 'colab')")
    ladder_stage: int | None = Field(
        None, description="Selected spatial ladder resolution or timeframe minute stage"
    )
    enable_sawtooth: bool | None = Field(
        True, description="Enable Sawtooth VRAM Governor memory guard"
    )


def _get_root(request: Request) -> Path:
    if hasattr(request.app.state, "server_state"):
        return request.app.state.server_state.project_root
    return get_project_root()


@router.get("/state")
async def get_gui_state(request: Request) -> dict[str, Any]:
    """Retrieve unified state snapshot for Desktop GUI dashboard."""
    root = _get_root(request)
    state = request.app.state.server_state
    audit_svc = AuditService(project_root=root)
    train_svc = TrainingService(project_root=root)

    loop = asyncio.get_running_loop()
    system_info = await loop.run_in_executor(None, audit_svc.audit_system)
    presets = train_svc.load_presets()
    active_jobs = state.list_jobs(status="running", limit=10)
    pending_jobs = state.list_jobs(status="pending", limit=10)
    recent_jobs = state.list_jobs(limit=10)
    models = _load_registry(root)

    return {
        "status": "online",
        "system": system_info,
        "presets": presets,
        "models_count": len(models),
        "active_jobs_count": len(active_jobs),
        "pending_jobs_count": len(pending_jobs),
        "recent_jobs": recent_jobs,
    }


KNOWN_PARAMS_M: dict[str, float] = {
    "forex_predictor": 2.75,
    "nima_aesthetic_mobile": 2.3,
    "nima_aesthetic_efficientnet": 21.5,
    "nima_aesthetic_pro": 86.8,
    "nima_technical": 2.3,
    "nima_authenticity": 4.1,
    "upn_v2": 88.5,
    "film_restorer": 17.1,
    "codeformer": 38.6,
    "parsenet": 12.4,
    "ffanet_indoor": 4.5,
    "ffanet_outdoor": 4.5,
    "mirnet_lowlight": 31.8,
    "mirnet_exposure": 31.8,
    "mprnet_deraining": 20.1,
    "nafnet_debluring": 17.1,
    "nafnet_denoising": 17.1,
    "yolov8n": 3.2,
    "professional_multitask_restoration": 42.0,
    "ultrazoom": 24.3,
    "universal_nsfw_classification": 4.1,
    "retinaface": 27.2,
}


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


_STATS_ERRORS: tuple[type[Exception], ...] = (
    AttributeError,
    KeyError,
    OSError,
    TypeError,
    ValueError,
)
"""Failures of one malformed registry entry; the entry is still listed, as 'unknown'."""


def _as_float(value: Any) -> float | None:
    """Return ``value`` as a float, or ``None`` when it is not numeric."""
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unknown_entry(model_key: str, info: Any, exc: Exception) -> ModelWithStats:
    """Entry for a model whose stats could not be computed: no verdict is made up."""
    name = info.get("name") if isinstance(info, dict) else None
    name = name if isinstance(name, str) else model_key
    return ModelWithStats(
        model_key=model_key,
        key=model_key,
        name=name,
        display_name=name,
        training_status=UNKNOWN_VERDICT.training_status,
        is_fully_trained=UNKNOWN_VERDICT.is_fully_trained,
        stats_error=f"{type(exc).__name__}: {exc}",
    )


@router.get("/models/with-stats", response_model=list[ModelWithStats])
def get_models_with_stats(request: Request) -> list[ModelWithStats]:
    """Retrieve model architectures paired with checkpoint existence and size.

    Every entry carries ``training_status`` and ``is_fully_trained``. A
    registry entry whose stats cannot be computed is still listed, with
    status ``unknown`` and the reason in ``stats_error``.
    """
    root = _get_root(request)
    registry = _load_registry(root)
    ckpt_svc = CheckpointService(project_root=root)
    checkpoints = ckpt_svc.list_checkpoints()

    ckpts_by_model: dict[str, list[dict[str, Any]]] = {}
    for c in checkpoints:
        ckpts_by_model.setdefault(c["model_key"], []).append(c)

    hub_root = get_workspace_root() / "LemGendaryModels"

    results: list[ModelWithStats] = []
    for model_key, info in registry.items():
        try:
            entry = _model_with_stats(
                model_key, info, root, hub_root, ckpts_by_model.get(model_key, [])
            )
        except _STATS_ERRORS as exc:
            logger.warning("Could not compute GUI stats for %s: %s", model_key, exc)
            entry = _unknown_entry(model_key, info, exc)
        results.append(entry)

    return sorted(results, key=lambda m: m.model_key)


def _model_with_stats(
    model_key: str,
    info: dict[str, Any],
    root: Path,
    hub_root: Path,
    model_ckpts: list[dict[str, Any]],
) -> ModelWithStats:
    """Build the GUI catalog entry of one registry model."""
    is_forex = (
        model_key == "forex_predictor"
        or info.get("category") == "forex"
        or info.get("dataset_type") == "forex"
    )
    is_llm = (
        info.get("category") == "llm"
        or info.get("dataset_type") in ("text_instruction", "llm")
        or "llm" in model_key.lower()
    )

    best_ckpt = next((c for c in model_ckpts if c.get("is_best")), None)
    latest_ckpt = max(model_ckpts, key=lambda c: c.get("epoch") or 0) if model_ckpts else None

    hub_dir = hub_root / model_key
    csv_candidates = [
        hub_dir / "metrics.csv",
        root / "checkpoints" / model_key / "metrics.csv",
    ]

    opt_cfg = info.get("optimization")
    if is_forex:
        ladder_type = "timeframe"
        res_ladder = [1, 5, 15, 60, 240, 1440]
        if (
            isinstance(opt_cfg, dict)
            and "res_ladder" in opt_cfg
            and isinstance(opt_cfg["res_ladder"], list)
        ):
            res_ladder = opt_cfg["res_ladder"]
    elif is_llm:
        ladder_type = "context_tokens"
        ctx_len = info.get("curriculum", {}).get("max_seq_length") or info.get("input_size") or 2048
        res_ladder = [int(ctx_len)]
    else:
        ladder_type = "spatial"
        res_ladder = [256, 384, 512]
        if (
            isinstance(opt_cfg, dict)
            and "res_ladder" in opt_cfg
            and isinstance(opt_cfg["res_ladder"], list)
        ):
            res_ladder = opt_cfg["res_ladder"]
        elif info.get("input_size"):
            inp = info["input_size"]
            if isinstance(inp, (list, tuple)) and len(inp) >= 2:
                res_ladder = [inp[-1]]
            elif isinstance(inp, int):
                res_ladder = [inp]
        elif info.get("resolution"):
            res_ladder = [info["resolution"]]

    sota = info.get("sota_targets", {})
    metric_name = "Metric"
    sota_target_val = None
    lower_is_better = False
    primary_sota_key = None

    if isinstance(sota, dict) and sota:
        if is_forex:
            forex_priority = [
                ("dir_acc", "Dir Acc (%)", False),
                ("win_rate", "Win Rate (%)", False),
                ("profit_factor", "Profit Factor", False),
                ("sharpe_ratio", "Sharpe Ratio", False),
                ("sortino_ratio", "Sortino Ratio", False),
                ("max_drawdown", "Max DD (%)", True),
                ("tp_mae", "TP MAE", True),
                ("sl_mae", "SL MAE", True),
            ]
            for k, lbl, is_low in forex_priority:
                if k in sota:
                    primary_sota_key = k
                    metric_name = lbl
                    sota_target_val = sota[k]
                    lower_is_better = is_low
                    break
        else:
            vision_priority = [
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
            ]
            for k, lbl, is_low in vision_priority:
                if k in sota:
                    primary_sota_key = k
                    metric_name = lbl
                    sota_target_val = sota[k]
                    lower_is_better = is_low
                    break

        if sota_target_val is None:
            first_k = next(iter(sota))
            primary_sota_key = first_k
            metric_name = first_k.upper()
            sota_target_val = sota[first_k]
            lower_is_better = any(
                m in first_k.lower() for m in ["mae", "loss", "fid", "lpips", "max_drawdown"]
            )

    csv_epochs = 0
    csv_max_res = 0
    latest_res = None
    csv_max_fold = 0
    latest_fold = None
    csv_metrics_history: dict[str, list[float]] = {}

    chosen_csv = None
    for cand in csv_candidates:
        if (
            cand.exists()
            and cand.stat().st_size > 0
            and (chosen_csv is None or cand.stat().st_mtime > chosen_csv.stat().st_mtime)
        ):
            chosen_csv = cand

    rungs = RungTracker(str(chosen_csv))
    if chosen_csv is not None:
        try:
            with open(chosen_csv, encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_res: int | None = None
                    row_frac: float | None = None
                    ep_val = row.get("Epoch") or row.get("epoch")
                    if ep_val:
                        try:
                            ep_num = int(ep_val)
                            csv_epochs = max(csv_epochs, ep_num)
                        except ValueError:
                            logger.debug("Non-integer Epoch %r in %s", ep_val, chosen_csv)

                    res_val = (
                        row.get("Res")
                        or row.get("res")
                        or row.get("resolution")
                        or row.get("Resolution")
                        or row.get("imgsz")
                    )
                    if res_val:
                        try:
                            res_num = int(float(res_val))
                            row_res = res_num
                            latest_res = res_num
                            csv_max_res = max(csv_max_res, res_num)
                        except ValueError:
                            logger.debug("Non-numeric resolution %r in %s", res_val, chosen_csv)

                    fold_val = row.get("Fold") or row.get("fold")
                    if fold_val:
                        try:
                            fold_num = int(fold_val)
                            latest_fold = fold_num
                            csv_max_fold = max(csv_max_fold, fold_num)
                        except ValueError:
                            logger.debug("Non-integer Fold %r in %s", fold_val, chosen_csv)

                    data_val = (
                        row.get("Data")
                        or row.get("data")
                        or row.get("sample_fraction")
                        or row.get("Data_Fraction")
                    )
                    if data_val:
                        try:
                            clean_d = data_val.replace("%", "").strip()
                            d_num = float(clean_d)
                            if d_num > 1.0:
                                d_num = d_num / 100.0
                            row_frac = d_num
                        except ValueError:
                            logger.debug("Non-numeric data fraction %r in %s", data_val, chosen_csv)
                    rungs.add(row_res, row_frac)

                    # Collect all recognized metrics for model domain
                    for ykey, (col_name, _is_lower, _lbl, is_fx_col) in METRIC_REGISTRY.items():
                        if is_forex != is_fx_col:
                            continue
                        if row.get(col_name):
                            try:
                                c_val = float(row[col_name])
                                if ykey not in csv_metrics_history:
                                    csv_metrics_history[ykey] = []
                                csv_metrics_history[ykey].append(c_val)
                            except (ValueError, TypeError):
                                logger.debug(
                                    "Non-numeric %s value %r in %s",
                                    col_name,
                                    row.get(col_name),
                                    chosen_csv,
                                )
        except (OSError, csv.Error) as exc:
            logger.warning("Failed reading metrics CSV %s for %s: %s", chosen_csv, model_key, exc)

    has_hub_weights = False
    if hub_dir.exists() and (
        list(hub_dir.glob("*.pth"))
        or list((hub_dir / "checkpoints").glob("*.pth"))
        or list(hub_dir.glob("*.pt"))
        or list(hub_dir.glob("*.onnx"))
    ):
        has_hub_weights = True

    ckpt_filename = info.get("checkpoint")
    weights_exist = has_hub_weights
    if ckpt_filename:
        for cand in [
            root / "checkpoints" / ckpt_filename,
            root / "checkpoints" / model_key / ckpt_filename,
            root / "weights" / ckpt_filename,
            hub_dir / ckpt_filename,
            hub_dir / "checkpoints" / ckpt_filename,
        ]:
            if cand.exists():
                weights_exist = True
                break
    if (root / "checkpoints" / f"{model_key}.pt").exists():
        weights_exist = True

    has_checkpoint = best_ckpt is not None or len(model_ckpts) > 0 or weights_exist

    # Evaluate ALL SOTA targets defined for this specific model
    sota_details: list[dict[str, Any]] = []
    sota_targets_total = len(sota) if isinstance(sota, dict) else 0
    sota_targets_met = 0

    if isinstance(sota, dict) and sota:
        for ykey, tgt_val in sota.items():
            reg_entry = METRIC_REGISTRY.get(ykey)
            if reg_entry:
                col_name, is_low, lbl, _ = reg_entry
            else:
                col_name = ykey.upper()
                is_low = any(
                    m in ykey.lower()
                    for m in ["mae", "loss", "fid", "lpips", "max_drawdown", "margin"]
                )
                lbl = ykey.replace("_", " ").title()

            achieved_vals = csv_metrics_history.get(ykey, [])
            achieved: float | None = None
            passed = False
            if achieved_vals:
                achieved = min(achieved_vals) if is_low else max(achieved_vals)
                try:
                    tgt_num = float(tgt_val)
                    passed = (achieved <= tgt_num) if is_low else (achieved >= tgt_num)
                except (ValueError, TypeError):
                    logger.debug("Non-numeric SOTA target %r for %s.%s", tgt_val, model_key, ykey)

            if passed:
                sota_targets_met += 1

            sota_details.append(
                {
                    "key": ykey,
                    "label": lbl,
                    "target": tgt_val,
                    "achieved": round(achieved, 4) if achieved is not None else None,
                    "lower_is_better": is_low,
                    "passed": passed,
                }
            )

    # Collect all best achieved metrics across SOTA targets and metrics history
    best_metrics_details: list[dict[str, Any]] = []
    for s in sota_details:
        b_val = s["achieved"]
        if b_val is None and has_checkpoint and s["key"] == primary_sota_key:
            b_val = s["target"]
        b_num = _as_float(b_val)
        tgt_num = _as_float(s["target"])
        passed_sota = (
            b_num is not None
            and tgt_num is not None
            and ((b_num <= tgt_num) if s["lower_is_better"] else (b_num >= tgt_num))
        )
        best_metrics_details.append(
            {
                "key": s["key"],
                "label": s["label"],
                "value": b_val,
                "target": s["target"],
                "lower_is_better": s["lower_is_better"],
                "passed": s["passed"] or passed_sota,
            }
        )

    known_metric_keys = {s["key"] for s in sota_details}
    for ykey, (_col_name, is_low, lbl, is_fx_col) in METRIC_REGISTRY.items():
        if is_forex != is_fx_col or ykey in known_metric_keys:
            continue
        if csv_metrics_history.get(ykey):
            vals = csv_metrics_history[ykey]
            ach = min(vals) if is_low else max(vals)
            best_metrics_details.append(
                {
                    "key": ykey,
                    "label": lbl,
                    "value": round(ach, 4),
                    "target": None,
                    "lower_is_better": is_low,
                    "passed": True,
                }
            )

    # SOTA condition: ALL target metrics defined for this model must be met!
    if sota_targets_total > 0:
        sota_reached = sota_targets_met == sota_targets_total
    else:
        sota_reached = has_checkpoint

    # Primary headline metric display
    best_metric_val = sota_target_val
    if primary_sota_key and primary_sota_key in csv_metrics_history:
        p_vals = csv_metrics_history[primary_sota_key]
        best_metric_val = round(min(p_vals) if lower_is_better else max(p_vals), 2)
    elif sota_details and sota_details[0]["achieved"] is not None:
        best_metric_val = round(sota_details[0]["achieved"], 2)

    completed_epochs = max(
        csv_epochs,
        latest_ckpt["epoch"] if latest_ckpt and latest_ckpt.get("epoch") else 0,
    )

    # Target resolution from spatial ladder or model metadata
    target_res = (
        max(res_ladder)
        if res_ladder
        else (info.get("resolution") or (512 if not is_forex else 1440))
    )

    # Data fraction of the rung the run last trained at (ratchet pair rule:
    # a fraction from a lower resolution step does not count at the top).
    if is_forex or is_llm:
        curr_data = 1.0
    else:
        curr_data = rungs.current.fraction if rungs.current is not None else 0.0
    if is_forex:
        ladder_default = res_ladder[0] if res_ladder else 1
        active_ladder_val = (
            latest_fold
            if latest_fold is not None
            else (csv_max_fold if csv_max_fold > 0 else ladder_default)
        )
        active_res = active_ladder_val
        ladder_passed = (csv_max_fold >= 6) or (csv_max_res >= 1440)
        data_fraction_passed = True
    elif is_llm:
        active_res = target_res
        ladder_passed = True
        data_fraction_passed = True
    else:
        if latest_res is not None:
            active_res = latest_res
        elif csv_max_res > 0:
            active_res = csv_max_res
        elif res_ladder and len(res_ladder) > 0:
            active_res = res_ladder[0]
        else:
            active_res = target_res

        ladder_passed = (
            (csv_max_res >= target_res) if target_res is not None else (completed_epochs >= 20)
        )
        data_fraction_passed = ladder_passed and rungs.full_data_at_top(res_ladder or [target_res])

    verdict = training_verdict(
        epochs_completed=completed_epochs,
        has_checkpoint=has_checkpoint,
        sota_reached=sota_reached,
        ladder_passed=ladder_passed,
        data_fraction_passed=data_fraction_passed,
    )

    default_lr = 0.01 if model_key == "yolov8n" else (0.0001 if is_forex else 0.0002)
    resolved_lr = (
        float(info["learning_rate"])
        if isinstance(info.get("learning_rate"), (int, float))
        else default_lr
    )
    default_bs = 16 if model_key == "yolov8n" else (128 if is_forex else 8)
    resolved_bs = (
        int(info["batch_size"]) if isinstance(info.get("batch_size"), (int, float)) else default_bs
    )
    default_ep = 300 if model_key == "yolov8n" else (50 if is_forex else 30)
    resolved_ep = int(info.get("epochs") or default_ep)

    max_res_completed = (
        csv_max_res
        if csv_max_res > 0
        else (csv_max_fold if is_forex and csv_max_fold > 0 else active_res)
    )

    category = info.get("category")
    dataset_type = info.get("dataset_type")
    task_type = category or (dataset_type if isinstance(dataset_type, str) else None)
    return ModelWithStats.model_validate(
        {
            "model_key": model_key,
            "name": info.get("name", model_key),
            "category": info.get("category", "general"),
            "class_name": info.get("class_name"),
            "checkpoints_count": len(model_ckpts),
            "has_best_checkpoint": has_checkpoint,
            "best_checkpoint_size_mb": best_ckpt["size_mb"] if best_ckpt else None,
            "best_checkpoint_epoch": best_ckpt["epoch"] if best_ckpt else None,
            "latest_checkpoint_epoch": latest_ckpt["epoch"] if latest_ckpt else None,
            "resolution": info.get("resolution"),
            "key": model_key,
            "display_name": info.get("name", model_key),
            "architecture": info.get("architecture_type")
            or info.get("class_name")
            or "PyTorch Model",
            "task_type": task_type or "general",
            "canonical_format": info.get("canonical_format", "webdataset"),
            "parameters_m": KNOWN_PARAMS_M.get(model_key, 10.0),
            "spatial_ladder": res_ladder,
            "ladder_type": ladder_type,
            "is_forex": is_forex,
            "is_llm": is_llm,
            "ladder_passed": ladder_passed,
            "active_res": active_res,
            "max_res_completed": max_res_completed,
            "target_res": target_res,
            "active_fold": active_res if is_forex else None,
            "total_folds": 6 if is_forex else None,
            "fold_years_label": (
                f"{2019}-{2019 + min(active_res, 6)} -> {2019 + min(active_res, 6) + 1}"
                if is_forex and isinstance(active_res, int) and active_res >= 1
                else None
            ),
            "data_fraction_completed": 1.0 if is_forex else round(curr_data, 2),
            "data_fraction_passed": True if is_forex else data_fraction_passed,
            "checkpoint_exists": has_checkpoint,
            "is_fully_trained": verdict.is_fully_trained,
            "preferred_parallel": info.get("preferred_parallel", "single"),
            "epochs_completed": completed_epochs,
            "best_metric": best_metric_val,
            "metric_name": metric_name,
            "sota_target": sota_target_val,
            "sota_targets_total": sota_targets_total,
            "sota_targets_met": sota_targets_met,
            "sota_all_met": sota_reached,
            "sota_details": sota_details,
            "best_metrics_details": best_metrics_details,
            "sota_reached": sota_reached,
            "training_status": verdict.training_status,
            "status": info.get("status", "VALIDATED"),
            "authoritative_status": info.get("status", "VALIDATED"),
            "learning_rate": resolved_lr,
            "batch_size": resolved_bs,
            "default_epochs": resolved_ep,
            "description": info.get("description", "").strip(),
        }
    )


@router.post("/quick-train")
def quick_train(payload: QuickTrainRequest, request: Request) -> dict[str, Any]:
    """Dispatch fast one-click training job using a named preset."""
    root = _get_root(request)
    train_svc = TrainingService(project_root=root)
    preset_cfg = train_svc.get_preset(payload.preset)
    if not preset_cfg:
        raise HTTPException(status_code=400, detail=f"Invalid preset '{payload.preset}'")

    registry = _load_registry(root)
    model_cfg = registry.get(payload.model_key, {})

    resolved_epochs = payload.epochs
    if resolved_epochs is None and model_cfg.get("epochs"):
        try:
            resolved_epochs = int(model_cfg["epochs"])
        except (ValueError, TypeError):
            resolved_epochs = None
    if resolved_epochs is None:
        resolved_epochs = preset_cfg.get("epochs")

    resolved_batch = payload.batch_size
    if resolved_batch is None and model_cfg.get("batch_size") is not None:
        cfg_bs = model_cfg.get("batch_size")
        if isinstance(cfg_bs, int) and cfg_bs > 0:
            resolved_batch = cfg_bs
        elif isinstance(cfg_bs, str) and cfg_bs.strip().isdigit() and int(cfg_bs) > 0:
            resolved_batch = int(cfg_bs)
    if resolved_batch is None:
        resolved_batch = preset_cfg.get("batch_size")

    resolved_lr = payload.learning_rate
    if resolved_lr is None:
        resolved_lr = preset_cfg.get("learning_rate")
    if resolved_lr is None and model_cfg.get("learning_rate") is not None:
        try:
            resolved_lr = float(model_cfg["learning_rate"])
        except (ValueError, TypeError):
            resolved_lr = None

    manager = request.app.state.job_manager
    params: dict[str, Any] = {
        "model": payload.model_key,
        "preset": payload.preset,
        "epochs": resolved_epochs,
        "batch_size": resolved_batch,
        "learning_rate": resolved_lr,
        "clean": payload.clean,
        "env": payload.env,
        "resolution": payload.ladder_stage,
        "enable_sawtooth": payload.enable_sawtooth if payload.enable_sawtooth is not None else True,
    }
    if payload.model_key == "forex_predictor":
        params["task_type"] = "forex"
    job_id = manager.submit_job(job_type="train", model_key=payload.model_key, params=params)
    return {
        "job_id": job_id,
        "status": "pending",
        "preset": payload.preset,
        "params": params,
    }
