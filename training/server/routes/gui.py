"""Desktop GUI aggregation endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from training.server.routes.models import _load_registry
from training.services.audit_service import AuditService
from training.services.checkpoint_service import CheckpointService
from training.services.training_service import TrainingService
from training.utils.paths import get_project_root

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


def _get_root(request: Request) -> Path:
    if hasattr(request.app.state, "server_state"):
        return request.app.state.server_state.project_root
    return get_project_root()


@router.get("/state")
def get_gui_state(request: Request) -> dict[str, Any]:
    """Retrieve unified state snapshot for Desktop GUI dashboard."""
    root = _get_root(request)
    state = request.app.state.server_state
    audit_svc = AuditService(project_root=root)
    train_svc = TrainingService(project_root=root)

    system_info = audit_svc.audit_system()
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
}


@router.get("/models/with-stats")
def get_models_with_stats(request: Request) -> list[dict[str, Any]]:
    """Retrieve model architectures paired with checkpoint existence and size."""
    root = _get_root(request)
    registry = _load_registry(root)
    ckpt_svc = CheckpointService(project_root=root)
    checkpoints = ckpt_svc.list_checkpoints()

    ckpts_by_model: dict[str, list[dict[str, Any]]] = {}
    for c in checkpoints:
        k = c["model_key"]
        if k not in ckpts_by_model:
            ckpts_by_model[k] = []
        ckpts_by_model[k].append(c)

    hub_root = (root / ".." / "LemGendaryModels").resolve()

    results: list[dict[str, Any]] = []
    for model_key, info in registry.items():
        is_forex = (
            model_key == "forex_predictor"
            or info.get("category") == "forex"
            or info.get("dataset_type") == "forex"
        )

        model_ckpts = ckpts_by_model.get(model_key, [])
        best_ckpt = next((c for c in model_ckpts if c.get("is_best")), None)
        latest_ckpt = max(model_ckpts, key=lambda c: c.get("epoch") or 0) if model_ckpts else None

        hub_dir = hub_root / model_key
        csv_candidates = [
            hub_dir / "metrics.csv",
            root / "checkpoints" / model_key / "metrics.csv",
        ]
        csv_epochs = 0
        csv_max_res = 0
        csv_max_data = 0.0
        csv_max_fold = 0
        csv_metrics_history: dict[str, list[float]] = {}

        opt_cfg = info.get("optimization")
        if is_forex:
            ladder_type = "timeframe"
            res_ladder = [1, 5, 15, 60, 240, 1440]
            if isinstance(opt_cfg, dict) and "res_ladder" in opt_cfg and isinstance(opt_cfg["res_ladder"], list):
                res_ladder = opt_cfg["res_ladder"]
        else:
            ladder_type = "spatial"
            res_ladder = [256, 384, 512]
            if isinstance(opt_cfg, dict) and "res_ladder" in opt_cfg and isinstance(opt_cfg["res_ladder"], list):
                res_ladder = opt_cfg["res_ladder"]
            elif info.get("input_size") and len(info["input_size"]) >= 2:
                res_ladder = [info["input_size"][-1]]
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
                lower_is_better = any(m in first_k.lower() for m in ["mae", "loss", "fid", "lpips", "max_drawdown"])

        for csv_path in csv_candidates:
            if csv_path.exists():
                try:
                    import csv
                    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            ep_val = row.get("Epoch") or row.get("epoch")
                            if ep_val:
                                try:
                                    ep_num = int(ep_val)
                                    if ep_num > csv_epochs:
                                        csv_epochs = ep_num
                                except ValueError:
                                    pass

                            res_val = row.get("Res") or row.get("res") or row.get("resolution")
                            if res_val:
                                try:
                                    res_num = int(float(res_val))
                                    if res_num > csv_max_res:
                                        csv_max_res = res_num
                                except ValueError:
                                    pass

                            fold_val = row.get("Fold") or row.get("fold")
                            if fold_val:
                                try:
                                    fold_num = int(fold_val)
                                    if fold_num > csv_max_fold:
                                        csv_max_fold = fold_num
                                except ValueError:
                                    pass

                            data_val = row.get("Data") or row.get("data") or row.get("sample_fraction")
                            if data_val:
                                try:
                                    clean_d = data_val.replace("%", "").strip()
                                    d_num = float(clean_d)
                                    if d_num > 1.0:
                                        d_num = d_num / 100.0
                                    if d_num > csv_max_data:
                                        csv_max_data = d_num
                                except ValueError:
                                    pass

                            # Collect all recognized metrics for model domain
                            for ykey, (col_name, is_lower, _lbl, is_fx_col) in METRIC_REGISTRY.items():
                                if is_forex != is_fx_col:
                                    continue
                                if col_name in row and row[col_name]:
                                    try:
                                        c_val = float(row[col_name])
                                        if ykey not in csv_metrics_history:
                                            csv_metrics_history[ykey] = []
                                        csv_metrics_history[ykey].append(c_val)
                                    except (ValueError, TypeError):
                                        pass
                except Exception:
                    pass

        has_hub_weights = False
        if hub_dir.exists():
            if (
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
                    is_low = any(m in ykey.lower() for m in ["mae", "loss", "fid", "lpips", "max_drawdown", "margin"])
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
                        pass

                if passed:
                    sota_targets_met += 1

                sota_details.append({
                    "key": ykey,
                    "label": lbl,
                    "target": tgt_val,
                    "achieved": round(achieved, 4) if achieved is not None else None,
                    "lower_is_better": is_low,
                    "passed": passed,
                })

        # SOTA condition: ALL target metrics defined for this model must be met!
        if sota_targets_total > 0:
            sota_reached = (sota_targets_met == sota_targets_total)
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

        # Evaluate ladder completion
        target_res = max(res_ladder) if res_ladder else None
        if is_forex:
            # Multi-timeframe confluence / walk-forward curriculum across 6 folds
            ladder_passed = (csv_max_fold >= 6) or (csv_max_res >= 1440)
        elif target_res is not None:
            ladder_passed = (csv_max_res >= target_res)
        else:
            ladder_passed = (completed_epochs > 0)

        # Evaluate 100% data fraction requirement
        data_fraction_passed = (csv_max_data >= 0.99)

        # Authoritative criteria:
        # A model is FULLY TRAINED if and only if:
        # 1. It has passed the entire resolution ladder (or confluence horizon)
        # 2. It has trained with 100% data fraction (Data >= 1.0)
        # 3. Model weights have reached or exceeded ALL SOTA values set in unified_models_v2.yaml
        is_fully_trained = sota_reached and ladder_passed and data_fraction_passed

        if is_fully_trained:
            training_status = "fully_trained"
        elif completed_epochs > 0:
            training_status = "partially_trained"
        elif has_checkpoint:
            training_status = "weights_ready"
        else:
            training_status = "initializing"

        results.append({
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
            "architecture": info.get("architecture_type") or info.get("class_name") or "PyTorch Model",
            "task_type": info.get("category") or info.get("dataset_type") or "general",
            "canonical_format": info.get("canonical_format", "webdataset"),
            "parameters_m": KNOWN_PARAMS_M.get(model_key, 10.0),
            "spatial_ladder": res_ladder,
            "ladder_type": ladder_type,
            "is_forex": is_forex,
            "ladder_passed": ladder_passed,
            "max_res_completed": csv_max_res if csv_max_res > 0 else (csv_max_fold if is_forex and csv_max_fold > 0 else None),
            "target_res": target_res,
            "data_fraction_completed": round(csv_max_data, 2) if csv_max_data > 0 else 0.0,
            "data_fraction_passed": data_fraction_passed,
            "checkpoint_exists": has_checkpoint,
            "preferred_parallel": info.get("preferred_parallel", "single"),
            "epochs_completed": completed_epochs,
            "best_metric": best_metric_val,
            "metric_name": metric_name,
            "sota_target": sota_target_val,
            "sota_targets_total": sota_targets_total,
            "sota_targets_met": sota_targets_met,
            "sota_all_met": sota_reached,
            "sota_details": sota_details,
            "sota_reached": sota_reached,
            "training_status": training_status,
            "description": info.get("description", "").strip(),
        })

    return sorted(results, key=lambda x: x["model_key"])


@router.post("/quick-train")
def quick_train(payload: QuickTrainRequest, request: Request) -> dict[str, Any]:
    """Dispatch fast one-click training job using a named preset."""
    root = _get_root(request)
    train_svc = TrainingService(project_root=root)
    preset_cfg = train_svc.get_preset(payload.preset)
    if not preset_cfg:
        raise HTTPException(status_code=400, detail=f"Invalid preset '{payload.preset}'")

    manager = request.app.state.job_manager
    params: dict[str, Any] = {
        "model": payload.model_key,
        "preset": payload.preset,
        "epochs": payload.epochs if payload.epochs is not None else preset_cfg.get("epochs"),
        "batch_size": payload.batch_size if payload.batch_size is not None else preset_cfg.get("batch_size"),
        "learning_rate": payload.learning_rate if payload.learning_rate is not None else preset_cfg.get("learning_rate"),
        "clean": payload.clean,
        "env": payload.env,
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

