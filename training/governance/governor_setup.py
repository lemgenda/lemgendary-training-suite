"""Construction of the ``SmartTrainingGovernor`` state groups from model and config dicts.

Each builder reads one concern (curriculum ladder, thermal start, SOTA tracker, tuning
knobs) so ``SmartTrainingGovernor.__init__`` only wires the groups together.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch

from training.governance.curriculum import CurriculumState
from training.governance.governor_groups import (
    Calibration,
    GovernorRuntime,
    GovernorTuning,
    LockTuning,
    LossWeights,
    RankShaping,
)
from training.governance.metrics import DEFAULT_METRIC_DIRECTIONS, DEFAULT_METRIC_WEIGHTS
from training.governance.sota import SotaTracker
from training.governance.thermal import ThermalState

logger = logging.getLogger("lemtrain.governance.governor")

# Tasks whose resolution ladder is capped by the GPU's VRAM.
_VRAM_CAPPED_TASKS = ("restoration", "enhancement", "face", "parameter_prediction")

# Score contribution of a lower-is-better SOTA target, by metric; others use 1 / target.
_LOWER_IS_BETTER_SCORE: dict[str, Callable[[float], float]] = {
    "fid": lambda v: 100.0 - v,
    "lpips": lambda v: 1.0 - v,
    "rank_margin": lambda v: 10.0 - v,
    "max_drawdown": lambda v: max(0.0, 100.0 - v),
    "tp_mae": lambda v: max(0.0, 50.0 - v),
    "sl_mae": lambda v: max(0.0, 50.0 - v),
    "dir_entropy": lambda v: max(0.0, 1.099 - v),
}


def resolve_task_type(model_info: dict[str, Any]) -> str:
    """Return the dataset type (first entry when it is a list)."""
    raw_task = model_info.get("dataset_type", "quality")
    return raw_task[0] if isinstance(raw_task, list) else str(raw_task)


def _trains_on_full_data(model_info: dict[str, Any]) -> bool:
    """Return True for forex and LLM models, which always train on the whole dataset."""
    is_forex_model = (
        model_info.get("category") == "forex"
        or model_info.get("task_type") == "forex"
        or model_info.get("dataset_type") == "forex"
        or model_info.get("filename") == "ForexPredictor"
    )
    is_llm_model = (
        model_info.get("category") == "llm"
        or model_info.get("task_type") in ("text_instruction", "llm")
        or model_info.get("dataset_type") in ("text_instruction", "llm")
    )
    return is_forex_model or is_llm_model


def _initial_resolution(model_info: dict[str, Any]) -> int:
    """Return the configured input size (height of a pair), 224 when unusable."""
    raw_size = model_info.get("input_size") or 224
    current_res = raw_size[1] if isinstance(raw_size, (list, tuple)) else raw_size
    try:
        return int(current_res) if current_res is not None else 224
    except (ValueError, TypeError):
        return 224


def _stride_ladder(start_res: int, manifold_defaults: dict[str, Any]) -> list[int]:
    """Build a ladder from ``start_res`` up to the manifold's max resolution."""
    stride = manifold_defaults.get("resolution_stride", 128)
    max_res = manifold_defaults.get("max_resolution", 1024)
    ladder: list[int] = []
    curr = start_res
    while curr <= max_res:
        ladder.append(curr)
        curr += stride
    return ladder or [start_res]


def _vram_safe_resolution(task_type: str, config: dict[str, Any]) -> tuple[float, int]:
    """Return ``(VRAM in GB, highest safe resolution)`` for a VRAM-capped task."""
    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if torch.cuda.is_available()
        else 8.0
    )
    if vram_gb < 4.5:
        max_safe_res = 256 if task_type == "parameter_prediction" else 384
    elif vram_gb < 8.5:
        max_safe_res = 384
    elif vram_gb < 16.5:
        max_safe_res = 512
    else:
        max_safe_res = 1024
    cfg_max_res = config.get("hardware", {}).get("max_allowed_resolution")
    if cfg_max_res:
        max_safe_res = min(max_safe_res, int(cfg_max_res))
    return vram_gb, max_safe_res


def _apply_vram_cap(curriculum: CurriculumState, task_type: str, config: dict[str, Any]) -> None:
    """Drop ladder rungs above the VRAM-safe resolution and clamp the start resolution."""
    vram_gb, max_safe_res = _vram_safe_resolution(task_type, config)
    curriculum.res_ladder = [r for r in curriculum.res_ladder if r <= max_safe_res]
    if not curriculum.res_ladder:
        curriculum.res_ladder = [max_safe_res]
    if curriculum.current_res is not None and curriculum.current_res > max_safe_res:
        logger.info(
            "Hardware VRAM Cap Active (%.1fGB): Clamping %dpx -> %dpx for memory stability.",
            vram_gb,
            curriculum.current_res,
            max_safe_res,
        )
        curriculum.current_res = max_safe_res


def _align_start_rung(curriculum: CurriculumState, task_type: str) -> None:
    """Start on the lowest rung (except forex) and keep the start resolution on the ladder."""
    if (
        curriculum.res_ladder
        and curriculum.current_res != curriculum.res_ladder[0]
        and task_type != "forex"
    ):
        logger.info(
            "Aligning start resolution %dpx -> lowest rung %dpx.",
            curriculum.current_res,
            curriculum.res_ladder[0],
        )
        curriculum.current_res = curriculum.res_ladder[0]

    if curriculum.current_res is not None and curriculum.current_res not in curriculum.res_ladder:
        curriculum.res_ladder = sorted(set(curriculum.res_ladder + [curriculum.current_res]))


def build_curriculum(
    model_info: dict[str, Any], config: dict[str, Any], task_type: str
) -> CurriculumState:
    """Build the starting rung, batch and resolution ladder of the curriculum."""
    opt = model_info.get("optimization", {})
    manifold_defaults = config.get("governor", {}).get("manifold", {})
    curriculum = CurriculumState()
    curriculum.target_effective_batch = opt.get(
        "target_effective_batch",
        manifold_defaults.get("target_effective_batch", 24),
    )
    curriculum.fraction_increment = opt.get(
        "fraction_increment",
        manifold_defaults.get("fraction_increment", 0.20),
    )
    if _trains_on_full_data(model_info):
        initial_fraction = 1.0
    else:
        initial_fraction = model_info.get("sample_fraction") or opt.get(
            "initial_fraction",
            manifold_defaults.get("initial_fraction", 0.15),
        )
    curriculum.current_fraction = float(initial_fraction)

    batch_val = model_info.get("batch_size", 16)
    curriculum.current_batch = int(batch_val) if batch_val and batch_val != "auto" else 16
    curriculum.current_acc = 1
    curriculum.current_res = _initial_resolution(model_info)

    res_ladder = opt.get("res_ladder")
    if not res_ladder:
        curriculum.res_ladder = _stride_ladder(curriculum.current_res, manifold_defaults)
    else:
        curriculum.res_ladder = list(res_ladder)

    if task_type in _VRAM_CAPPED_TASKS:
        _apply_vram_cap(curriculum, task_type, config)
    _align_start_rung(curriculum, task_type)
    return curriculum


def build_tuning(
    model_info: dict[str, Any], config: dict[str, Any], task_type: str
) -> GovernorTuning:
    """Read the tuning knobs from the optimization block and the governor config."""
    opt = model_info.get("optimization", {})
    gov_cfg = config.get("governor", {})
    base_delta = opt.get("min_delta", 0.0005)
    if task_type == "parameter_prediction":
        min_delta = base_delta
    else:
        min_delta = base_delta if task_type == "quality" else (base_delta * 100.0)
    return GovernorTuning(
        manifold_maturity=opt.get(
            "manifold_maturity",
            gov_cfg.get("manifold", {}).get("maturity_soak", 3),
        ),
        plateau_priority=opt.get("plateau_priority", "data"),
        cooling_factor=opt.get("cooling_factor", 0.5),
        clamp_range=opt.get("clamp_range", [15.0, 45.0]),
        sharpening_rate=gov_cfg.get("sharpening_cooling_rate", 0.98),
        min_delta=min_delta,
        locks=LockTuning(
            jolt_cooldown=gov_cfg.get("jolt_cooldown_epochs", 5),
            stabilization_lock=gov_cfg.get("stabilization_lock_epochs", 3),
            breakout_threshold=gov_cfg.get("emergency_breakout_threshold", 0.10),
        ),
    )


def build_thermal(stab: dict[str, Any], task_type: str, clamp_range: list[float]) -> ThermalState:
    """Return the thermal state with the task's starting and minimum temperature."""
    if task_type == "forex":
        min_temp = max(0.75, float(stab.get("min_temp", 0.75)))
        temperature = max(min_temp, float(stab.get("softmax_temp", 1.0)))
    else:
        default_min = (
            0.5
            if task_type == "quality"
            else (0.01 if task_type == "parameter_prediction" else 0.1)
        )
        min_temp = float(stab.get("min_temp", default_min))
        temperature = float(stab.get("softmax_temp", min_temp))
    thermal = ThermalState()
    thermal.clamp_range = list(clamp_range)
    thermal.temperature = temperature
    thermal.min_temperature = min_temp
    return thermal


def build_runtime(stab: dict[str, Any], min_temp: float) -> GovernorRuntime:
    """Return the per-epoch state with the stabilizer block's loss shaping."""
    return GovernorRuntime(
        rank=RankShaping.from_stabilizers(stab),
        weights=LossWeights.from_stabilizers(stab),
        calibration=Calibration(
            min_temp=min_temp,
            current_clamp=float(stab.get("logit_clamp", 15.0)),
        ),
    )


def target_quality_score(sota_targets: dict[str, float], task_type: str) -> float:
    """Combine the SOTA targets into the quality score the curriculum aims for."""
    target_score = 0.0
    for k, target_v in sota_targets.items():
        weight = DEFAULT_METRIC_WEIGHTS.get(k, 1.0)
        if DEFAULT_METRIC_DIRECTIONS.get(k, True):
            target_score += target_v * weight
        elif k in _LOWER_IS_BETTER_SCORE:
            target_score += _LOWER_IS_BETTER_SCORE[k](target_v) * weight
        else:
            target_score += (1.0 / (target_v + 1e-6)) * weight
    if target_score <= 0:
        return 1.0
    if target_score <= 1.0 and task_type == "quality":
        target_score *= 100.0
    return target_score


def build_sota(
    model_info: dict[str, Any],
    config: dict[str, Any],
    task_type: str,
    sota_targets: dict[str, float],
) -> SotaTracker:
    """Return the SOTA tracker with the loop-breaker policy and the target quality score."""
    opt = model_info.get("optimization", {})
    sota = SotaTracker()
    sota.plateau_patience = opt.get(
        "plateau_patience",
        config.get("governor", {}).get("manifold", {}).get("plateau_patience", 6),
    )
    sota.task_type = task_type
    if task_type == "forex":
        sota.plateau_patience = max(sota.plateau_patience, 15)
    sota.loop_breaker_enabled = opt.get("loop_breaker_enabled", True)
    sota.loop_breaker_threshold = opt.get("loop_breaker_threshold", 2)
    sota.loop_breaker_strategy = opt.get("loop_breaker_strategy", "auto")
    sota.sota_targets = sota_targets
    sota.target_quality_score = target_quality_score(sota_targets, task_type)
    return sota
