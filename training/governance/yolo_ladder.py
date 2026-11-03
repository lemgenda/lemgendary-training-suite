"""YOLOv8n spatial-resolution ladder planning.

AMP support detection, VRAM-safe batch sizing, the Sawtooth batch governor and the
rung curriculum (resolution x data-fraction stages) used by ``YOLOCurriculumGovernor``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger("lemtrain.governance.yolo")


@dataclass
class YOLOLadderStage:
    """Configuration for an individual curriculum resolution ladder stage."""

    stage_index: int
    resolution: int
    batch_size: int
    fraction: float
    target_epochs: int
    patience: int


def is_amp_supported_for_device(device_arg: str) -> bool:
    """Evaluate whether AMP FP16 is numerically stable on target GPU hardware.

    NVIDIA GTX 16xx series cards (e.g. GTX 1650, 1660, 1660 Ti with TU117/TU116 chips)
    lack Tensor Cores and have documented cuDNN FP16 gradient underflow issues that
    produce NaN losses during YOLO backward passes. In such environments, FP32
    is required for stability.
    """
    if not torch.cuda.is_available():
        return False

    try:
        if device_arg in ("cpu", "-1"):
            return False

        device_index = 0
        if device_arg.isdigit():
            device_index = int(device_arg)
        elif "," in device_arg:
            first_dev = device_arg.split(",", maxsplit=1)[0].strip()
            if first_dev.isdigit():
                device_index = int(first_dev)

        dev_name = torch.cuda.get_device_name(device_index).lower()
        if any(
            unsupported in dev_name for unsupported in ["gtx 1650", "gtx 1660", "gtx 10", "gtx 9"]
        ):
            return False

        major, minor = torch.cuda.get_device_capability(device_index)
        return not (major == 7 and minor == 5 and "rtx" not in dev_name)
    except (RuntimeError, ValueError) as exc:
        logger.debug("Failed to probe GPU device capability for AMP: %s", exc)
        return False


def compute_safe_batch_size(
    imgsz: int,
    vram_gb: float,
    requested_batch: int | None = None,
    opt_config: dict[str, Any] | None = None,
) -> int:
    """Calculate maximum safe physical batch size from first-principles VRAM arithmetic.

    Derives the safe batch size by estimating actual GPU memory consumption for
    YOLOv8n training at the requested spatial resolution, rather than using a
    hardcoded lookup table. All tuning constants are configurable via opt_config
    (sourced from unified_models_v2.yaml > optimization) with sensible defaults.

    VRAM budget model:
      total_budget_mb  = vram_gb * 1024 * vram_safety_factor
      available_mb     = total_budget_mb - static_overhead_mb
      per_sample_mb    = per_sample_vram_mb_640 * (imgsz / 640) ** 2
      max_safe_batch   = floor(available_mb / per_sample_mb)

    Empirical baseline (YOLOv8n):
      - static_overhead_mb  ~ 260 MB  (weights + gradients + AdamW states)
      - per_sample_vram_mb_640 ~ 480 MB  (activation maps at 640px, batch=1)

    Args:
        imgsz: Input spatial resolution in pixels (square assumed).
        vram_gb: Total physical GPU VRAM in gigabytes.
        requested_batch: Optional caller-requested batch size upper bound.
        opt_config: Model optimization config dict from unified_models_v2.yaml.

    Returns:
        Safe integer batch size, clamped to [1, requested_batch] if provided.
    """
    cfg = opt_config or {}

    # Constants configurable from unified_models_v2.yaml > optimization:
    #   static_vram_mb           - fixed VRAM cost regardless of batch size
    #                              (model weights + optimizer states + CUDA context)
    #   per_sample_vram_mb_640   - VRAM consumed per training sample at 640px
    #                              (scales quadratically with imgsz)
    #   sawtooth_vram_safety     - fraction of total VRAM to target (headroom reserve)
    #   batch_min / batch_max    - absolute clamps for safety and hardware alignment
    static_mb: float = float(cfg.get("static_vram_mb", 260.0))
    per_sample_mb_640: float = float(cfg.get("per_sample_vram_mb_640", 240.0))
    safety: float = float(cfg.get("sawtooth_vram_safety", 0.82))
    batch_min: int = int(cfg.get("batch_min", 1))
    batch_max: int = int(cfg.get("batch_max", 512))

    # Activation memory scales quadratically with spatial resolution
    scale = (imgsz / 640.0) ** 2
    per_sample_mb = per_sample_mb_640 * scale

    total_budget_mb = vram_gb * 1024.0 * safety
    available_mb = total_budget_mb - static_mb

    if available_mb <= 0 or per_sample_mb <= 0:
        max_safe = batch_min
    else:
        max_safe = max(batch_min, min(batch_max, math.floor(available_mb / per_sample_mb)))

    # Power-of-2 quantisation improves GPU utilisation (not strictly required)
    pow2 = 1
    while pow2 * 2 <= max_safe:
        pow2 *= 2
    max_safe = pow2

    if requested_batch is not None and requested_batch > 0:
        return max(batch_min, min(requested_batch, max_safe))

    return max_safe


def _build_gradual_fractions(
    start: float,
    step_min: float = 0.15,
    step_max: float = 0.20,
) -> list[float]:
    """Build fractions from start to 1.0 where every increment is between step_min and step_max."""
    if start >= 0.99:
        return [1.0]
    step_mid = (step_min + step_max) / 2.0
    span = 1.0 - start
    steps = max(1, round(span / step_mid))
    step_size = span / steps
    if step_size > step_max:
        steps += 1
    elif step_size < step_min and steps > 1:
        steps -= 1
    fractions = [round(start + i * (span / steps), 2) for i in range(steps)]
    fractions.append(1.0)
    return sorted(set(fractions))


def get_fractions_for_resolution(
    resolution: int,
    is_lowest_rung: bool,
    opt_config: dict[str, Any] | None = None,
) -> list[float]:
    """Generate gradual fraction progression in configurable step increments up to 100%.

    Step bounds are read from opt_config keys fraction_step_min / fraction_step_max
    (defaults: 0.15 / 0.20).

    Rules:
      - Lowest resolution rung (<= 320px) starts from initial fraction (15%)
        and increases gradually: [0.15, 0.35, 0.55, 0.75, 1.00].
      - Subsequent higher resolution rungs (480px, 640px) start from 15% data
        and increase gradually: [0.15, 0.35, 0.55, 0.75, 1.00].
    """
    opt_config = opt_config or {}
    step_min = float(opt_config.get("fraction_step_min", 0.15))
    step_max = float(opt_config.get("fraction_step_max", 0.20))
    initial = float(opt_config.get("initial_fraction", 0.15))
    if is_lowest_rung and resolution <= 320:
        if (
            abs(initial - 0.15) < 0.05
            and abs(step_min - 0.15) < 0.01
            and abs(step_max - 0.20) < 0.01
        ):
            return [0.15, 0.35, 0.55, 0.75, 1.00]
        return _build_gradual_fractions(initial, step_min, step_max)
    next_res_start = float(opt_config.get("next_res_start_fraction", 0.15))
    if (
        abs(next_res_start - 0.15) < 0.05
        and abs(step_min - 0.15) < 0.01
        and abs(step_max - 0.20) < 0.01
    ):
        return [0.15, 0.35, 0.55, 0.75, 1.00]
    return _build_gradual_fractions(next_res_start, step_min, step_max)


def build_ladder_curriculum(
    res_ladder: list[int],
    total_epochs: int,
    vram_gb: float,
    requested_batch: int | None,
    opt_config: dict[str, Any],
    *,
    start_resolution: int | None = None,
) -> list[YOLOLadderStage]:
    """Construct deterministic multi-stage resolution ladder curriculum.

    Implements gradual intra-resolution fraction progression in 15%-20% increments:
      - Lowest resolution rung (<= 320px) starts from initial fraction (15%) and
        gradually progresses: 15% -> 35% -> 55% -> 75% -> 100%.
      - Subsequent higher resolution rungs (480px, 640px) start from 15% data
        and gradually progress: 15% -> 35% -> 55% -> 75% -> 100%.
    This prevents overfitting and breaks performance plateaus when scaling across spatial rungs.
    """
    opt_config = opt_config or {}
    rungs = sorted(set(res_ladder)) if res_ladder else [320, 480, 640]

    if start_resolution is not None and start_resolution in rungs:
        rungs = [r for r in rungs if r >= start_resolution]
        if not rungs:
            rungs = [start_resolution]

    patience_base = int(opt_config.get("plateau_patience", 10))
    lowest_ladder_res = min(res_ladder) if res_ladder else 320

    stage_specs: list[tuple[int, float]] = []
    for rung in rungs:
        is_lowest = rung == lowest_ladder_res
        fracs = get_fractions_for_resolution(rung, is_lowest, opt_config)
        for f in fracs:
            stage_specs.append((rung, f))

    lowest_res = min(s[0] for s in stage_specs)
    # Stage weights scale with spatial resolution and data manifold fraction
    weights = [(res / lowest_res) * frac for res, frac in stage_specs]
    total_weight = sum(weights)
    epoch_allocations = [max(1, round(total_epochs * (w / total_weight))) for w in weights]
    diff = total_epochs - sum(epoch_allocations)
    if diff != 0:
        epoch_allocations[-1] = max(1, epoch_allocations[-1] + diff)

    stages: list[YOLOLadderStage] = []
    for idx, ((res, frac), ep_budget) in enumerate(
        zip(stage_specs, epoch_allocations, strict=True)
    ):
        safe_batch = compute_safe_batch_size(
            imgsz=res,
            vram_gb=vram_gb,
            requested_batch=requested_batch,
            opt_config=opt_config,
        )
        stage_patience = max(5, patience_base + (idx * 2))
        stages.append(
            YOLOLadderStage(
                stage_index=idx + 1,
                resolution=res,
                batch_size=safe_batch,
                fraction=frac,
                target_epochs=ep_budget,
                patience=stage_patience,
            )
        )

    return stages


@dataclass(frozen=True)
class SawtoothMessages:
    """Console and log wording of one Sawtooth batch-governor call site.

    ``pressure`` and ``headroom`` are ``str.format`` templates over ``pct`` (peak VRAM use,
    percent), ``thresh`` (threshold, percent), ``old`` and ``new`` (batch sizes).
    """

    pressure: str
    headroom: str
    probe_notice: str


def sawtooth_batch(
    batch: int,
    resolution: int,
    vram_gb: float,
    requested_batch: int | None,
    opt_config: dict[str, Any],
    messages: SawtoothMessages,
) -> int:
    """Return ``batch`` adjusted to the peak VRAM use recorded since the last probe.

    Sawtooth VRAM Governor: halves the batch (floor 2) when peak use crossed the pressure
    threshold and promotes it toward the VRAM-safe limit when use stayed below the headroom
    threshold. Both thresholds come from ``opt_config`` (``sawtooth_vram_pressure_thresh``,
    default 0.90, and ``sawtooth_vram_headroom_thresh``, default 0.60). The CUDA peak counter
    is reset after every probe. Without CUDA the batch is returned unchanged.
    """
    if not (torch.cuda.is_available() and vram_gb > 0):
        return batch
    try:
        peak_bytes = torch.cuda.max_memory_allocated(0)
        peak_gb = peak_bytes / (1024**3)
        vram_pct = peak_gb / vram_gb
        pressure_thresh = float(opt_config.get("sawtooth_vram_pressure_thresh", 0.90))
        headroom_thresh = float(opt_config.get("sawtooth_vram_headroom_thresh", 0.60))
        pct, new_batch, template, thresh = vram_pct * 100, batch, "", 0.0
        # 1. Sawtooth Pressure Sentinel: downscale if memory spiked dangerously
        if vram_pct >= pressure_thresh:
            new_batch = max(2, batch // 2)
            if new_batch < batch:
                template, thresh = messages.pressure, pressure_thresh
        # 2. Dynamic Headroom Promotion: scale up if VRAM utilization was low
        elif peak_bytes > 0 and vram_pct < headroom_thresh:
            safe_limit = compute_safe_batch_size(resolution, vram_gb, requested_batch, opt_config)
            if safe_limit > batch:
                new_batch = min(safe_limit, max(batch * 2, safe_limit))
                template, thresh = messages.headroom, headroom_thresh
        if template:
            print(
                template.format(pct=pct, thresh=thresh * 100, old=batch, new=new_batch),
                flush=True,
            )
            batch = new_batch
        torch.cuda.reset_peak_memory_stats(0)
    except (RuntimeError, ValueError) as mem_err:
        logger.debug(messages.probe_notice, mem_err)
    return batch
