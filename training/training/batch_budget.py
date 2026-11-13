"""Re-audit the VRAM-safe micro-batch when the curriculum steps up a resolution rung.

A resolution jump multiplies the activation memory of every sample, but the governor
kept the micro-batch of the previous rung; only a resume re-ran the VRAM audit. After a
jump :func:`refit_batch_for_resolution` probes the new resolution and shrinks the
micro-batch to the safe size while keeping ``batch * accumulation`` exactly equal
(:meth:`CurriculumState.fit_batch_to_budget`), so the optimisation trajectory keeps
its effective batch.

The probe runs forward/backward passes in train mode. Gradients are dropped at every
epoch start anyway, but BatchNorm running statistics would absorb the dummy batch, so
the model's buffers and train/eval mode are restored after the probe.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from torch import nn

from training.hardware.device_memory import device_memory_for
from training.hardware.probe import VramProbe, audit_hardware_vram
from training.training.context import TrainingContext

logger = logging.getLogger("lemtrain.batch_budget")

SafeBatchProbe = Callable[[nn.Module, int], int]
"""``(model, resolution) -> safe micro-batch``; injectable for tests."""


def _probe_model(ctx: TrainingContext) -> nn.Module:
    raw = ctx.models.raw_model
    return raw if isinstance(raw, nn.Module) else ctx.models.model


def probe_safe_batch(ctx: TrainingContext) -> SafeBatchProbe:
    """The run's VRAM audit as a :data:`SafeBatchProbe`."""

    def probe(model: nn.Module, resolution: int) -> int:
        governor = ctx.governance.governor
        return audit_hardware_vram(
            model_key=ctx.model_name,
            model_info=ctx.run.model_info,
            config=ctx.run.config,
            model=model,
            probe=VramProbe(
                device=ctx.hardware.device_info.device,
                res_override=resolution,
                mode="train",
                sample_fraction=float(getattr(governor, "current_fraction", 1.0)),
                fold=getattr(ctx.run.args, "fold", None),
                pairs=getattr(ctx.run.args, "pairs", None) or ctx.run.model_info.get("pairs"),
            ),
        )

    return probe


def _probe_preserving_state(model: nn.Module, resolution: int, probe: SafeBatchProbe) -> int:
    """Run ``probe`` and restore ``model``'s buffers and train/eval mode afterwards."""
    buffers = {name: buf.detach().clone() for name, buf in model.named_buffers()}
    was_training = model.training
    try:
        return int(probe(model, resolution))
    finally:
        with torch.no_grad():
            for name, buf in model.named_buffers():
                saved = buffers.get(name)
                if saved is not None:
                    buf.copy_(saved)
        model.train(was_training)


def refit_batch_for_resolution(ctx: TrainingContext, probe: SafeBatchProbe | None = None) -> bool:
    """Shrink the governor's micro-batch to the VRAM-safe size of its current resolution.

    Every GPU backend is probed through its device-memory adapter (CUDA/ROCm, XPU, MPS,
    DirectML; estimated budgets are logged by the adapter). The CPU has no device memory
    budget, so no probe runs there and the batch is kept unless a ``probe`` is injected.
    Returns True when the micro-batch changed.
    """
    if probe is None:
        device = ctx.hardware.device_info.device
        if device_memory_for(device, ctx.run.config).budget() is None:
            return False
        probe = probe_safe_batch(ctx)
    governor = ctx.governance.governor
    curriculum = governor.curriculum
    resolution = int(governor.current_res)
    safe_batch = _probe_preserving_state(_probe_model(ctx), resolution, probe)
    changed = curriculum.fit_batch_to_budget(safe_batch)
    if changed:
        logger.warning(
            "[VRAM REFIT] %dpx rung: micro-batch %d x accum %d (VRAM-safe batch %d)",
            resolution,
            curriculum.current_batch,
            curriculum.current_acc,
            safe_batch,
        )
    return changed
