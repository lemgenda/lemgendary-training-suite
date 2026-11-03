"""Stochastic Weight Averaging (SWA) for the training engine.

Built on ``torch.optim.swa_utils``: :class:`~torch.optim.swa_utils.AveragedModel`
keeps the running mean of the weights, :class:`~torch.optim.swa_utils.SWALR`
anneals the learning rate in the SWA phase and
:func:`~torch.optim.swa_utils.update_bn` re-estimates BatchNorm statistics of the
averaged weights before they are evaluated.

The engine talks to one object, :class:`SwaPhase`, through three calls:

* :func:`attach_swa` at the start of the run (``None`` when SWA is disabled, so
  a disabled run executes exactly the pre-SWA code path);
* :meth:`SwaPhase.end_epoch` after the governor audit of every epoch;
* :meth:`SwaPhase.finish` once the epoch loop has ended.

The train pass asks :meth:`SwaPhase.owns_lr` whether ``SWALR`` has taken over the
learning rate from the main scheduler.

SWA never touches the governor: its metrics are not fed to the SOTA tracker or
to ``audit_epoch``, so curriculum transitions still go only through the
governor's ``_commit_rung`` (Monotonic Ratchet pair rule). The averaged weights
are written as ``vault_swa.pth`` next to the vault checkpoints, which makes
them an ingredient of the Pareto Model Soup (``lemtrain soup``). The SWA state
needed to resume is kept in the ``swa_state.pth`` sidecar, so ``latest.pth``,
``best.pth`` and the vault checkpoints keep a single model each.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.swa_utils import SWALR, AveragedModel, update_bn

from training.checkpoint import safe_atomic_save, safe_load_checkpoint
from training.checkpoint.manager import CHECKPOINT_LOAD_ERRORS
from training.swa_config import SwaConfig, SwaLrSchedule
from training.training.context import TrainingContext
from training.training.epoch import _unpack_batch
from training.training.step_progress import is_main_process
from training.training.validation import validate_one_epoch

logger = logging.getLogger("lemtrain.training.swa")

SWA_STATE_FILENAME = "swa_state.pth"
SWA_VAULT_FILENAME = "vault_swa.pth"
SWA_VAULT_METRIC = "swa"


class SwaAverager:
    """The averaged copy of a model and the optional ``SWALR`` schedule (no I/O).

    The ``AveragedModel`` and ``SWALR`` are created on first use, so nothing is
    copied and the learning rate is untouched before the SWA phase begins.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, config: SwaConfig):
        self._model = model
        self._optimizer = optimizer
        self._config = config
        self.averaged: AveragedModel | None = None
        self.scheduler: SWALR | None = None

    @property
    def n_averaged(self) -> int:
        """Number of weight snapshots in the average."""
        return 0 if self.averaged is None else int(self.averaged.n_averaged.item())

    @property
    def module(self) -> nn.Module | None:
        """The averaged network (``None`` before the first update)."""
        return None if self.averaged is None else self.averaged.module

    def update(self) -> None:
        """Add the live model's current weights to the running average."""
        if self.averaged is None:
            self.averaged = AveragedModel(self._model)
        self.averaged.update_parameters(self._model)

    def step_lr(self) -> None:
        """Advance ``SWALR`` by one epoch (no-op when ``swa_lr`` is not configured)."""
        schedule = self._config.lr
        if schedule is None:
            return
        if self.scheduler is None:
            self.scheduler = _new_swalr(self._optimizer, schedule)
        self.scheduler.step()

    def refresh_bn(self, loader: Iterable[Any], device: torch.device) -> None:
        """Recompute BatchNorm running statistics of the averaged network."""
        if self.averaged is None:
            return
        update_bn(
            _model_inputs(loader, device, self._config.bn_update_batches), self.averaged.module
        )

    def state_dict(self) -> dict[str, Any]:
        """Averaged weights (with ``n_averaged``) and the ``SWALR`` state."""
        return {
            "averaged": None if self.averaged is None else self.averaged.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore what :meth:`state_dict` returned."""
        if state.get("averaged") is not None:
            self.averaged = AveragedModel(self._model)
            self.averaged.load_state_dict(state["averaged"])
        schedule = self._config.lr
        if state.get("scheduler") is not None and schedule is not None:
            self.scheduler = _new_swalr(self._optimizer, schedule)
            self.scheduler.load_state_dict(state["scheduler"])


def _new_swalr(optimizer: torch.optim.Optimizer, schedule: SwaLrSchedule) -> SWALR:
    return SWALR(
        optimizer,
        swa_lr=schedule.swa_lr,
        anneal_epochs=schedule.anneal_epochs,
        anneal_strategy="linear" if schedule.anneal_strategy == "linear" else "cos",
    )


def _model_inputs(
    loader: Iterable[Any], device: torch.device, max_batches: int | None
) -> Iterator[Any]:
    """Yield the model inputs of ``loader`` batches (all batches when ``max_batches`` is None)."""
    for batch in itertools.islice(loader, max_batches):
        inputs, _targets, _task_idx = _unpack_batch(batch, device)
        yield inputs


class SwaPhase:
    """Engine-facing SWA component: when to average, persist, evaluate and publish."""

    def __init__(
        self,
        config: SwaConfig,
        averager: SwaAverager,
        start_epoch: int,
        checkpoints_dir: Path | None,
    ) -> None:
        self.config = config
        self.averager = averager
        self.start_epoch = start_epoch
        self.checkpoints_dir = checkpoints_dir
        self.evaluated_epoch: int | None = None

    def owns_lr(self, epoch: int) -> bool:
        """True when ``SWALR`` (not the main scheduler) sets the LR of ``epoch``."""
        return self.config.lr is not None and epoch >= self.start_epoch

    def is_update_epoch(self, epoch: int) -> bool:
        """True when ``epoch`` adds its weights to the average."""
        if epoch < self.start_epoch:
            return False
        return (epoch - self.start_epoch) % self.config.update_every == 0

    def end_epoch(self, ctx: TrainingContext, epoch: int) -> None:
        """Average this epoch's weights when due, step ``SWALR`` and save the SWA state."""
        if not self.is_update_epoch(epoch):
            return
        self.averager.update()
        self.averager.step_lr()
        self._save_state(epoch)
        count = self.averager.n_averaged
        logger.info("SWA averaged epoch %d (n_averaged=%d)", epoch, count)
        if is_main_process():
            print(f"[SWA] Epoch {epoch} added to the weight average (n={count}).", flush=True)
        if self.config.eval_every and count % self.config.eval_every == 0:
            self.evaluate(ctx, epoch)

    def finish(self, ctx: TrainingContext, epoch: int) -> dict[str, float] | None:
        """Evaluate and publish the final average unless that already happened at ``epoch``."""
        if self.averager.n_averaged == 0 or self.evaluated_epoch == epoch:
            return None
        return self.evaluate(ctx, epoch)

    def evaluate(self, ctx: TrainingContext, epoch: int) -> dict[str, float]:
        """Refresh BatchNorm, validate the averaged model and write ``vault_swa.pth``.

        The live model, the governor and the SOTA tracker are not changed.
        """
        module = self.averager.module
        if module is None:
            return {}
        self.averager.refresh_bn(ctx.train_loader, ctx.device_info.device)
        swa_ctx = dataclasses.replace(
            ctx, model=module, raw_model=module, lifecycle_manager=None, resume_state=None, swa=None
        )
        metrics = validate_one_epoch(swa_ctx, epoch)
        self.evaluated_epoch = epoch
        ctx.metadata["swa_metrics"] = dict(metrics)
        if is_main_process():
            shown = " | ".join(
                f"{key}: {value:.4f}"
                for key, value in sorted(metrics.items())
                if isinstance(value, (int, float))
            )
            print(f"[SWA] Averaged model (n={self.averager.n_averaged}): {shown}", flush=True)
        self._save_ingredient(ctx, epoch, metrics)
        return metrics

    def restore(self, resume_state: Any) -> None:
        """Load the SWA sidecar of a resumed run, or delete a stale one on a fresh run.

        The sidecar is used only when it was written at the last completed epoch of
        the resumed checkpoint; otherwise averaging starts again from scratch.
        """
        path = self._path(SWA_STATE_FILENAME)
        if path is None or not path.exists():
            return
        if resume_state is None:
            self._discard(path, "fresh run")
            return
        completed = int(resume_state.epoch) - (1 if resume_state.is_intra_epoch else 0)
        try:
            state = safe_load_checkpoint(path, map_location="cpu")
        except (*CHECKPOINT_LOAD_ERRORS, TypeError) as load_err:
            logger.warning(
                "SWA state %s could not be loaded, averaging restarts: %s", path, load_err
            )
            return
        if int(state.get("epoch", -1)) != completed:
            logger.warning(
                "SWA state %s is from epoch %s but the resumed run completed epoch %d; "
                "averaging restarts.",
                path,
                state.get("epoch"),
                completed,
            )
            return
        self.start_epoch = int(state.get("start_epoch", self.start_epoch))
        self.averager.load_state_dict(state)
        logger.info("Restored SWA state (n_averaged=%d)", self.averager.n_averaged)

    def _path(self, filename: str) -> Path | None:
        return None if self.checkpoints_dir is None else self.checkpoints_dir / filename

    def _save_state(self, epoch: int) -> None:
        path = self._path(SWA_STATE_FILENAME)
        if path is None or not is_main_process():
            return
        payload = {"epoch": epoch, "start_epoch": self.start_epoch, **self.averager.state_dict()}
        if not safe_atomic_save(payload, path):
            logger.error("Failed saving SWA state at epoch %d to %s", epoch, path)

    def _save_ingredient(self, ctx: TrainingContext, epoch: int, metrics: dict[str, float]) -> None:
        path = self._path(SWA_VAULT_FILENAME)
        module = self.averager.module
        if path is None or module is None or not is_main_process():
            return
        governor_state = ctx.governor.get_state() if hasattr(ctx.governor, "get_state") else None
        payload = {
            "epoch": epoch,
            "model_name": ctx.model_name,
            "model_state": module.state_dict(),
            "governor_state": governor_state,
            "current_metrics": dict(metrics),
            "vault_metric": SWA_VAULT_METRIC,
            "vault_score": float(metrics.get("val_loss", 0.0)),
            "vault_epoch": epoch,
            "is_swa": True,
            "swa_n_averaged": self.averager.n_averaged,
            "swa_start_epoch": self.start_epoch,
        }
        if safe_atomic_save(payload, path):
            print(f"[SWA] Soup ingredient saved to {path.name}.", flush=True)
        else:
            logger.error("Failed saving SWA soup ingredient to %s", path)

    @staticmethod
    def _discard(path: Path, reason: str) -> None:
        if not is_main_process():
            return
        try:
            path.unlink()
            logger.info("Removed stale SWA state %s (%s)", path, reason)
        except OSError as unlink_err:
            logger.warning("Could not remove stale SWA state %s: %s", path, unlink_err)


def attach_swa(ctx: TrainingContext) -> SwaPhase | None:
    """Build the run's :class:`SwaPhase` from ``ctx.config`` (``None`` when SWA is disabled)."""
    config = SwaConfig.from_config(ctx.config)
    if not config.enabled:
        return None
    model = ctx.raw_model if ctx.raw_model is not None else ctx.model
    checkpoints_dir = (
        ctx.lifecycle_manager.checkpoints_dir if ctx.lifecycle_manager is not None else None
    )
    phase = SwaPhase(
        config,
        SwaAverager(model, ctx.optimizer, config),
        config.first_epoch(ctx.total_epochs),
        checkpoints_dir,
    )
    phase.restore(ctx.resume_state)
    if is_main_process():
        lr_note = f"SWALR to {config.lr.swa_lr:g}" if config.lr is not None else "main LR schedule"
        print(
            f"[SWA] Enabled: averaging from epoch {phase.start_epoch} every "
            f"{config.update_every} epoch(s), {lr_note}.",
            flush=True,
        )
    return phase
