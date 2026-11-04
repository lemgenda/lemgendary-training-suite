"""Measure FID with the training validation path, enforcing the Publication-Grade FID invariant.

The FID module is the one ``training.training.val_metrics`` builds for validation
(``FrechetInceptionDistance(feature=2048)``) and every batch goes through its ``update_image``
updater (micro-chunks of at most 2 images). This module only adds checks around that path:

* the module's feature size must be :data:`FID_FEATURE_DIM` (2048);
* every ``update`` call is routed through :class:`ChunkGuard`, which refuses chunks larger than
  :data:`FID_MAX_CHUNK` (2);
* every batch must reach the FID module (the validation updater logs and skips failed batches;
  a re-evaluation must not report a FID computed on part of the data).

:class:`FidInvariantError` deliberately is not one of the metric error types the validation
updater catches, so a violation stops the run instead of being logged and skipped.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from training.training import val_metrics
from training.training.epoch import _unpack_batch
from training.training.val_accumulators import ValAccumulators
from training.training.val_metrics import _fid_module

FID_FEATURE_DIM = 2048
FID_MAX_CHUNK = 2
# Frechet distance needs a covariance estimate, so at least two images per side.
MIN_IMAGES = 2


class FidInvariantError(Exception):
    """A ROBOTS.md FID invariant or a completeness check failed."""


class FidUnavailableError(RuntimeError):
    """The FID backend (torchmetrics + torch-fidelity Inception weights) cannot be built."""


class ChunkGuard:
    """Proxy for the FID metric that refuses oversized chunks and counts images per side."""

    def __init__(self, metric: Any) -> None:
        self.metric = metric
        self.real_images = 0
        self.fake_images = 0
        self.max_chunk = 0

    def update(self, imgs: torch.Tensor, real: bool) -> None:
        """Forward one chunk to the metric after the chunk-size check."""
        n = int(imgs.size(0))
        if n > FID_MAX_CHUNK:
            raise FidInvariantError(
                f"FID chunk of {n} images exceeds the ROBOTS.md limit of {FID_MAX_CHUNK}"
            )
        self.max_chunk = max(self.max_chunk, n)
        self.metric.update(imgs, real=real)
        if real:
            self.real_images += n
        else:
            self.fake_images += n

    def compute(self) -> torch.Tensor:
        """Frechet distance of the accumulated real and fake features."""
        return self.metric.compute()


def fid_backend_available() -> bool:
    """True when ``torchmetrics.image.fid.FrechetInceptionDistance`` could be imported."""
    return val_metrics.FrechetInceptionDistance is not None


def feature_dim(metric: Any) -> int:
    """Feature size of a torchmetrics FID module, read from its running feature sum."""
    sums = getattr(metric, "real_features_sum", None)
    if not isinstance(sums, torch.Tensor):
        raise FidInvariantError(f"cannot read the feature size of FID module {type(metric)!r}")
    return int(sums.numel())


def build_guarded_fid(acc: ValAccumulators, device: torch.device) -> ChunkGuard:
    """Build the validation FID module, check its feature size and install the chunk guard."""
    if not fid_backend_available():
        raise FidUnavailableError(
            "torchmetrics FID is not importable; install requirements.txt "
            "(torchmetrics, torch-fidelity, scipy)"
        )
    # The exact module validation builds, not a copy of its settings.
    metric = _fid_module(acc.modules, device)
    if metric is None or metric is False:
        raise FidUnavailableError(
            "FID module could not be built (see the warning above; the Inception weights "
            "download needs internet access on first use)"
        )
    dim = feature_dim(metric)
    if dim != FID_FEATURE_DIM:
        raise FidInvariantError(f"FID feature size is {dim}, ROBOTS.md requires {FID_FEATURE_DIM}")
    guard = ChunkGuard(metric)
    acc.modules.fid = guard
    return guard


def predict(model: Any, inputs: Any) -> Any:
    """Model predictions the way the validation pass obtains them (``eval_step`` first)."""
    eval_step_fn = getattr(model, "eval_step", None)
    if callable(eval_step_fn):
        out = eval_step_fn(inputs)
        return out.get("preds", inputs) if isinstance(out, dict) else inputs
    return model(inputs)


@dataclass(frozen=True)
class FidMeasurement:
    """Result of one FID pass."""

    fid: float
    n_images: int
    batches: int
    feature_dim: int
    max_chunk: int


def _accumulate(model: Any, batches: Iterable[Any], device: torch.device) -> tuple[Any, int]:
    """Feed every batch through ``update_image``; returns ``(accumulators, batch count)``."""
    acc = ValAccumulators()
    build_guarded_fid(acc, device)
    count = 0
    with torch.no_grad():
        for batch in batches:
            inputs, targets, _ = _unpack_batch(batch, device)
            before = acc.image.fid_batches_accumulated
            view = val_metrics.ValBatch(
                preds=predict(model, inputs),
                targets=targets,
                sota_cfg={"fid": 0.0},
                dataset_type=None,
                device=device,
                governor=None,
            )
            val_metrics.update_image(acc, view)
            count += 1
            if acc.image.fid_batches_accumulated != before + 1:
                raise FidInvariantError(
                    f"batch {count} did not reach the FID module (prediction/target shape "
                    "mismatch or a logged FID update error); refusing a partial FID"
                )
    return acc, count


def measure_fid(model: Any, batches: Iterable[Any], device: torch.device) -> FidMeasurement:
    """Compute FID of ``model`` predictions against targets over all ``batches``."""
    model.eval()
    acc, count = _accumulate(model, batches, device)
    guard: ChunkGuard = acc.modules.fid
    if guard.real_images != guard.fake_images:
        raise FidInvariantError(
            f"real ({guard.real_images}) and generated ({guard.fake_images}) image counts differ"
        )
    if guard.real_images < MIN_IMAGES:
        raise FidInvariantError(f"FID needs at least {MIN_IMAGES} images, got {guard.real_images}")
    fid = max(0.0, float(guard.compute().item()))
    return FidMeasurement(
        fid=fid,
        n_images=guard.real_images,
        batches=count,
        feature_dim=feature_dim(guard.metric),
        max_chunk=guard.max_chunk,
    )
