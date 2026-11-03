"""Running validation totals, grouped by metric family.

The flat snapshot written into intra-epoch progress checkpoints (``val_accumulators``) keeps
its historical key names and order; :meth:`ValAccumulators.snapshot` and
:meth:`ValAccumulators.restore` translate between it and the groups.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from typing import Any

# Field metadata marking totals that are not part of the progress-checkpoint snapshot.
_TRANSIENT = {"snapshot": False}


@dataclass
class ImageTotals:
    """Reconstruction fidelity totals (SSIM, MSE, LPIPS, FID batches)."""

    total_ssim: float = 0.0
    total_mse: float = 0.0
    total_lpips: float = 0.0
    lpips_count: int = 0
    fid_batches_accumulated: int = 0
    has_image_metrics: bool = False


@dataclass
class ForexDirectionTotals:
    """Directional prediction totals of Forex models."""

    has_forex_metrics: bool = False
    total_forex_samples: int = 0
    correct_forex_samples: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    total_forex_entropy: float = 0.0


@dataclass
class ForexTradeTotals:
    """Take-profit / stop-loss errors and simulated trade returns of Forex models."""

    total_tp_mae: float = 0.0
    total_sl_mae: float = 0.0
    total_mag_samples: int = 0
    trade_returns: list[float] = field(default_factory=list)


@dataclass
class QualityTotals:
    """Quality-assessment scores (NIMA / AARD) and the rank-margin loss."""

    quality_preds_list: list[float] = field(default_factory=list)
    quality_targets_list: list[float] = field(default_factory=list)
    total_rank_margin_loss: float = field(default=0.0, metadata=_TRANSIENT)
    rank_margin_count: int = field(default=0, metadata=_TRANSIENT)


@dataclass
class TaskTotals:
    """Classification, parameter-prediction, segmentation and detection totals."""

    total_cls_correct: int = 0
    total_cls_samples: int = 0
    total_upn_mae: float = 0.0
    upn_mae_count: int = 0
    total_miou_inter: float = 0.0
    total_miou_union: float = 0.0
    detection_ap_scores: list[float] = field(default_factory=list)


@dataclass
class ImageMetricModules:
    """Lazily built LPIPS and FID modules: None before the first try, False after a failure."""

    lpips: Any = None
    fid: Any = None


_Group = ImageTotals | ForexDirectionTotals | ForexTradeTotals | QualityTotals | TaskTotals


def _snapshot_items(group: _Group) -> Iterator[tuple[str, Any]]:
    for spec in fields(group):
        if spec.metadata.get("snapshot", True):
            value = getattr(group, spec.name)
            yield spec.name, list(value) if isinstance(value, list) else value


def _restore_group(group: _Group, saved: dict[str, Any]) -> None:
    for spec in fields(group):
        if spec.metadata.get("snapshot", True):
            current = getattr(group, spec.name)
            setattr(group, spec.name, type(current)(saved.get(spec.name, current)))


@dataclass
class ValAccumulators:
    """All running totals of one validation pass."""

    total_loss: float = 0.0
    image: ImageTotals = field(default_factory=ImageTotals)
    direction: ForexDirectionTotals = field(default_factory=ForexDirectionTotals)
    trades: ForexTradeTotals = field(default_factory=ForexTradeTotals)
    quality: QualityTotals = field(default_factory=QualityTotals)
    tasks: TaskTotals = field(default_factory=TaskTotals)
    modules: ImageMetricModules = field(default_factory=ImageMetricModules)

    def _groups(self) -> tuple[_Group, ...]:
        return (self.image, self.direction, self.trades, self.quality, self.tasks)

    def snapshot(self) -> dict[str, Any]:
        """Return the flat progress-checkpoint snapshot (lists are copied)."""
        snap: dict[str, Any] = {"total_loss": self.total_loss}
        for group in self._groups():
            snap.update(_snapshot_items(group))
        return snap

    def restore(self, saved: dict[str, Any]) -> None:
        """Load totals from a progress-checkpoint snapshot; missing keys keep their defaults."""
        self.total_loss = float(saved.get("total_loss", 0.0))
        for group in self._groups():
            _restore_group(group, saved)
