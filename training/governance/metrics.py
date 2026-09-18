"""Metric registry, direction mappings, weights, and composite quality score calculation."""

from dataclasses import dataclass
import logging
from typing import Any

logger = logging.getLogger("lemtrain.governance.metrics")

DEFAULT_METRIC_DIRECTIONS: dict[str, bool] = {
    # True = Higher is better, False = Lower is better
    "val_loss": False,
    "train_loss": False,
    "loss": False,
    "plcc": True,
    "srcc": True,
    "psnr": True,
    "ssim": True,
    "lpips": False,
    "fid": False,
    "map50": True,
    "map50_95": True,
    "rank_margin": False,
    "accuracy": True,
    "mae": False,
    "miou": True,
    "map_medium": True,
    "map_hard": True,
    "accuracy_vqa": True,
    # Forex & Time-Series
    "dir_acc": True,
    "win_rate": True,
    "profit_factor": True,
    "sharpe_ratio": True,
    "sortino_ratio": True,
    "max_drawdown": False,
    "tp_mae": False,
    "sl_mae": False,
    "dir_entropy": False,
}

DEFAULT_METRIC_WEIGHTS: dict[str, float] = {
    "plcc": 50.0,
    "srcc": 50.0,
    "psnr": 10.0,
    "ssim": 40.0,
    "lpips": 40.0,
    "fid": 1.0,
    "map50": 100.0,
    "map50_95": 100.0,
    "rank_margin": 20.0,
    "accuracy": 100.0,
    "mae": 100.0,
    "miou": 100.0,
    "map_medium": 100.0,
    "map_hard": 100.0,
    "accuracy_vqa": 100.0,
    # Forex
    "dir_acc": 5.0,
    "win_rate": 5.0,
    "profit_factor": 10.0,
    "sharpe_ratio": 10.0,
    "sortino_ratio": 10.0,
    "max_drawdown": 1.0,
    "tp_mae": 1.0,
    "sl_mae": 1.0,
    "dir_entropy": 2.0,
}


@dataclass
class MetricDefinition:
    """Specification for an individual tracked metric."""

    name: str
    higher_is_better: bool
    weight: float


class MetricRegistry:
    """Central registry defining direction, weightings, and composite scoring across tasks."""

    def __init__(
        self,
        custom_directions: dict[str, bool] | None = None,
        custom_weights: dict[str, float] | None = None,
    ) -> None:
        self.directions = dict(DEFAULT_METRIC_DIRECTIONS)
        if custom_directions:
            self.directions.update(custom_directions)

        self.weights = dict(DEFAULT_METRIC_WEIGHTS)
        if custom_weights:
            self.weights.update(custom_weights)

    def register(self, name: str, higher_is_better: bool, weight: float = 1.0) -> None:
        """Register or update a metric definition."""
        self.directions[name.lower()] = higher_is_better
        self.weights[name.lower()] = weight

    def is_higher_better(self, metric_name: str) -> bool:
        """Return True if higher values indicate performance improvement."""
        name_clean = metric_name.lower().replace(" ", "_")
        return self.directions.get(name_clean, True)

    def is_improved(
        self,
        metric_name: str,
        new_val: float,
        baseline_val: float,
        min_delta: float = 1e-4,
    ) -> bool:
        """Evaluate whether new_val represents a significant improvement over baseline_val."""
        higher_better = self.is_higher_better(metric_name)
        if higher_better:
            return (new_val - baseline_val) > min_delta
        return (baseline_val - new_val) > min_delta

    def compute_composite_score(
        self,
        metrics: dict[str, float],
        task_type: str = "image",
    ) -> float:
        """Compute normalized composite quality score across tracked metrics."""
        if not metrics:
            return 0.0

        if task_type == "forex":
            # Forex-specific composite formula
            dir_acc = metrics.get("dir_acc", 0.0)
            win_rate = metrics.get("win_rate", 0.0)
            profit_factor = metrics.get("profit_factor", 0.0)
            sharpe = metrics.get("sharpe_ratio", 0.0)
            max_dd = metrics.get("max_drawdown", 0.0)
            score = (dir_acc * 40.0) + (win_rate * 30.0) + (min(profit_factor, 3.0) * 10.0) + (min(sharpe, 3.0) * 10.0) - (max_dd * 20.0)
            return round(score, 4)

        # Vision composite score
        total_weighted_score = 0.0
        total_weight = 0.0

        for key, val in metrics.items():
            k_lower = key.lower().replace(" ", "_")
            if k_lower in self.weights:
                w = self.weights[k_lower]
                higher_better = self.directions.get(k_lower, True)
                norm_val = val if higher_better else (1.0 / max(1e-4, abs(val)))
                total_weighted_score += norm_val * w
                total_weight += w

        if total_weight == 0.0:
            return 0.0

        return round(total_weighted_score / total_weight, 4)
