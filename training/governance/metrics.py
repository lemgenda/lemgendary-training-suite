"""Metric registry, direction mappings, weights, and composite quality score calculation."""

import logging
from dataclasses import dataclass

logger = logging.getLogger("lemtrain.governance.metrics")

DEFAULT_METRIC_DIRECTIONS: dict[str, bool] = {
    # True = Higher is better, False = Lower is better
    "val_loss": False,
    "eval_loss": False,
    "perplexity": False,
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
    "mse": False,
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

LOWER_IS_BETTER_METRICS: set[str] = {
    "loss",
    "val_loss",
    "eval_loss",
    "perplexity",
    "train_loss",
    "lpips",
    "fid",
    "mae",
    "mse",
    "tp_mae",
    "sl_mae",
    "max_drawdown",
    "maxdd",
    "drawdown",
    "rank_margin",
    "entropy",
    "dir_entropy",
}

CANONICAL_METRIC_ALIASES: dict[str, str] = {
    "evalloss": "eval_loss",
    "eval_loss": "eval_loss",
    "valloss": "eval_loss",
    "val_loss": "eval_loss",
    "perplexity": "perplexity",
    "tokenaccuracy": "token_accuracy",
    "token_accuracy": "token_accuracy",
    "diracc": "dir_acc",
    "dir_acc": "dir_acc",
    "winrate": "win_rate",
    "win_rate": "win_rate",
    "profitfactor": "profit_factor",
    "profit_factor": "profit_factor",
    "sharpe": "sharpe_ratio",
    "sharperatio": "sharpe_ratio",
    "sharpe_ratio": "sharpe_ratio",
    "sortino": "sortino_ratio",
    "sortinoratio": "sortino_ratio",
    "sortino_ratio": "sortino_ratio",
    "maxdd": "max_drawdown",
    "max_drawdown": "max_drawdown",
    "tpmae": "tp_mae",
    "tp_mae": "tp_mae",
    "slmae": "sl_mae",
    "sl_mae": "sl_mae",
    "direntropy": "dir_entropy",
    "dir_entropy": "dir_entropy",
    "map_50": "map50",
    "map50": "map50",
    "map_50_95": "map50_95",
    "map50_95": "map50_95",
    "map50-95": "map50_95",
}


def normalize_metric_name(name: str) -> str:
    """Normalize metric key to canonical lower-cased snake_case representation."""
    cleaned = name.strip().lower().replace("-", "_")
    unscored = cleaned.replace("_", "")
    return CANONICAL_METRIC_ALIASES.get(cleaned, CANONICAL_METRIC_ALIASES.get(unscored, cleaned))


def is_lower_better(metric_name: str) -> bool:
    """Determine whether a metric improves when its value decreases."""
    norm = normalize_metric_name(metric_name)
    if norm in LOWER_IS_BETTER_METRICS:
        return True
    if norm in DEFAULT_METRIC_DIRECTIONS:
        return not DEFAULT_METRIC_DIRECTIONS[norm]
    return any(sub in norm for sub in ("loss", "mae", "mse", "drawdown", "lpips", "fid", "margin"))


def _lower_is_better_contribution(metric_key: str, val: float) -> float:
    """Bounded positive contribution for a lower-is-better metric (matches TelemetryEngine)."""
    bounded_metrics = {
        "fid": 100.0 - val,
        "lpips": 1.0 - val,
        "rank_margin": 10.0 - val,
        "max_drawdown": max(0.0, 100.0 - val),
        "tp_mae": max(0.0, 50.0 - val),
        "sl_mae": max(0.0, 50.0 - val),
        "dir_entropy": max(0.0, 1.099 - val),
    }
    return bounded_metrics.get(metric_key, 1.0 / (val + 1e-6))


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
        if name_clean in self.directions:
            return self.directions[name_clean]
        return not is_lower_better(name_clean)

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
        """Compute canonical weighted Quality Score (aligned with TelemetryEngine)."""
        if not metrics:
            return 0.0

        if task_type == "quality":
            plcc_val = metrics.get("plcc", 0.0)
            srcc_val = metrics.get("srcc", 0.0)
            if plcc_val < 0.0 or srcc_val < 0.0:
                return 0.0

        quality_score = 0.0
        for key, val in metrics.items():
            k_lower = normalize_metric_name(key.replace(" ", "_"))
            if k_lower in self.weights:
                w = self.weights[k_lower]
                higher_better = self.directions.get(k_lower, True)
                if higher_better:
                    quality_score += val * w
                else:
                    quality_score += _lower_is_better_contribution(k_lower, val) * w

        return round(quality_score, 4)
