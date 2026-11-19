"""Expected Calibration Error (ECE) and reliability diagram binning for directional models.

Quantifies the calibration fidelity of predicted directional probabilities:
    ECE = \\sum_{m=1}^{M} \\frac{|B_m|}{N} |\\text{acc}(B_m) - \\text{conf}(B_m)|
Ensures risk-sizing and downstream trade execution policies (such as Kelly Criterion)
receive honest, trustworthy probabilities rather than overconfident probability spikes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReliabilityBin:
    """Bin summary statistics for reliability diagrams."""

    bin_idx: int
    bin_lower: float
    bin_upper: float
    confidence: float
    accuracy: float
    sample_count: int


@dataclass(frozen=True)
class CalibrationReport:
    """Complete confidence calibration report including scalar ECE and reliability bins."""

    expected_calibration_error: float
    max_calibration_error: float
    brier_score: float
    bins: list[ReliabilityBin]


class CalibrationEvaluator:
    """Evaluates probability calibration, Expected Calibration Error (ECE), and reliability bins.

    Args:
        num_bins: Number of confidence bins across [0.0, 1.0] (default: 10).
    """

    def __init__(self, num_bins: int = 10) -> None:
        self.num_bins = max(2, int(num_bins))

    def get_num_bins(self) -> int:
        """Return the number of calibration bins."""
        return self.num_bins

    def _bin_sample(
        self,
        conf_arr: np.ndarray,
        acc_arr: np.ndarray,
        b_low: float,
        b_high: float,
        is_last: bool,
    ) -> tuple[int, float, float, float]:
        in_bin = (conf_arr >= b_low) & (conf_arr <= b_high if is_last else conf_arr < b_high)
        count = int(np.sum(in_bin))
        if count > 0:
            mean_conf = float(np.mean(conf_arr[in_bin]))
            mean_acc = float(np.mean(acc_arr[in_bin]))
            gap = abs(mean_acc - mean_conf)
        else:
            mean_conf = float((b_low + b_high) / 2.0)
            mean_acc = 0.0
            gap = 0.0
        return count, mean_conf, mean_acc, gap

    def _compute_single_bin(
        self,
        arrays: tuple[np.ndarray, np.ndarray],
        bounds: tuple[float, float],
        is_last: bool,
        idx: int,
    ) -> tuple[ReliabilityBin, int, float]:
        conf_arr, acc_arr = arrays
        b_low, b_high = bounds
        count, mean_c, mean_a, gap = self._bin_sample(conf_arr, acc_arr, b_low, b_high, is_last)
        bin_obj = ReliabilityBin(
            bin_idx=idx,
            bin_lower=round(b_low, 4),
            bin_upper=round(b_high, 4),
            confidence=round(mean_c, 4),
            accuracy=round(mean_a, 4),
            sample_count=count,
        )
        return bin_obj, count, gap

    def _compute_bins(
        self,
        conf_arr: np.ndarray,
        acc_arr: np.ndarray,
        bin_edges: np.ndarray,
    ) -> tuple[list[ReliabilityBin], float, float]:
        bins: list[ReliabilityBin] = []
        ece = 0.0
        mce = 0.0
        total_n = len(conf_arr)

        for i in range(self.num_bins):
            b_low, b_high = float(bin_edges[i]), float(bin_edges[i + 1])
            bin_obj, count, gap = self._compute_single_bin(
                (conf_arr, acc_arr), (b_low, b_high), i == self.num_bins - 1, i
            )
            if count > 0:
                ece += (float(count) / float(total_n)) * gap
                mce = max(mce, gap)
            bins.append(bin_obj)

        return bins, float(ece), float(mce)

    def evaluate(
        self,
        confidences: Sequence[float],
        accuracies: Sequence[bool | int],
    ) -> CalibrationReport:
        """Compute ECE, MCE, Brier score, and bin-level statistics."""
        if len(confidences) != len(accuracies):
            raise ValueError("confidences and accuracies must have identical lengths.")

        if len(confidences) == 0:
            return CalibrationReport(
                expected_calibration_error=0.0,
                max_calibration_error=0.0,
                brier_score=0.0,
                bins=[],
            )

        conf_arr = np.asarray(confidences, dtype=np.float64)
        acc_arr = np.asarray(accuracies, dtype=np.float64)
        brier = float(np.mean((conf_arr - acc_arr) ** 2))

        bin_edges = np.linspace(0.0, 1.0, self.num_bins + 1)
        bins, ece, mce = self._compute_bins(conf_arr, acc_arr, bin_edges)

        return CalibrationReport(
            expected_calibration_error=ece,
            max_calibration_error=mce,
            brier_score=brier,
            bins=bins,
        )
