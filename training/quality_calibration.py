"""Optional post-hoc temperature calibration of quality (NIMA) models (R4-C4).

NIMA trains at temperature 1. Temperature scaling (Guo et al., 2017) is a post-hoc step: one
scalar ``T`` is fitted on the validation split with the weights frozen. For NIMA, ``T``
changes the predicted mean score and not only its confidence, so a fitted ``T`` is judged on
the EMD, PLCC and SRCC together: :func:`fit_temperature` picks the ``T`` of the grid
(0.5 to 2.0) with the lowest EMD and accepts it only if PLCC and SRCC do not drop below
their values at ``T = 1``; otherwise the result is ``T = 1``.

An accepted value goes into the model's registry entry as ``calibration.temperature``;
the export ``SoftmaxWrapper`` reads it through :func:`calibrated_temperature` (default 1.0).

Command (validation logits and target distributions saved as ``logits`` / ``targets`` arrays
of shape ``[N, 10]`` in one ``.npz``)::

    python -m training.quality_calibration val_scores.npz
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

CALIBRATION_KEY = "calibration"
TEMPERATURE_KEY = "temperature"
DEFAULT_GRID: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(31))
"""Temperatures 0.5, 0.55, ..., 2.0."""


def calibrated_temperature(model_info: Mapping[str, Any]) -> float:
    """The registry ``calibration.temperature`` (a positive number), else 1.0."""
    block = model_info.get(CALIBRATION_KEY) or {}
    if not isinstance(block, Mapping):
        raise TypeError(f"calibration must be a mapping, got {type(block).__name__}")
    value = float(block.get(TEMPERATURE_KEY, 1.0))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"calibration.temperature must be positive, got {value}")
    return value


@dataclass(frozen=True)
class ScoreQuality:
    """EMD (paper form), PLCC and SRCC of the predicted mean scores at one temperature."""

    temperature: float
    emd: float
    plcc: float
    srcc: float


@dataclass(frozen=True)
class CalibrationResult:
    """The temperature to use, whether the fit was accepted, and both measurements."""

    temperature: float
    accepted: bool
    baseline: ScoreQuality
    best: ScoreQuality


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    weights = np.exp(scaled)
    return weights / weights.sum(axis=1, keepdims=True)


def score_quality(logits: np.ndarray, targets: np.ndarray, temperature: float) -> ScoreQuality:
    """EMD, PLCC and SRCC of ``logits`` at ``temperature`` against target distributions."""
    probs = _softmax(np.asarray(logits, dtype=np.float64), temperature)
    target = np.asarray(targets, dtype=np.float64)
    target = target / np.clip(target.sum(axis=1, keepdims=True), 1e-12, None)
    cdf_gap = np.cumsum(probs, axis=1) - np.cumsum(target, axis=1)
    emd = float(np.mean(np.sqrt(np.mean(cdf_gap**2, axis=1) + 1e-8)))
    bins = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    pred_mean, target_mean = probs @ bins, target @ bins
    plcc = float(np.asarray(pearsonr(pred_mean, target_mean)[0]))
    srcc = float(np.asarray(spearmanr(pred_mean, target_mean)[0]))
    return ScoreQuality(temperature, emd, plcc, srcc)


def fit_temperature(
    logits: np.ndarray, targets: np.ndarray, grid: Sequence[float] = DEFAULT_GRID
) -> CalibrationResult:
    """The grid temperature with the lowest EMD, kept only if PLCC and SRCC do not drop."""
    baseline = score_quality(logits, targets, 1.0)
    best = min((score_quality(logits, targets, t) for t in grid), key=lambda q: q.emd)
    accepted = best.emd < baseline.emd and best.plcc >= baseline.plcc and best.srcc >= baseline.srcc
    return CalibrationResult(best.temperature if accepted else 1.0, accepted, baseline, best)


def main(argv: list[str] | None = None) -> int:
    """Fit the temperature of saved validation logits and print the registry entry to add."""
    parser = argparse.ArgumentParser(description="NIMA post-hoc temperature calibration")
    parser.add_argument("scores", help=".npz with 'logits' and 'targets' arrays [N, bins]")
    args = parser.parse_args(argv)
    with np.load(args.scores) as data:
        result = fit_temperature(data["logits"], data["targets"])
    for label, quality in (("T=1", result.baseline), ("best", result.best)):
        print(
            f"{label}: T={quality.temperature:.2f} EMD={quality.emd:.5f} "
            f"PLCC={quality.plcc:.4f} SRCC={quality.srcc:.4f}"
        )
    if result.accepted:
        print(f"Accepted. Registry entry: calibration: {{temperature: {result.temperature:.2f}}}")
    else:
        print("Not accepted (EMD not lower, or PLCC/SRCC dropped); keep temperature 1.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
