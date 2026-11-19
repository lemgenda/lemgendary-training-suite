"""Unit and smoke tests for Prompt 18: FOREX predictor model training improvements."""

from __future__ import annotations

import torch

from training.data.forex_cv import PurgedKFoldCV, PurgedWalkForwardCV
from training.losses import ForexDualLoss, ForexLossWeights, pinball_loss
from training.training.forex_calibration import CalibrationEvaluator
from training.training.forex_equity import compute_regime_metrics


def test_purged_walk_forward_cv() -> None:
    """Verify walk-forward splits honor purging boundaries and produce valid folds."""
    num_samples = 1000
    num_folds = 5
    purge_window = 15

    cv = PurgedWalkForwardCV(
        num_samples=num_samples,
        num_folds=num_folds,
        warmup_fraction=0.3,
        purge_window=purge_window,
    )

    splits = list(cv.split())
    assert len(splits) == num_folds

    for fold in splits:
        # Train indices must precede validation indices
        assert max(fold.train_indices) < min(fold.val_indices)
        # Purge window gap must be maintained
        gap = min(fold.val_indices) - max(fold.train_indices)
        assert gap >= purge_window
        assert fold.purged_count == purge_window


def test_purged_kfold_cv() -> None:
    """Verify purged K-Fold cross-validation removes both lookahead and embargo leaks."""
    num_samples = 500
    n_splits = 4
    purge_window = 10
    embargo_pct = 0.02  # 10 bars

    cv = PurgedKFoldCV(
        num_samples=num_samples,
        n_splits=n_splits,
        purge_window=purge_window,
        embargo_pct=embargo_pct,
    )

    splits = list(cv.split())
    assert len(splits) == n_splits

    for fold in splits:
        train_set = set(fold.train_indices)
        val_set = set(fold.val_indices)

        # Train and val must be completely disjoint
        assert train_set.isdisjoint(val_set)

        # Samples in purge window immediately before val_start must not be in train
        val_start = min(val_set)
        for i in range(max(0, val_start - purge_window), val_start):
            assert i not in train_set

        # Samples in embargo window immediately after val_end must not be in train
        val_end = max(val_set) + 1
        for i in range(val_end, min(num_samples, val_end + cv.embargo_bars)):
            assert i not in train_set


def test_regime_aware_metrics() -> None:
    """Verify regime metrics correctly segment Sharpe, WinRate, and MaxDD by market condition."""
    # 6 trades: 3 in Trending, 3 in Volatile
    r_multiples = [1.5, 2.0, -1.0, -1.0, -1.0, 3.0]
    regimes = ["Trending", "Trending", "Trending", "Volatile", "Volatile", "Volatile"]

    metrics = compute_regime_metrics(r_multiples, regimes, risk=0.01)

    assert "Trending" in metrics
    assert "Volatile" in metrics

    trending = metrics["Trending"]
    assert trending.sample_count == 3
    # 2 wins out of 3 = 66.67%
    assert abs(trending.win_rate_pct - 66.67) < 0.1
    assert trending.profit_factor == 3.5

    volatile = metrics["Volatile"]
    assert volatile.sample_count == 3
    # 1 win out of 3 = 33.33%
    assert abs(volatile.win_rate_pct - 33.33) < 0.1
    assert volatile.max_drawdown_pct > 0.0


def test_pinball_loss_quantile_behavior() -> None:
    """Verify pinball loss penalizes errors asymmetrically based on quantile tau."""
    preds = torch.tensor([10.0])
    targets_high = torch.tensor([12.0])  # under-prediction by 2.0

    # tau = 0.9 heavily penalizes under-predictions: 0.9 * 2.0 = 1.8
    loss_high_tau = pinball_loss(preds, targets_high, tau=0.9)
    assert abs(loss_high_tau.item() - 1.8) < 1e-5

    # tau = 0.1 mildly penalizes under-predictions: 0.1 * 2.0 = 0.2
    loss_low_tau = pinball_loss(preds, targets_high, tau=0.1)
    assert abs(loss_low_tau.item() - 0.2) < 1e-5

    # Test integration inside ForexDualLoss with use_pinball_loss=True
    weights = ForexLossWeights(use_pinball_loss=True, pinball_quantiles=(0.1, 0.5, 0.9))
    criterion = ForexDualLoss(weights=weights)

    pred_dict = {
        "direction": torch.zeros(2, 3),
        "magnitude": torch.tensor([[10.0, 10.0], [15.0, 15.0]]),
    }
    label_dict = {
        "direction": torch.tensor([0, 2]),
        "magnitude": torch.tensor([[12.0, 8.0], [14.0, 16.0]]),
    }

    loss = criterion(pred_dict, label_dict)
    assert loss.item() > 0.0


def test_confidence_calibration_evaluator() -> None:
    """Verify Expected Calibration Error (ECE) and reliability binning calculations."""
    evaluator = CalibrationEvaluator(num_bins=5)

    # Perfectly calibrated predictions:
    # 10 samples with 0.8 confidence and 8 correct
    confidences = [0.8] * 10
    accuracies = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]

    report = evaluator.evaluate(confidences, accuracies)
    assert len(report.bins) == 5
    # Since all samples fall into [0.6, 0.8] bin and accuracy is exactly 80%, ECE should be near 0
    assert report.expected_calibration_error < 0.05
    assert report.brier_score > 0.0

    # Test empty input handling
    empty_report = evaluator.evaluate([], [])
    assert empty_report.expected_calibration_error == 0.0
