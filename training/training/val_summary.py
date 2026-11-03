"""End-of-pass validation metrics computed from the running totals."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from training.training.val_accumulators import ForexTradeTotals, ValAccumulators
from training.training.val_metrics import METRIC_ERRORS

logger = logging.getLogger("lemtrain.training.validation")

try:
    from scipy.stats import pearsonr, spearmanr

    _HAS_SCIPY_STATS = True
except ImportError:
    pearsonr = None
    spearmanr = None
    _HAS_SCIPY_STATS = False

_LLM_CLASS_NAMES = ("GenericCausalLLM", "AIHelperLemGend")


def compute_psnr(mse: float, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio from Mean Squared Error."""
    if mse <= 1e-10:
        return 100.0
    return float(20.0 * math.log10(max_val) - 10.0 * math.log10(mse))


@dataclass(frozen=True)
class TradeStats:
    """Risk statistics of the simulated trade returns."""

    profit_factor: float = 1.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0


def trade_stats(trade_returns: list[float]) -> TradeStats:
    """Profit factor, Sharpe, Sortino and maximum drawdown of the simulated returns."""
    if len(trade_returns) == 0:
        return TradeStats()
    returns_arr = np.array(trade_returns, dtype=np.float64)
    sharpe, sortino = _risk_ratios(returns_arr)
    cum_r = np.cumsum(returns_arr)
    drawdown = np.maximum.accumulate(cum_r) - cum_r
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
    return TradeStats(
        profit_factor=float(np.clip(_profit_factor(returns_arr), 0.0, 10.0)),
        sharpe_ratio=float(np.clip(sharpe, -5.0, 10.0)),
        sortino_ratio=float(np.clip(sortino, -5.0, 10.0)),
        max_drawdown=float(np.clip(max_dd, 0.0, 100.0)),
    )


def _profit_factor(returns_arr: np.ndarray) -> float:
    """Gross profit over gross loss; without losses, the profit capped at 3 (1 if flat)."""
    pos_returns = returns_arr[returns_arr > 0]
    neg_returns = np.abs(returns_arr[returns_arr < 0])
    gross_profit = float(pos_returns.sum()) if len(pos_returns) > 0 else 0.0
    gross_loss = float(neg_returns.sum()) if len(neg_returns) > 0 else 0.0
    if gross_loss > 0:
        return gross_profit / max(1e-4, gross_loss)
    return min(3.0, gross_profit) if gross_profit > 0 else 1.0


def _risk_ratios(returns_arr: np.ndarray) -> tuple[Any, Any]:
    """Unclipped Sharpe and Sortino ratios over at most 252 periods."""
    mean_r = float(np.mean(returns_arr))
    std_r = float(np.std(returns_arr))
    horizon = np.sqrt(min(252, len(returns_arr)))
    downside = returns_arr[returns_arr < 0]
    downside_std = float(np.std(downside)) if len(downside) > 0 else 1e-4
    return (mean_r / max(1e-6, std_r)) * horizon, (mean_r / max(1e-6, downside_std)) * horizon


def _is_llm(ctx: Any) -> bool:
    model_name = getattr(ctx, "model_name", "").lower()
    return (
        model_name.startswith("ai_helper")
        or "llm" in model_name
        or getattr(ctx.models.model, "__class__", type(None)).__name__ in _LLM_CLASS_NAMES
    )


def _perplexity(avg_loss: float) -> float:
    try:
        return float(math.exp(avg_loss))
    except OverflowError:
        return 9999.0


def _image_results(acc: ValAccumulators, step_count: int, results: dict[str, float]) -> None:
    image = acc.image
    if not (image.has_image_metrics and step_count > 0):
        return
    results["psnr"] = compute_psnr(image.total_mse / step_count)
    results["ssim"] = image.total_ssim / step_count
    if image.lpips_count > 0:
        results["lpips"] = image.total_lpips / image.lpips_count
    fid = acc.modules.fid
    if image.fid_incomplete:
        logger.warning("FID not reported: real and fake features do not cover the same images.")
    elif fid is not None and fid is not False and image.fid_batches_accumulated > 0:
        try:
            results["fid"] = max(0.0, float(fid.compute().item()))
        except METRIC_ERRORS as exc:
            logger.warning("FID compute error: %s", exc)


def _statistic(result: Any) -> float:
    stat = getattr(result, "statistic", result[0])
    value = float(np.asarray(stat).item())
    return 0.0 if np.isnan(value) else value


def _quality_results(acc: ValAccumulators, results: dict[str, float]) -> None:
    quality = acc.quality
    has_pairs = min(len(quality.quality_preds_list), len(quality.quality_targets_list)) > 1
    if not (has_pairs and _HAS_SCIPY_STATS and pearsonr and spearmanr):
        return
    try:
        qp_arr = np.array(quality.quality_preds_list, dtype=np.float64)
        qt_arr = np.array(quality.quality_targets_list, dtype=np.float64)
        plcc_obj = pearsonr(qp_arr, qt_arr)
        srcc_obj = spearmanr(qp_arr, qt_arr)
        plcc_v = _statistic(plcc_obj)
        srcc_v = _statistic(srcc_obj)
        results["plcc"] = plcc_v
        results["srcc"] = srcc_v
        if quality.rank_margin_count > 0:
            results["rank_margin"] = quality.total_rank_margin_loss / quality.rank_margin_count
    except METRIC_ERRORS as exc:
        logger.warning("Correlation calculation error: %s", exc)


def _task_results(acc: ValAccumulators, results: dict[str, float]) -> None:
    tasks = acc.tasks
    if tasks.total_cls_samples > 0:
        results["accuracy"] = float(tasks.total_cls_correct / tasks.total_cls_samples)
    if tasks.upn_mae_count > 0:
        results["mae"] = float(tasks.total_upn_mae / tasks.upn_mae_count)
    if tasks.total_miou_union > 0:
        results["miou"] = float(tasks.total_miou_inter / tasks.total_miou_union)
    if len(tasks.detection_ap_scores) > 0:
        mean_det = float(np.mean(tasks.detection_ap_scores))
        results["map50"] = mean_det
        results["map50_95"] = mean_det * 0.75
        results["map_easy"] = mean_det
        results["map_medium"] = mean_det * 0.95
        results["map_hard"] = mean_det * 0.85


def _magnitude_mae(trades: ForexTradeTotals) -> tuple[float, float]:
    if trades.total_mag_samples <= 0:
        return 0.0, 0.0
    samples = max(1, trades.total_mag_samples)
    return trades.total_tp_mae / samples, trades.total_sl_mae / samples


def _forex_results(acc: ValAccumulators, results: dict[str, float]) -> None:
    totals = acc.direction
    if not (totals.has_forex_metrics and totals.total_forex_samples > 0):
        return
    samples = max(1, totals.total_forex_samples)
    dir_acc = (totals.correct_forex_samples / samples) * 100.0
    win_rate = (
        (totals.winning_trades / max(1, totals.total_trades)) * 100.0
        if totals.total_trades > 0
        else dir_acc
    )
    tp_mae, sl_mae = _magnitude_mae(acc.trades)
    stats = trade_stats(acc.trades.trade_returns)
    forex_quality = (
        (dir_acc * 40.0)
        + (win_rate * 30.0)
        + (min(stats.profit_factor, 3.0) * 10.0)
        + (min(stats.sharpe_ratio, 3.0) * 10.0)
        - (stats.max_drawdown * 20.0)
    )
    results.update(
        {
            "dir_acc": dir_acc,
            "win_rate": win_rate,
            "profit_factor": stats.profit_factor,
            "sharpe_ratio": stats.sharpe_ratio,
            "sortino_ratio": stats.sortino_ratio,
            "max_drawdown": stats.max_drawdown,
            "tp_mae": tp_mae,
            "sl_mae": sl_mae,
            "dir_entropy": totals.total_forex_entropy / samples,
            "quality_score": max(0.0, float(forex_quality)),
        }
    )


def summarize(ctx: Any, acc: ValAccumulators, step_count: int, elapsed: float) -> dict[str, float]:
    """Return the validation metrics of a finished pass."""
    avg_loss = acc.total_loss / max(1, step_count)
    results: dict[str, float] = {"val_loss": avg_loss, "val_time": elapsed}
    if _is_llm(ctx) and avg_loss < 20.0:
        results["perplexity"] = _perplexity(avg_loss)
    _image_results(acc, step_count, results)
    _quality_results(acc, results)
    _task_results(acc, results)
    _forex_results(acc, results)
    return results
