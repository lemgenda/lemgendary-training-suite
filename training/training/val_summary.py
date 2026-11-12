"""End-of-pass validation metrics computed from the running totals."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from training.training.forex_equity import max_drawdown_pct
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
    """Risk statistics of the simulated trades, as measured (no clipping).

    ``max_drawdown`` is a percentage of the compounded equity curve
    (:mod:`training.training.forex_equity`). Bounds the quality score needs are applied
    inside the score (``training.governance.quality_score``), never to these values.
    """

    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float


def trade_stats(trade_returns: list[float], risk_multiples: list[float]) -> TradeStats | None:
    """Profit factor, Sharpe, Sortino and equity drawdown; ``None`` without any trade."""
    if len(trade_returns) == 0:
        return None
    returns_arr = np.array(trade_returns, dtype=np.float64)
    sharpe, sortino = _risk_ratios(returns_arr)
    if len(risk_multiples) != len(trade_returns):
        logger.warning(
            "Equity curve covers %d of %d simulated trades (trades without a positive "
            "stop-loss, or a validation pass resumed from an older progress checkpoint).",
            len(risk_multiples),
            len(trade_returns),
        )
    return TradeStats(
        profit_factor=float(_profit_factor(returns_arr)),
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        max_drawdown=max_drawdown_pct(risk_multiples),
    )


def _profit_factor(returns_arr: np.ndarray) -> float:
    """Gross profit over gross loss: ``inf`` without losses, ``nan`` when every trade is flat."""
    gross_profit = float(returns_arr[returns_arr > 0].sum())
    gross_loss = float(np.abs(returns_arr[returns_arr < 0]).sum())
    if gross_loss > 0:
        return gross_profit / gross_loss
    return math.inf if gross_profit > 0 else math.nan


def _risk_ratios(returns_arr: np.ndarray) -> tuple[Any, Any]:
    """Unclipped Sharpe and Sortino ratios over at most 252 periods."""
    mean_r = float(np.mean(returns_arr))
    std_r = float(np.std(returns_arr))
    horizon = np.sqrt(min(252, len(returns_arr)))
    downside = returns_arr[returns_arr < 0]
    downside_std = float(np.std(downside)) if len(downside) > 0 else 1e-4
    return (mean_r / max(1e-6, std_r)) * horizon, (mean_r / max(1e-6, downside_std)) * horizon


def is_llm(ctx: Any) -> bool:
    """Whether the context's model is a causal LLM (by model key or class name)."""
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


def _llm_results(acc: ValAccumulators, avg_loss: float, results: dict[str, float]) -> None:
    """``eval_loss``, ``token_accuracy`` (percent) and ``perplexity`` of a causal LLM."""
    results["eval_loss"] = avg_loss
    tasks = acc.llm
    if tasks.total_llm_valid_tokens > 0:
        results["token_accuracy"] = round(
            (tasks.total_llm_correct_tokens / tasks.total_llm_valid_tokens) * 100.0, 4
        )
    if avg_loss < 20.0:
        results["perplexity"] = round(_perplexity(avg_loss), 4)


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
    """Direction accuracy, magnitude errors and, when the model traded, the trade metrics.

    Without a single directional (non-HOLD) prediction there is no trade to measure: the
    win rate, profit factor, Sharpe, Sortino and drawdown are left out (not substituted by
    the direction accuracy or by neutral defaults) and the pass is logged. The epoch's
    quality score is computed by the engine from these metrics with the single score
    definition (``training.governance.quality_score``).
    """
    totals = acc.direction
    if not (totals.has_forex_metrics and totals.total_forex_samples > 0):
        return
    samples = max(1, totals.total_forex_samples)
    tp_mae, sl_mae = _magnitude_mae(acc.trades)
    results.update(
        {
            "dir_acc": (totals.correct_forex_samples / samples) * 100.0,
            "tp_mae": tp_mae,
            "sl_mae": sl_mae,
            "dir_entropy": totals.total_forex_entropy / samples,
        }
    )
    stats = trade_stats(acc.trades.trade_returns, acc.trades.trade_risk_multiples)
    if totals.total_trades <= 0 or stats is None:
        logger.warning(
            "Forex validation: no directional predictions over %d samples; win rate and "
            "trade metrics are not reported for this epoch.",
            totals.total_forex_samples,
        )
        return
    results.update(
        {
            "win_rate": (totals.winning_trades / totals.total_trades) * 100.0,
            "profit_factor": stats.profit_factor,
            "sharpe_ratio": stats.sharpe_ratio,
            "sortino_ratio": stats.sortino_ratio,
            "max_drawdown": stats.max_drawdown,
        }
    )


def summarize(ctx: Any, acc: ValAccumulators, step_count: int, elapsed: float) -> dict[str, float]:
    """Return the validation metrics of a finished pass."""
    avg_loss = acc.total_loss / max(1, step_count)
    results: dict[str, float] = {"val_loss": avg_loss, "val_time": elapsed}
    if is_llm(ctx):
        _llm_results(acc, avg_loss, results)
    _image_results(acc, step_count, results)
    _quality_results(acc, results)
    _task_results(acc, results)
    _forex_results(acc, results)
    return results
