"""Project-specific evaluation metrics."""
import numpy as np
from typing import Iterable, Optional

MIN_INVESTMENT = 0.0
MAX_INVESTMENT = 2.0
TRADING_DAYS_PER_YEAR = 252


def hull_sharpe_ratio(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    risk_free_rate: Optional[Iterable[float]] = None,
    forward_returns: Optional[Iterable[float]] = None,
) -> float:
    """Compute the Hull competition adjusted Sharpe ratio.

    This mirrors the Hull Tactical Kaggle competition metric:
    - Build strategy returns as a mix of risk-free and market returns.
    - Compute a Sharpe-like figure from geometric mean excess return.
    - Penalize excess volatility vs market and underperformance vs market.
    """

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_pred_arr = np.clip(y_pred_arr, 0, 2)

    # Fallback: simplified version with rf = 0, forward_returns = y_true
    if risk_free_rate is None or forward_returns is None:
        forward_returns_arr = y_true_arr
        risk_free_rate_arr = np.zeros_like(y_true_arr)
    else:
        forward_returns_arr = np.asarray(forward_returns, dtype=float)
        risk_free_rate_arr = np.asarray(risk_free_rate, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if forward_returns_arr.shape != y_true_arr.shape:
        raise ValueError("forward_returns must have same shape as y_true")
    if risk_free_rate_arr.shape != y_true_arr.shape:
        raise ValueError("risk_free_rate must have same shape as y_true")

    # Validate position constraints
    if float(y_pred_arr.max()) > MAX_INVESTMENT:
        raise ValueError(
            f"Position of {float(y_pred_arr.max())} exceeds maximum of {MAX_INVESTMENT}"
        )
    if float(y_pred_arr.min()) < MIN_INVESTMENT:
        raise ValueError(
            f"Position of {float(y_pred_arr.min())} below minimum of {MIN_INVESTMENT}"
        )

    # Strategy returns: convex mix of risk-free and market
    strategy_returns = risk_free_rate_arr * (1.0 - y_pred_arr) + y_pred_arr * forward_returns_arr

    # Strategy excess returns
    strategy_excess_returns = strategy_returns - risk_free_rate_arr

    # If excess returns are essentially zero everywhere, Sharpe is 0
    if np.allclose(strategy_excess_returns, 0.0, atol=1e-15):
        return 0.0

    # Geometric mean of excess returns
    strategy_excess_cumulative = np.prod(1.0 + strategy_excess_returns)
    strategy_mean_excess_return = strategy_excess_cumulative ** (1.0 / len(strategy_returns)) - 1.0

    strategy_std = strategy_returns.std()
    if strategy_std == 0:
        # Extremely degenerate edge case; avoid exploding Sharpe
        strategy_std = 1e-12

    sharpe = (
        strategy_mean_excess_return
        / strategy_std
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    strategy_volatility = float(strategy_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)

    # Market return and volatility
    market_excess_returns = forward_returns_arr - risk_free_rate_arr
    market_excess_cumulative = np.prod(1.0 + market_excess_returns)
    market_mean_excess_return = market_excess_cumulative ** (1.0 / len(forward_returns_arr)) - 1.0
    market_std = forward_returns_arr.std()

    if market_std == 0:
        # Perfectly flat market is not realistic; just treat as tiny to avoid div-by-zero.
        market_std = 1e-12

    market_volatility = float(market_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)

    # Volatility penalty: penalize vol > 1.2x market
    excess_vol = max(0.0, strategy_volatility / market_volatility - 1.2)
    vol_penalty = 1.0 + excess_vol

    # Return penalty: penalize underperformance vs market
    return_gap = max(
        0.0,
        (market_mean_excess_return - strategy_mean_excess_return)
        * 100.0
        * TRADING_DAYS_PER_YEAR,
    )
    return_penalty = 1.0 + (return_gap ** 2) / 100.0

    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return float(min(adjusted_sharpe, 1_000_000.0))


def averaged_metric(scores):
    """Return the simple average of a collection of fold scores."""
    scores_arr = np.asarray(list(scores), dtype=float)
    return float(np.mean(scores_arr))


def hull_sharpe_lightgbm(arg1, arg2):
    """LightGBM ``feval`` implementing the competition Sharpe metric.

    Supports both native LightGBM callback signature (preds, dataset) and
    sklearn callback signature (y_true, preds).
    """
    # Detect calling convention
    if hasattr(arg2, "get_label"):
        preds = np.asarray(arg1, dtype=float)
        y_true = np.asarray(arg2.get_label(), dtype=float)
    else:
        y_true = np.asarray(arg1, dtype=float)
        preds = np.asarray(arg2, dtype=float)

    try:
        score = hull_sharpe_ratio(y_true, preds)
    except ValueError:
        score = -1_000_000.0
    return "hull_sharpe", score, True  # higher is better


def hull_sharpe_xgboost(preds, dtrain):
    """XGBoost custom evaluation metric matching the competition Sharpe score.

    Note: This uses a simplified version of the metric for training monitoring,
    assuming risk_free_rate=0 and using labels as forward_returns.
    """
    y_true = dtrain.get_label()
    try:
        score = hull_sharpe_ratio(y_true, preds)
    except ValueError:
        # If validation fails (e.g., division by zero), return a poor score
        score = -1_000_000.0
    return "hull_sharpe", score
