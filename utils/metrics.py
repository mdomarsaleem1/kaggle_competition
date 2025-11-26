"""Project-specific evaluation metrics."""
from typing import Iterable

import numpy as np


def hull_sharpe_ratio(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-9) -> float:
    """Compute the Hull competition Sharpe-inspired score.

    The score mirrors the Kaggle metric where higher is better by computing
    the mean of the position-adjusted returns divided by their standard
    deviation. A small ``eps`` keeps the denominator stable when variance is
    close to zero.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    pnl = y_true_arr * y_pred_arr
    return float(np.mean(pnl) / (np.std(pnl) + eps))


def averaged_metric(scores):
    """Return the simple average of a collection of fold scores."""
    scores_arr = np.asarray(list(scores), dtype=float)
    return float(np.mean(scores_arr))
