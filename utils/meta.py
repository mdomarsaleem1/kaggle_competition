"""Meta-learning helpers for blending model outputs."""
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def blend_predictions(
    predictions: Dict[str, Iterable[float]],
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """Blend model predictions with optional custom weights.

    Args:
        predictions: Mapping of model name to iterable predictions (aligned).
        weights: Optional mapping of model name to blending weight. If not
            provided, uniform weights are used.

    Returns:
        Pandas Series containing the blended prediction vector.
    """
    if not predictions:
        raise ValueError("At least one model prediction vector is required")

    pred_frames = []
    model_names = list(predictions.keys())

    for name, preds in predictions.items():
        pred_frames.append(pd.Series(preds, name=name))

    stacked = pd.concat(pred_frames, axis=1)

    if weights is None:
        weights_arr = np.ones(len(model_names)) / len(model_names)
    else:
        weights_arr = np.array([weights.get(name, 0.0) for name in model_names], dtype=float)
        weights_arr = weights_arr / weights_arr.sum()

    return pd.Series(stacked.values @ weights_arr, name="blended_prediction")
