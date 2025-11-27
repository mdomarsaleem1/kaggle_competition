"""Chronos-2 fine-tuning and forecasting for the Hull Tactical competition.

This script:
- Loads raw train/test from ``data/01_raw``.
- Sorts by ``date_id`` and extracts the target ``market_forward_excess_returns``.
- Loads the Chronos-2 foundation model.
- Fine-tunes on the full training series (quick default hyperparameters).
- Forecasts the full test horizon and writes a submission CSV.

Note: Fine-tuning is resource-intensive; adjust ``NUM_STEPS``/``BATCH_SIZE`` as needed.
"""
from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List,Iterable, Optional
from chronos import BaseChronosPipeline, Chronos2Pipeline

from pathlib import Path
path = Path(os.getcwd())

project_root = path.parent.absolute()
sys.path.insert(0, str(project_root))
import os

print(f"Current working directory: {os.getcwd()}")
print(f"Files in data/01_raw: {os.listdir(project_root)}") # This might also throw an error if data/01_raw doesn't exist in CWD

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the target folder (e.g., a 'lib' folder in the same directory)
lib_dir = os.path.join(current_dir, "lib")

# Add the folder to sys.path if it exists and is not already present
if os.path.exists(lib_dir) and lib_dir not in sys.path:
    sys.path.append(lib_dir)


import os

DATA_DIR = os.path.join(project_root , "data/01_raw")
print(DATA_DIR)
ID_COL = "date_id"
TARGET_COL = "market_forward_excess_returns"
MODEL_ID = "amazon/chronos-2"

# Fine-tuning hyperparameters (feel free to tweak)
NUM_STEPS = 200
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
CONTEXT_LEN = 512
PREDICTION_SAMPLES = 30
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




def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(DATA_DIR , "train.csv")  
    test_path = os.path.join(DATA_DIR , "test.csv")   
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def clean_and_sort(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for df in (train_df, test_df):
        df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce").astype(int)
    train_df = train_df.sort_values(ID_COL).reset_index(drop=True)
    test_df = test_df.sort_values(ID_COL).reset_index(drop=True)
    return train_df, test_df


def chronos_iterative_forecast(
    pipeline: BaseChronosPipeline,
    history: dict[str, np.ndarray],
    horizon: int,
    context_length: int,
    num_samples: int,
    future_covariates: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Iteratively extend Chronos forecasts to cover the full horizon, with covariate support."""
    preds: List[np.ndarray] = []
    
    # Initialize buffers
    target_buffer = history["target"].astype(float).copy()
    past_covs_buffers = {k: v.copy() for k, v in history.items() if k.startswith("past_") and k != "target"}

    remaining = horizon
    fut_cov_offset = 0

    while remaining > 0:
        step = remaining
        
        # Prepare context for prediction
        context_dict = {
            "target": target_buffer[-context_length:],
            "past_covariates": {k: v[-context_length:] for k,v in past_covs_buffers.items()}
        }

        if future_covariates:
            context_dict["future_covariates"] = {
                k: v[fut_cov_offset : fut_cov_offset + step] for k, v in future_covariates.items()
            }

        # Predict returns a list of forecasts for a list of contexts
        forecast_samples = pipeline.predict(
            context=[context_dict],
            prediction_length=step,
            num_samples=num_samples,
        )[0]
        
        step_pred = np.median(forecast_samples, axis=0)
        
        preds.append(step_pred)
        target_buffer = np.concatenate([target_buffer, step_pred])
        
        # IMPORTANT: This simplistic iterative forecast assumes that past covariates
        # for the forecast period are either known or can be generated. 
        # Here we are not extending them, which will cause an error if not handled.
        # For a real use-case, you would need to append generated or known future
        # values of past covariates to `past_covs_buffers` here.
        
        fut_cov_offset += len(step_pred)
        remaining -= len(step_pred)
        if len(step_pred) == 0:
            break

    return np.concatenate(preds) if preds else np.array([], dtype=float)


def main():
    train_df, test_df = load_raw()
    train_df, test_df = clean_and_sort(train_df, test_df)

    horizon = len(test_df)

    # --- START: COVARIATE DEFINITION (ACTION REQUIRED) ---
    # As shown in `models/chronos 2 sample.py`, using covariates can significantly
    # improve forecast accuracy. Please define your covariate columns here.
    # I am unable to access your data files, so I cannot suggest which columns to use.
    
    # `PAST_COVARIATE_COLS` are features known in the past. If they are not known for the
    # future, the iterative forecast function will need to be adapted.
    PAST_COVARIATE_COLS: List[str] = []  # e.g., ["feature1", "feature2"]
    
    # `FUTURE_COVARIATE_COLS` are features known for the forecast horizon (e.g., holidays, promotions).
    # These should be present in your test_df.
    FUTURE_COVARIATE_COLS: List[str] = []  # e.g., ["is_holiday"]
    # --- END: COVARIATE DEFINITION ---


    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu"
    print(f"Using device: {device}")
    print(f"Loading Chronos-2 model: {MODEL_ID}")
    base_pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map=device)

    # Prepare inputs for fine-tuning, now including covariates
    # This structure is aligned with the fine-tuning example in `chronos 2 sample.py`
    print("Preparing data for fine-tuning...")
    train_inputs = [
        {
            "target": train_df[TARGET_COL].values,
            "past_covariates": {
                f"past_{col}": train_df[col].values for col in PAST_COVARIATE_COLS
            },
            "future_covariates": {
                f"future_{col}": None for col in FUTURE_COVARIATE_COLS
            },
        }
    ]

    print("Fine-tuning Chronos-2...")
    ft_pipeline = base_pipeline.fit(
        inputs=train_inputs,
        prediction_length=horizon,  # Using full horizon for tuning
        num_steps=NUM_STEPS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        logging_steps=10,
    )
    print("Fine-tuning complete.")

    # --- START: VALIDATION WITH SHARPE RATIO ---
    print("Running validation to calculate Sharpe ratio...")
    val_split_idx = int(len(train_df) * 0.8)
    train_context_df = train_df.iloc[:val_split_idx]
    validation_df = train_df.iloc[val_split_idx:]
    val_horizon = len(validation_df)

    val_history = {
        "target": train_context_df[TARGET_COL].values,
        **{f"past_{col}": train_context_df[col].values for col in PAST_COVARIATE_COLS}
    }
    val_future_covs = {
        f"future_{col}": validation_df[col].values for col in FUTURE_COVARIATE_COLS
    }

    val_preds = chronos_iterative_forecast(
        ft_pipeline,
        history=val_history,
        horizon=val_horizon,
        context_length=min(CONTEXT_LEN, len(train_context_df)),
        num_samples=PREDICTION_SAMPLES,
        future_covariates=val_future_covs if FUTURE_COVARIATE_COLS else None,
    )

    if len(val_preds) == val_horizon:
        y_true_val = validation_df[TARGET_COL].values
        sharpe_score = hull_sharpe_ratio(y_true=y_true_val, y_pred=val_preds)
        print(f"Validation Hull Sharpe Ratio: {sharpe_score:.4f}")
    else:
        print("Validation prediction length mismatch, skipping Sharpe ratio calculation.")
    # --- END: VALIDATION WITH SHARPE RATIO ---

    # Prepare data for forecasting
    history_for_forecast = {
        "target": train_df[TARGET_COL].values,
        **{f"past_{col}": train_df[col].values for col in PAST_COVARIATE_COLS}
    }
    
    future_covs_for_forecast = {
        f"future_{col}": test_df[col].values for col in FUTURE_COVARIATE_COLS
    }

    print("Generating forecasts for submission...")
    preds = chronos_iterative_forecast(
        ft_pipeline,
        history=history_for_forecast,
        horizon=horizon,
        context_length=min(CONTEXT_LEN, len(train_df)),
        num_samples=PREDICTION_SAMPLES,
        future_covariates=future_covs_for_forecast if FUTURE_COVARIATE_COLS else None,
    )

    if len(preds) < horizon:
        preds = np.pad(preds, (0, horizon - len(preds)), constant_values=0.0)

    submission = pd.DataFrame({ID_COL: test_df[ID_COL].values, "prediction": preds})
    out_path = Path("data") / "04_models" / "submission_chronos_finetuned.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"Saved fine-tuned Chronos submission to {out_path}")


if __name__ == "__main__":
    main()
