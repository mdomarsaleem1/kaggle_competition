# %% [markdown]
# # 00 - Data cleaning, alignment, and folds
# Clean the Hull Tactical data using the numeric ``date_id`` time key, align train/test feature names, and surface helpers for the Sharpe feval and simple ensembles.
# 

# %%
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing project utils
project_root = Path('..').resolve()
sys.path.insert(0, str(project_root))

from utils import (
    create_chronological_folds,
    default_catalog,
    save_folds_to_disk,
    save_table,
)
from utils.metrics import hull_sharpe_ratio, hull_sharpe_lightgbm, hull_sharpe_xgboost

# %% [markdown]
# ## Load raw data
# The catalog centralizes locations. ``date_id`` remains numeric and drives ordering for all downstream splits.
# 

# %%
catalog = default_catalog()
train_path = catalog["train_raw"]
test_path = catalog["test_raw"]

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")
print(f"Train columns (tail): {train_df.columns[-5:].tolist()}")
print(f"Test columns (tail):  {test_df.columns[-5:].tolist()}")

# %% [markdown]
# ## Cleaning pipeline (mirrors getting_started)
# - Sort by ``date_id`` (numeric).
# - Forward fill then median-fill feature columns.
# - Add missing test columns to train (e.g., lagged fields) so feature names align.
# - Keep ``market_forward_excess_returns`` as the training target and expose a ``target`` helper column.
# 

# %%
id_col = "date_id"
target_col = "market_forward_excess_returns"

lag_map = {
    "lagged_forward_returns": "forward_returns",
    "lagged_risk_free_rate": "risk_free_rate",
    "lagged_market_forward_excess_returns": target_col,
}


def prepare_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train = train_df.copy()
    test = test_df.copy()

    # Ensure numeric ordering and consistent sorting
    for df in (train, test):
        df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype(int)
    train = train.sort_values(id_col)
    test = test.sort_values(id_col)

    # Build feature set from the test columns (excluding metadata/target)
    test_feature_cols = [c for c in test.columns if c not in {"is_scored"}]
    feature_cols = [c for c in test_feature_cols if c not in {target_col,id_col} ]

    # Backfill lagged features in train so shapes match test
    for lagged_col, source_col in lag_map.items():
        if lagged_col in feature_cols and lagged_col not in train.columns and source_col in train.columns:
            train[lagged_col] = train[source_col].shift(1)

    # Add placeholder flag to match test schema
    if "is_scored" in test.columns and "is_scored" not in train.columns:
        train["is_scored"] = 1

    fill_cols = [c for c in feature_cols if c in train.columns]
    medians = train[fill_cols].median()

    def _fill(df: pd.DataFrame) -> pd.DataFrame:
        filled = df.copy()
        for col in fill_cols:
            if filled[col].isnull().any():
                filled[col] = filled[col].ffill()
                filled[col] = filled[col].fillna(medians[col])
        return filled

    clean_train = _fill(train)
    clean_test = _fill(test)

    # Add target helper column for modeling convenience
    clean_train["target"] = clean_train[target_col]

    # Align order: id + features (+ target/label columns)
    clean_train = clean_train[[id_col] + fill_cols + ["target", target_col]]
    extra_test_cols = ["is_scored"] if "is_scored" in clean_test.columns else []
    clean_test = clean_test[[id_col] + fill_cols + extra_test_cols]

    return clean_train, clean_test, fill_cols

# %%
clean_train, clean_test, feature_cols = prepare_train_test(train_df, test_df)
print(f"Clean train shape: {clean_train.shape}")
print(f"Clean test shape:  {clean_test.shape}")
print(f"Feature count: {len(feature_cols)}")

# %% [markdown]
# ## Persist cleaned data and folds
# Save aligned datasets and materialize chronological folds keyed by ``date_id`` for reuse across notebooks.
# 

# %%
clean_train_path = catalog["clean_train"]
clean_test_path = clean_train_path.with_name("clean_test.parquet")

save_table(clean_train, clean_train_path)
save_table(clean_test, clean_test_path)
print(f"Saved clean train -> {clean_train_path}")
print(f"Saved clean test  -> {clean_test_path}")

folds = create_chronological_folds(clean_train, n_splits=10, date_col=id_col)
save_folds_to_disk(folds, catalog["folds_dir"])

# %% [markdown]
# ## Custom Sharpe feval
# The Kaggle metric is the Sharpe-style ratio of ``prediction * label``. Use the provided callbacks during model training for direct leaderboard-aligned feedback.
# 

# %%
example_sharpe = hull_sharpe_ratio(
    clean_train[target_col].values,
    clean_train[target_col].shift(1).fillna(0).values,
)
print(f"Example Hull Sharpe (target vs lagged target): {example_sharpe:.6f}")

lightgbm_fit_kwargs = {
    "eval_metric": [hull_sharpe_lightgbm, "rmse"],
}

xgboost_eval_metric = hull_sharpe_xgboost  # pass via XGBRegressor(eval_metric=...)

# %% [markdown]
# ## Simple and nested ensembles
# Utility functions for combining base model predictions and emitting a submission ready for Kaggle.
# 

# %%
from typing import Dict, Iterable


def simple_mean_ensemble(predictions: Dict[str, Iterable[float]]) -> np.ndarray:
    """Equal-weight average across model prediction arrays."""
    stacked = np.column_stack(list(predictions.values()))
    return stacked.mean(axis=1)


def nested_ensemble(group_predictions: Dict[str, Dict[str, Iterable[float]]],
                    group_weights: Dict[str, float] | None = None) -> np.ndarray:
    """Two-level ensemble: average within groups, then weighted average across groups."""
    group_weights = group_weights or {}
    group_outputs = []
    weights = []

    for group, preds in group_predictions.items():
        group_mean = simple_mean_ensemble(preds)
        weight = float(group_weights.get(group, 1.0))
        group_outputs.append(group_mean * weight)
        weights.append(weight)

    weight_total = np.sum(weights) if weights else 1.0
    return np.sum(group_outputs, axis=0) / weight_total


def predict_submission(test_df: pd.DataFrame,
                       feature_cols: Iterable[str],
                       base_model_preds: Dict[str, Iterable[float]],
                       nested_groups: Dict[str, Dict[str, Iterable[float]]] | None = None,
                       id_column: str = id_col,
                       submission_path: Path | None = None) -> pd.DataFrame:
    """Create a submission using either a simple mean or nested ensemble."""
    if nested_groups:
        predictions = nested_ensemble(nested_groups)
    else:
        predictions = simple_mean_ensemble(base_model_preds)

    submission = pd.DataFrame({
        id_column: test_df[id_column].values,
        "prediction": predictions,
    })

    submission_path = submission_path or catalog.get("submission", Path("submission.csv"))
    save_table(submission, submission_path)
    print(f"Submission saved -> {submission_path}")
    return submission

# %%



