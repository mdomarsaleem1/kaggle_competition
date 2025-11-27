# %% [markdown]
# # Chronos-2 iterative folds tutorial
#         Load the cleaned Hull Tactical data, iterate over saved chronological folds, and score Chronos-2 with the competition Sharpe metric.
# 

# %%
from pathlib import Path
import sys
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

# Project imports
project_root = Path('..').resolve()
sys.path.insert(0, str(project_root))
from utils import default_catalog
from utils.catalog import load_table
from utils.metrics import hull_sharpe_ratio
from models import ChronosTimeSeriesModel

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# %% [markdown]
# ## Load cleaned data and folds
#         We consume the already-cleaned artifacts produced by `00_data_cleaning.ipynb` and the chronological folds written to `data/02_interim/folds`.
# 

# %%
catalog = default_catalog()

clean_train_path = catalog["clean_train"]
clean_test_path = clean_train_path.with_name("clean_test.parquet")
folds_dir = catalog["folds_dir"]
submission_path = catalog.get("submission", Path("submission.csv"))

clean_train = load_table(clean_train_path)
clean_test = load_table(clean_test_path)

print(f"Clean train shape: {clean_train.shape}")
print(f"Clean test shape:  {clean_test.shape}")
print(f"Folds dir: {folds_dir}")


# %%
from pathlib import Path

def load_folds(folds_dir: Path):
    folds = []
    train_files = sorted(Path(folds_dir).glob("train_fold_*.csv"))
    for train_file in train_files:
        fold_id = int(train_file.stem.split('_')[-1])
        test_file = train_file.parent / f"test_fold_{fold_id}.csv"
        if not test_file.exists():
            continue
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        folds.append((fold_id, train_df, test_df))
    return folds

folds = load_folds(folds_dir)
print(f"Loaded {len(folds)} folds")


# %% [markdown]
# ## Helper: evaluate Chronos across folds
#         For each fold, we take the most recent `context_length` points from the training slice, forecast the full test window, and compute the Hull Sharpe metric plus MAE/RMSE.
# 

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error

context_length = 64
MIN_INVESTMENT, MAX_INVESTMENT = 0,2

def score_fold(model: ChronosTimeSeriesModel, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> Dict:
    series = train_df[target_col].values
    context = series[-context_length:]
    prediction_length = len(test_df)

    forecasts = model.predict(
        context=context,
        prediction_length=prediction_length,
        num_samples=20,
    )
    # predict returns samples x horizon; use median path
    median_forecast = np.median(forecasts, axis=1).reshape(-1)

    y_true = test_df[target_col].values
    
    # Convert forecasts to positions (0-2 range)
    # Option 1: Clip predictions to valid range if they're already meant to be positions
    positions = np.clip(median_forecast, MIN_INVESTMENT, MAX_INVESTMENT)
        
    print(y_true.shape, forecasts.shape, median_forecast.shape)
    
    # Use the updated metric with optional risk_free_rate if available
    if 'lagged_risk_free_rate' in test_df.columns and 'lagged_forward_returns' in test_df.columns:
        sharpe = hull_sharpe_ratio(
            y_true=test_df['lagged_forward_returns'].values,
            y_pred=positions,
            risk_free_rate=test_df['lagged_risk_free_rate'].values,
            forward_returns=test_df['lagged_forward_returns'].values
        )
    else:
        # Simplified version: assumes y_true are forward returns, risk_free_rate=0
        sharpe = hull_sharpe_ratio(y_true, positions)
    
    mae = mean_absolute_error(y_true, median_forecast)
    rmse = mean_squared_error(y_true, median_forecast)

    return {
        "sharpe": sharpe,
        "mae": mae,
        "rmse": rmse,
        "prediction": median_forecast,
        "positions": positions,  # Also return the positions used for Sharpe
    }

# %% [markdown]
# ## Load Chronos-2 model
#         Chronos-2 is pre-trained; we only load weights and reuse them for each fold.
# 

# %%
chronos = ChronosTimeSeriesModel(model_size='small', device=device)
chronos.load_model()

# %% [markdown]
# ## Iterate over folds
#         Evaluate Chronos on each fold and aggregate metrics.
# 

# %%
results = []
all_predictions = []

for fold_id, train_df, test_df in folds:
    metrics = score_fold(chronos, train_df, test_df, target_col='market_forward_excess_returns')
    results.append({
        "fold": fold_id,
        **{k: v for k, v in metrics.items() if k != "prediction"},
    })
    all_predictions.append(pd.DataFrame({
        "fold": fold_id,
        "date_id": test_df["date_id"].values,
        "prediction": metrics["prediction"],
        "actual": test_df["market_forward_excess_returns"].values,
    }))
    print(f"Fold {fold_id}: Sharpe={metrics['sharpe']:.4f}, MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}")

results_df = pd.DataFrame(results).sort_values('fold')
predictions_df = pd.concat(all_predictions, ignore_index=True)

print("Aggregate metrics:")
print(results_df[['fold', 'sharpe', 'mae', 'rmse']])
print(f"Mean Sharpe: {results_df['sharpe'].mean():.4f}")


# %% [markdown]
# ## Fit on full training data and forecast test set
#         Use all cleaned training targets as context and produce a submission-ready forecast for the provided test horizon.
# 

# %%
full_context = clean_train['market_forward_excess_returns'].values
prediction_length = len(clean_test)

full_forecasts = chronos.predict(
    context=full_context[-context_length:],
    prediction_length=prediction_length,
    num_samples=40,
)
full_pred = np.median(full_forecasts, axis=1).reshape(-1)

submission = pd.DataFrame({ "date_id": clean_test["date_id"].values,
    "prediction": full_pred,
})

submission_path.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(submission_path, index=False)

print(f"Saved submission to {submission_path}")
print(submission.head())


# %% [markdown]
# ## Save evaluation artifacts
#         Persist fold metrics and predictions for downstream analysis.
# 

# %%
out_dir = Path("../data/03_processed")
out_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(out_dir / "chronos_fold_metrics.csv", index=False)
predictions_df.to_csv(out_dir / "chronos_fold_predictions.csv", index=False)

print("Artifacts saved:")
print(out_dir / "chronos_fold_metrics.csv")
print(out_dir / "chronos_fold_predictions.csv")


# %%



