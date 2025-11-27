
# %% [markdown]
# # Stage-by-Stage Testing Notebook
# 
# This notebook walks through each step of the time-series workflow so you can validate changes incrementally:
# 
# 1. Load and inspect the training/test data.
# 2. Build **10 chronological cross-validation folds**.
# 3. Add a numeric date index alongside your features.
# 4. Train a lightweight baseline model fold-by-fold.
# 5. Aggregate validation metrics and train a final model for submission.
# 
# Configure the paths and columns in the **Configuration** cell, then execute the rest of the notebook top-to-bottom.
# 

# %% [markdown]
# ## Prerequisites
# 
# - Place your CSVs in the `data/` directory (or update the paths below).
# - Ensure dependencies from `requirements.txt` are installed: `pip install -r requirements.txt`.
# - If you want GPU acceleration for LightGBM/transformers, install the appropriate extras and set the device flags in the model constructors.
# 

# %%
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project importable when running the notebook from the notebooks/ folder
PROJECT_ROOT = Path('..').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)


# %% [markdown]
# ## 1) Configuration
# Update the file names and column names to match your dataset before running the rest of the notebook.
# 

# %%
DATA_DIR = Path('../data')
TRAIN_FILE = DATA_DIR / '01_raw/train.csv'   # update if your file name differs
TEST_FILE = DATA_DIR / '01_raw/test.csv'     # set to a CSV path if you have a test set
TARGET_COL = 'market_forward_excess_returns'                 # change to the name of your target column
DATE_COL = 'date_id'                     # change to your datetime column (or set to None)
N_SPLITS = 10                         # number of chronological folds
FOLD_OUTPUT = DATA_DIR / 'cv_folds'   # where to persist fold CSVs


# %% [markdown]
# ## 2) Load data
# 
# The helper below parses and sorts the date column so folds and models respect chronology.
# 

# %%
from utils.data_utils import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor(scaler_type=None)

train_df, test_df = preprocessor.load_data(TRAIN_FILE, TEST_FILE if TEST_FILE.exists() else None)
print(f"Train shape: {train_df.shape}")
if test_df is not None:
    print(f"Test shape:  {test_df.shape}")

train_df.head()


# %% [markdown]
# ## 3) Create chronological folds
# 
# The folds are time-ordered using `TimeSeriesSplit` and can be saved to disk for re-use across models.
# 

# %%
from utils.cross_validation import create_chronological_folds, save_folds_to_disk

folds = create_chronological_folds(train_df, n_splits=N_SPLITS, date_col=DATE_COL)
for fold in folds:
    print(f"Fold {fold.fold_id}: train {fold.train_range} | test {fold.test_range}")

# Persist the fold CSVs so other scripts can consume the same splits
save_folds_to_disk(folds, FOLD_OUTPUT)
print(f"Folds saved under: {FOLD_OUTPUT.resolve()}")


# %% [markdown]
# ## 4) Feature engineering with numeric date indices
# 
# The helper below converts the datetime column to an ordinal integer (days since year 1) so transformers and
# tree models consume a numeric representation instead of raw timestamps. Add your own domain features as needed.
# 

# %%
def add_date_ordinal(df: pd.DataFrame, date_col: str):
    df_out = df.copy()
    if date_col and date_col in df_out.columns:
        #df_out[date_col] = pd.to_datetime(df_out[date_col])
        #df_out['date_ordinal'] = df_out[date_col].dt.toordinal()
        df_out['date_ordinal'] = df_out[date_col].astype(int)
    return df_out


def build_feature_matrix(df: pd.DataFrame, target_col: str, date_col: str):
    # Return feature matrix, target array, and feature column names.
    df_feat = add_date_ordinal(df, date_col)
    feature_cols = [c for c in df_feat.columns if c not in {target_col, date_col}]
    X = df_feat[feature_cols].values
    y = df_feat[target_col].values if target_col in df_feat else None
    return X, y, feature_cols

# Quick sanity check on the transformed columns
sample_with_ordinal = add_date_ordinal(train_df.head(), DATE_COL)
sample_with_ordinal[[col for col in sample_with_ordinal.columns if 'date' in col]]


# %% [markdown]
# ## 5) Fold-by-fold training (LightGBM example)
# 
# This section trains a lightweight LightGBM regressor on each fold to validate the pipeline. Adjust the
# model or hyperparameters as needed for your experiments.
# 

# %%
from models import LightGBMTimeSeriesModel

fold_metrics = []

for fold in folds:
    X_train, y_train, feature_cols = build_feature_matrix(fold.train, TARGET_COL, DATE_COL)
    X_val, y_val, _ = build_feature_matrix(fold.test, TARGET_COL, DATE_COL)

    model = LightGBMTimeSeriesModel(
        params={
            'n_estimators': 250,
            'learning_rate': 0.05,
            'num_leaves': 63,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
        }
    )

    metrics = model.train(X_train, y_train, X_val, y_val, verbose=False)
    fold_metrics.append({
        'fold': fold.fold_id,
        'val_rmse': metrics.get('val_rmse'),
        'val_mae': metrics.get('val_mae'),
        'best_iteration': metrics.get('best_iteration')
    })

    print(
        f"Fold {fold.fold_id} → val_rmse: {metrics.get('val_rmse'):.5f}, "
        f"val_mae: {metrics.get('val_mae'):.5f}, best_iter: {metrics.get('best_iteration')}"
    )


# %% [markdown]
# ## 6) Aggregate validation metrics
# 
# Inspect the per-fold metrics and the average validation score. You can use these to compare different
# feature sets or models while keeping the folds fixed.
# 

# %%
metrics_df = pd.DataFrame(fold_metrics)

display(metrics_df)
print("Average val RMSE:", metrics_df['val_rmse'].mean())
print("Average val MAE: ", metrics_df['val_mae'].mean())

# %% [markdown]
# ## 7) Train on full data and generate submission-ready predictions
# 
# This cell refits the model on the entire training set (with the same numeric date features) and produces
# predictions for the test set. Adjust the submission schema to match the competition requirements.
# 

# %%
if test_df is None:
    print("No test file found. Skipping final training and prediction.")
else:
    X_full, y_full, feature_cols = build_feature_matrix(train_df, TARGET_COL, DATE_COL)
    X_test, _, _ = build_feature_matrix(test_df, TARGET_COL, DATE_COL)

    final_model = LightGBMTimeSeriesModel(
        params={
            'n_estimators': 400,
            'learning_rate': 0.03,
            'num_leaves': 127,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
        }
    )

    final_model.train(X_full, y_full, verbose=False)
    test_pred = final_model.predict(X_test)

    submission = pd.DataFrame({
        'row_id': np.arange(len(test_pred)),
        TARGET_COL: test_pred
    })

    submission_path = DATA_DIR / 'submission_stage_testing.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path.resolve()}")
    submission.head()


# %%



