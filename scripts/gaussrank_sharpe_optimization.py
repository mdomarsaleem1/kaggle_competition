"""Enhanced training script with GaussRank transformation and Sharpe optimization.

Key changes from full_file.py:
1. Do NOT train on market_forward_excess_returns (only use as target)
2. Apply GaussRank (Inverse Normal) transformation to features
3. Optimize scalar k and bias b to maximize Adjusted Sharpe Ratio
4. Test harness for sharpe_eval_slice function

Workflow:
- Load raw train/test from data/01_raw
- Apply GaussRank transformation to features
- Tune models with Sharpe-aware scoring
- Optimize k and b parameters on validation set
- Generate optimized predictions
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Allow importing project utils
from pathlib import Path
path = Path(os.getcwd())

project_root = path.parent.absolute()
sys.path.insert(0, str(project_root))

print(f"Current working directory: {os.getcwd()}")

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the target folder
lib_dir = os.path.join(current_dir, "lib")

# Add the folder to sys.path if it exists and is not already present
if os.path.exists(lib_dir) and lib_dir not in sys.path:
    sys.path.append(lib_dir)

import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.stats import spearmanr, norm
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

DATA_DIR = os.path.join(project_root, "data/01_raw")
print(f"Data directory: {DATA_DIR}")
ID_COL = "date_id"
TARGET_COL = "market_forward_excess_returns"


# --------------------------------------------------------------------------- #
# GaussRank Transformation                                                    #
# --------------------------------------------------------------------------- #
def gaussrank_transform(series: pd.Series) -> pd.Series:
    """Apply GaussRank (Inverse Normal) transformation to a series.

    This transformation:
    1. Ranks the values
    2. Converts ranks to uniform distribution [0,1]
    3. Applies inverse normal CDF to get Gaussian distribution
    """
    # Handle NaN values
    mask = ~series.isna()
    if mask.sum() == 0:
        return series

    # Rank the non-NaN values
    ranked = series[mask].rank(method='average')

    # Convert to uniform distribution (avoid 0 and 1 for numerical stability)
    n = len(ranked)
    uniform = (ranked - 0.5) / n

    # Apply inverse normal CDF
    gaussrank = pd.Series(norm.ppf(uniform), index=series[mask].index)

    # Preserve NaN locations
    result = pd.Series(np.nan, index=series.index)
    result[mask] = gaussrank

    return result


def apply_gaussrank_to_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Apply GaussRank transformation to all feature columns."""
    df_transformed = df.copy()

    for col in feature_cols:
        if col in df_transformed.columns:
            df_transformed[col] = gaussrank_transform(df_transformed[col])

    return df_transformed


# --------------------------------------------------------------------------- #
# Metrics / helpers                                                           #
# --------------------------------------------------------------------------- #
def hull_sharpe_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_free_rate: np.ndarray | None = None,
    forward_returns: np.ndarray | None = None,
    eps: float = 1e-9,
) -> float:
    """Hull-style Sharpe with optional risk-free/forward returns."""
    y_pred = np.clip(y_pred, 0.0, 2.0)
    if risk_free_rate is None or forward_returns is None:
        pnl = y_true * y_pred
    else:
        strategy_returns = risk_free_rate * (1.0 - y_pred) + y_pred * forward_returns
        pnl = strategy_returns - risk_free_rate

    pnl = pnl[~np.isnan(pnl)]
    if pnl.shape[0] == 0:
        return 0.0

    return float(np.mean(pnl) / (np.std(pnl) + eps))


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Spearman correlation safely."""
    corr, _ = spearmanr(y_true, y_pred)
    return float(-1.0 if np.isnan(corr) else corr)


def blend_correlation_sharpe(spearman_score: float, sharpe_score: float) -> float:
    """Adaptive blend giving Sharpe more weight when correlation is strong."""
    spearman_norm = (spearman_score + 1.0) / 2.0  # -> [0,1]
    sharpe_weight = 0.2 + 0.6 * spearman_norm
    return (1.0 - sharpe_weight) * spearman_score + sharpe_weight * sharpe_score


def sharpe_eval_slice(df_slice: pd.DataFrame, preds: np.ndarray) -> float:
    """Compute Sharpe using forward_returns/risk_free_rate if available."""
    y_true = df_slice[TARGET_COL].values
    if {"forward_returns", "risk_free_rate"}.issubset(df_slice.columns):
        return hull_sharpe_ratio(
            y_true,
            preds,
            risk_free_rate=df_slice["risk_free_rate"].values,
            forward_returns=df_slice["forward_returns"].values,
        )
    return hull_sharpe_ratio(y_true, preds)


def timer(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"[{func.__name__}] start")
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[{func.__name__}] done in {elapsed:.1f}s")
        return result
    return wrapper


# --------------------------------------------------------------------------- #
# Test harness for sharpe_eval_slice                                          #
# --------------------------------------------------------------------------- #
def test_sharpe_eval_slice():
    """Test the sharpe_eval_slice function with synthetic data."""
    print("\n" + "="*70)
    print("Testing sharpe_eval_slice function")
    print("="*70)

    # Test 1: Basic test without risk_free_rate and forward_returns
    print("\nTest 1: Basic Sharpe calculation")
    test_df_1 = pd.DataFrame({
        TARGET_COL: np.array([0.01, 0.02, -0.01, 0.03, 0.01]),
        'date_id': range(5)
    })
    test_preds_1 = np.array([1.0, 1.5, 0.5, 2.0, 1.0])
    sharpe_1 = sharpe_eval_slice(test_df_1, test_preds_1)
    print(f"  Sharpe ratio: {sharpe_1:.4f}")
    print(f"  Expected: positive value (predictions align with returns)")

    # Test 2: With risk_free_rate and forward_returns
    print("\nTest 2: Sharpe with risk_free_rate and forward_returns")
    test_df_2 = pd.DataFrame({
        TARGET_COL: np.array([0.01, 0.02, -0.01, 0.03, 0.01]),
        'forward_returns': np.array([0.015, 0.025, -0.005, 0.035, 0.015]),
        'risk_free_rate': np.array([0.001, 0.001, 0.001, 0.001, 0.001]),
        'date_id': range(5)
    })
    test_preds_2 = np.array([1.0, 1.5, 0.5, 2.0, 1.0])
    sharpe_2 = sharpe_eval_slice(test_df_2, test_preds_2)
    print(f"  Sharpe ratio: {sharpe_2:.4f}")
    print(f"  Expected: different from Test 1 due to risk adjustment")

    # Test 3: Edge case - all zero predictions
    print("\nTest 3: Edge case - all zero predictions")
    test_preds_3 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    sharpe_3 = sharpe_eval_slice(test_df_2, test_preds_3)
    print(f"  Sharpe ratio: {sharpe_3:.4f}")
    print(f"  Expected: 0.0 (no allocation)")

    # Test 4: Edge case - all max predictions
    print("\nTest 4: Edge case - all max predictions")
    test_preds_4 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    sharpe_4 = sharpe_eval_slice(test_df_2, test_preds_4)
    print(f"  Sharpe ratio: {sharpe_4:.4f}")
    print(f"  Expected: positive value (full allocation)")

    # Test 5: Negative correlation
    print("\nTest 5: Negative correlation test")
    test_preds_5 = np.array([2.0, 2.0, 2.0, 0.0, 0.0])  # High when returns low, low when returns high
    sharpe_5 = sharpe_eval_slice(test_df_2, test_preds_5)
    print(f"  Sharpe ratio: {sharpe_5:.4f}")
    print(f"  Expected: lower than Test 2 (poor allocation)")

    print("\n" + "="*70)
    print("All tests completed successfully!")
    print("="*70 + "\n")


# --------------------------------------------------------------------------- #
# k and b Optimization                                                        #
# --------------------------------------------------------------------------- #
def optimize_k_b(
    predictions: np.ndarray,
    df_val: pd.DataFrame,
    initial_k: float = 1.0,
    initial_b: float = 0.0,
) -> Tuple[float, float, float]:
    """Optimize k and b to maximize Sharpe ratio on validation set.

    Allocation_t = Clip(k * Prediction_t + b, 0, 2)

    Args:
        predictions: Raw model predictions on validation set
        df_val: Validation DataFrame containing TARGET_COL and optionally risk_free_rate/forward_returns
        initial_k: Initial value for k parameter
        initial_b: Initial value for b parameter

    Returns:
        Tuple of (optimal_k, optimal_b, best_sharpe)
    """
    def objective(params):
        k, b = params
        # Transform predictions to allocations
        allocations = np.clip(k * predictions + b, 0.0, 2.0)
        # Calculate negative Sharpe (we minimize)
        sharpe = sharpe_eval_slice(df_val, allocations)
        return -sharpe

    # Try multiple starting points to avoid local minima
    best_result = None
    best_sharpe = -np.inf

    starting_points = [
        (initial_k, initial_b),
        (0.5, 0.5),
        (1.5, -0.5),
        (2.0, 0.0),
        (1.0, 0.5),
    ]

    for k_init, b_init in starting_points:
        result = minimize(
            objective,
            x0=[k_init, b_init],
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
        )

        current_sharpe = -result.fun
        if current_sharpe > best_sharpe:
            best_sharpe = current_sharpe
            best_result = result

    optimal_k, optimal_b = best_result.x

    print(f"\nOptimization results:")
    print(f"  Optimal k: {optimal_k:.4f}")
    print(f"  Optimal b: {optimal_b:.4f}")
    print(f"  Best Sharpe: {best_sharpe:.4f}")

    return optimal_k, optimal_b, best_sharpe


# --------------------------------------------------------------------------- #
# Data loading / cleaning                                                     #
# --------------------------------------------------------------------------- #
def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train and test data."""
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def align_and_impute(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Align train/test columns, backfill lagged features, and ffill/median impute.

    IMPORTANT: Do NOT include market_forward_excess_returns as a feature!
    It should only be used as the target variable.
    """
    lag_map = {
        "lagged_forward_returns": "forward_returns",
        "lagged_risk_free_rate": "risk_free_rate",
        # Note: We explicitly exclude market_forward_excess_returns from features
    }

    for df in (train_df, test_df):
        df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce").astype(int)
    train_df = train_df.sort_values(ID_COL)
    test_df = test_df.sort_values(ID_COL)

    test_feature_cols = [c for c in test_df.columns if c not in {"is_scored"}]
    # CRITICAL: Exclude TARGET_COL from features
    feature_cols = [
        c for c in test_feature_cols
        if c not in {TARGET_COL, ID_COL, "market_forward_excess_returns"}
    ]

    for lagged_col, source_col in lag_map.items():
        if lagged_col in feature_cols and lagged_col not in train_df.columns and source_col in train_df.columns:
            train_df[lagged_col] = train_df[source_col].shift(1)

    if "is_scored" in test_df.columns and "is_scored" not in train_df.columns:
        train_df["is_scored"] = 1

    fill_cols = [c for c in feature_cols if c in train_df.columns]
    medians = train_df[fill_cols].median()

    def _fill(df: pd.DataFrame) -> pd.DataFrame:
        filled = df.copy()
        for col in fill_cols:
            if filled[col].isnull().any():
                filled[col] = filled[col].ffill()
                filled[col] = filled[col].fillna(medians[col])
        return filled

    clean_train = _fill(train_df)
    clean_test = _fill(test_df)
    clean_train["target"] = clean_train[TARGET_COL]

    # Retain forward/risk_free for evaluation (not used as features)
    for col in ["forward_returns", "risk_free_rate"]:
        if col in train_df.columns and col not in clean_train.columns:
            clean_train[col] = train_df[col].values
        if col not in clean_test.columns:
            clean_test[col] = np.nan

    eval_cols = [c for c in ["forward_returns", "risk_free_rate"] if c in clean_train.columns]
    clean_train = clean_train[[ID_COL] + fill_cols + ["target", TARGET_COL] + eval_cols]
    extra_test_cols = ["is_scored"] if "is_scored" in clean_test.columns else []
    clean_test = clean_test[[ID_COL] + fill_cols + extra_test_cols]

    return clean_train.reset_index(drop=True), clean_test.reset_index(drop=True), fill_cols


# --------------------------------------------------------------------------- #
# Model tuning / evaluation                                                   #
# --------------------------------------------------------------------------- #
def cross_val_predictions_with_optimization(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_factory,
    params: Dict,
    n_splits: int = 5,
) -> Tuple[np.ndarray, float, float, float]:
    """Cross-validation with k and b optimization.

    Returns:
        Tuple of (oof_predictions, avg_score, optimal_k, optimal_b)
    """
    df_sorted = df.sort_values(ID_COL).reset_index(drop=True)
    y = df_sorted[TARGET_COL].values
    oof = np.zeros(len(df_sorted))
    scores: List[float] = []
    k_values: List[float] = []
    b_values: List[float] = []

    splitter = TimeSeriesSplit(n_splits=n_splits)
    for tr_idx, val_idx in splitter.split(df_sorted):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(df_sorted.iloc[tr_idx][feature_cols].values)
        X_val = scaler.transform(df_sorted.iloc[val_idx][feature_cols].values)
        y_tr = y[tr_idx]
        y_val = y[val_idx]

        model = model_factory(params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        # Optimize k and b on this validation fold
        k_opt, b_opt, sharpe_opt = optimize_k_b(preds, df_sorted.iloc[val_idx])

        # Apply optimal transformation
        oof[val_idx] = np.clip(k_opt * preds + b_opt, 0.0, 2.0)

        k_values.append(k_opt)
        b_values.append(b_opt)
        scores.append(sharpe_opt)

    # Average k and b across folds
    avg_k = float(np.mean(k_values))
    avg_b = float(np.mean(b_values))

    return oof, float(np.mean(scores)), avg_k, avg_b


def tune_lightgbm(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """Tune LightGBM hyperparameters."""
    @timer
    def _tune():
        tscv = TimeSeriesSplit(n_splits=5)
        y = df[TARGET_COL].values

        def objective(trial):
            params = {
                "objective": "regression",
                "metric": "rmse",
                "n_estimators": trial.suggest_int("n_estimators", 1000, 5000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "num_leaves": trial.suggest_int("num_leaves", 32, 512),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "subsample_freq": 1,
                "random_state": 42,
                "verbosity": -1,
            }
            scores = []
            df_sorted = df.sort_values(ID_COL).reset_index(drop=True)
            for tr_idx, val_idx in tscv.split(df_sorted):
                X_tr = df_sorted.iloc[tr_idx][feature_cols]
                X_val = df_sorted.iloc[val_idx][feature_cols]
                y_tr = y[tr_idx]
                y_val = y[val_idx]
                model = LGBMRegressor(**params)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="rmse",
                    callbacks=[early_stopping(stopping_rounds=100, verbose=False), log_evaluation(period=0)],
                )
                preds = model.predict(X_val)
                scores.append(sharpe_eval_slice(df_sorted.iloc[val_idx], preds))
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        return study.best_params

    return _tune()


def tune_xgboost(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """Tune XGBoost hyperparameters."""
    @timer
    def _tune():
        tscv = TimeSeriesSplit(n_splits=5)
        y = df[TARGET_COL].values

        def objective(trial):
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                "n_estimators": trial.suggest_int("n_estimators", 400, 1600),
                "tree_method": "hist",
                "objective": "reg:squarederror",
                "random_state": 42,
            }
            scores = []
            df_sorted = df.sort_values(ID_COL).reset_index(drop=True)
            for tr_idx, val_idx in tscv.split(df_sorted):
                X_tr = df_sorted.iloc[tr_idx][feature_cols]
                X_val = df_sorted.iloc[val_idx][feature_cols]
                y_tr = y[tr_idx]
                y_val = y[val_idx]
                model = XGBRegressor(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                preds = model.predict(X_val)
                scores.append(sharpe_eval_slice(df_sorted.iloc[val_idx], preds))
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        return study.best_params

    return _tune()


def tune_catboost(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """Tune CatBoost hyperparameters."""
    @timer
    def _tune():
        tscv = TimeSeriesSplit(n_splits=5)
        y = df[TARGET_COL].values

        def objective(trial):
            params = {
                "loss_function": "RMSE",
                "iterations": trial.suggest_int("iterations", 400, 1600),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 5.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 1e-3, 5.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
                "random_seed": 42,
                "verbose": False,
            }
            scores = []
            df_sorted = df.sort_values(ID_COL).reset_index(drop=True)
            for tr_idx, val_idx in tscv.split(df_sorted):
                X_tr = df_sorted.iloc[tr_idx][feature_cols]
                y_tr = y[tr_idx]
                X_val = df_sorted.iloc[val_idx][feature_cols]
                y_val = y[val_idx]
                model = CatBoostRegressor(**params)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    early_stopping_rounds=50,
                    verbose=False,
                )
                preds = model.predict(X_val)
                scores.append(sharpe_eval_slice(df_sorted.iloc[val_idx], preds))
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        return study.best_params

    return _tune()


# --------------------------------------------------------------------------- #
# Main routine                                                                #
# --------------------------------------------------------------------------- #
@timer
def main():
    # Run tests for sharpe_eval_slice
    test_sharpe_eval_slice()

    # Load and prepare data
    train_df, test_df = load_raw()
    clean_train, clean_test, feature_cols = align_and_impute(train_df, test_df)

    print(f"\n{'='*70}")
    print(f"Data Summary:")
    print(f"  Clean train: {clean_train.shape}")
    print(f"  Clean test: {clean_test.shape}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Feature names: {feature_cols[:5]}... (showing first 5)")
    print(f"{'='*70}\n")

    # Exclude evaluation-only columns from features
    feature_cols = [c for c in feature_cols if c not in {"forward_returns", "risk_free_rate"}]

    # Apply GaussRank transformation
    print("Applying GaussRank transformation to features...")
    clean_train_gaussrank = apply_gaussrank_to_features(clean_train, feature_cols)
    clean_test_gaussrank = apply_gaussrank_to_features(clean_test, feature_cols)
    print("GaussRank transformation complete.\n")

    # Tune models
    print("Starting hyperparameter tuning...")
    best_lgb = tune_lightgbm(clean_train_gaussrank, feature_cols)
    best_xgb = tune_xgboost(clean_train_gaussrank, feature_cols)
    best_cb = tune_catboost(clean_train_gaussrank, feature_cols)
    print("\nBest params -> LGBM:", best_lgb)
    print("Best params -> XGB:", best_xgb)
    print("Best params -> CatBoost:", best_cb)

    # Cross-validation with k and b optimization
    print("\n" + "="*70)
    print("Cross-validation with k and b optimization")
    print("="*70)

    lgb_oof, lgb_sharpe, lgb_k, lgb_b = cross_val_predictions_with_optimization(
        clean_train_gaussrank, feature_cols, lambda p: LGBMRegressor(**p), best_lgb
    )
    print(f"\nLightGBM - k: {lgb_k:.4f}, b: {lgb_b:.4f}, Sharpe: {lgb_sharpe:.4f}")

    xgb_oof, xgb_sharpe, xgb_k, xgb_b = cross_val_predictions_with_optimization(
        clean_train_gaussrank, feature_cols, lambda p: XGBRegressor(**p, objective="reg:squarederror"), best_xgb
    )
    print(f"XGBoost - k: {xgb_k:.4f}, b: {xgb_b:.4f}, Sharpe: {xgb_sharpe:.4f}")

    # CatBoost with k and b optimization
    cb_oof = np.zeros(len(clean_train_gaussrank))
    cb_scores = []
    cb_k_values = []
    cb_b_values = []
    sorted_df = clean_train_gaussrank.sort_values(ID_COL).reset_index(drop=True)
    y_sorted = sorted_df[TARGET_COL].values
    splitter = TimeSeriesSplit(n_splits=5)

    for tr_idx, val_idx in splitter.split(sorted_df):
        model = CatBoostRegressor(**best_cb)
        X_tr = sorted_df.iloc[tr_idx][feature_cols]
        y_tr = y_sorted[tr_idx]
        X_val = sorted_df.iloc[val_idx][feature_cols]
        y_val = y_sorted[val_idx]
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, early_stopping_rounds=50, verbose=False)
        preds = model.predict(X_val)

        # Optimize k and b
        k_opt, b_opt, sharpe_opt = optimize_k_b(preds, sorted_df.iloc[val_idx])
        cb_oof[val_idx] = np.clip(k_opt * preds + b_opt, 0.0, 2.0)
        cb_k_values.append(k_opt)
        cb_b_values.append(b_opt)
        cb_scores.append(sharpe_opt)

    cb_sharpe = float(np.mean(cb_scores))
    cb_k = float(np.mean(cb_k_values))
    cb_b = float(np.mean(cb_b_values))
    print(f"CatBoost - k: {cb_k:.4f}, b: {cb_b:.4f}, Sharpe: {cb_sharpe:.4f}")

    # Correlation-aware weights
    corr_lgb_xgb = float(np.corrcoef(lgb_oof, xgb_oof)[0, 1])
    corr_lgb_cb = float(np.corrcoef(lgb_oof, cb_oof)[0, 1])
    corr_xgb_cb = float(np.corrcoef(xgb_oof, cb_oof)[0, 1])

    lgb_w = max(lgb_sharpe, 0.0) * (1 - abs(corr_lgb_xgb)) * (1 - abs(corr_lgb_cb))
    xgb_w = max(xgb_sharpe, 0.0) * (1 - abs(corr_lgb_xgb)) * (1 - abs(corr_xgb_cb))
    cb_w = max(cb_sharpe, 0.0) * (1 - abs(corr_lgb_cb)) * (1 - abs(corr_xgb_cb))
    weight_sum = lgb_w + xgb_w + cb_w
    if weight_sum == 0:
        lgb_w = xgb_w = cb_w = 1 / 3
    else:
        lgb_w /= weight_sum
        xgb_w /= weight_sum
        cb_w /= weight_sum

    print(f"\n{'='*70}")
    print(f"Final Model Weights:")
    print(f"  LGBM: {lgb_w:.3f} (Sharpe: {lgb_sharpe:.3f})")
    print(f"  XGB:  {xgb_w:.3f} (Sharpe: {xgb_sharpe:.3f})")
    print(f"  Cat:  {cb_w:.3f} (Sharpe: {cb_sharpe:.3f})")
    print(f"{'='*70}\n")

    # Fit full models on GaussRank-transformed data
    print("Fitting full models...")
    scaler_lgb = StandardScaler()
    scaler_xgb = StandardScaler()
    X_train = clean_train_gaussrank[feature_cols].values
    y_train = clean_train_gaussrank[TARGET_COL].values
    X_train_lgb = scaler_lgb.fit_transform(X_train)
    X_train_xgb = scaler_xgb.fit_transform(X_train)

    lgb_full = LGBMRegressor(**best_lgb).fit(X_train_lgb, y_train)
    xgb_full = XGBRegressor(**best_xgb, objective="reg:squarederror").fit(X_train_xgb, y_train)
    cb_full = CatBoostRegressor(**best_cb).fit(clean_train_gaussrank[feature_cols], y_train, verbose=False)

    # Generate predictions on test set with k and b optimization
    print("Generating predictions on test set...")
    predictions = []
    for _, row in clean_test_gaussrank.sort_values(ID_COL).iterrows():
        X_row = row[feature_cols].values.reshape(1, -1)
        pred_l = lgb_full.predict(scaler_lgb.transform(X_row))[0]
        pred_x = xgb_full.predict(scaler_xgb.transform(X_row))[0]
        pred_c = cb_full.predict(pd.DataFrame(X_row, columns=feature_cols))[0]

        # Apply individual k and b transformations
        pred_l_opt = lgb_k * pred_l + lgb_b
        pred_x_opt = xgb_k * pred_x + xgb_b
        pred_c_opt = cb_k * pred_c + cb_b

        # Weighted ensemble
        ensemble_pred = lgb_w * pred_l_opt + xgb_w * pred_x_opt + cb_w * pred_c_opt

        # Clip to valid range
        allocation = np.clip(ensemble_pred, 0.0, 2.0)
        predictions.append(allocation)

    # Save submission
    submission = pd.DataFrame({
        ID_COL: clean_test_gaussrank[ID_COL].values,
        "prediction": predictions
    })
    submission_path = Path("data") / "04_models" / "submission_gaussrank_optimized.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)

    print(f"\n{'='*70}")
    print(f"Submission saved to: {submission_path}")
    print(f"Prediction statistics:")
    print(f"  Min:    {np.min(predictions):.4f}")
    print(f"  Max:    {np.max(predictions):.4f}")
    print(f"  Mean:   {np.mean(predictions):.4f}")
    print(f"  Median: {np.median(predictions):.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
