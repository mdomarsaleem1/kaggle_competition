"""Single-file training and inference script with no intra-project imports.

Workflow:
- Load raw train/test from ``data/01_raw``.
-(Optionally) Align schemas and basic imputation.
- Tune LightGBM/XGBoost/CatBoost with a small grid using Sharpe-aware scoring.
- Derive ensemble weights from out-of-fold Sharpe scores and model correlations.
- Fit full models and expose a simple `predict_rows` helper to get signals.
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

import time

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

#import kaggle_evaluation.default_inference_server

DATA_DIR = os.path.join(project_root , "data/01_raw")
print(DATA_DIR)
ID_COL = "date_id"
TARGET_COL = "market_forward_excess_returns"


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
    corr, _ = spearmanr(y_true, y_pred)
    return float(-1.0 if np.isnan(corr) else corr)


def blend_correlation_sharpe(spearman_score: float, sharpe_score: float) -> float:
    """Adaptive blend giving Sharpe more weight when correlation is strong."""
    spearman_norm = (spearman_score + 1.0) / 2.0  # -> [0,1]
    sharpe_weight = 0.2 + 0.6 * spearman_norm
    return (1.0 - sharpe_weight) * spearman_score + sharpe_weight * sharpe_score


def convert_ret_to_signal(ret: float) -> int:
    """Map continuous return predictions to discrete signals (0,1,2)."""
    arr = np.array([ret], dtype=float)
    q75 = max(0.0, float(np.quantile(arr, 0.75)))
    if ret <= 0:
        return 0
    if ret <= q75:
        return 1
    return 2


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lightweight RMSE helper."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"[{func.__name__}] start")
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[{func.__name__}] done in {elapsed:.1f}s")
        return result

    return wrapper


# --------------------------------------------------------------------------- #
# Data loading / cleaning                                                     #
# --------------------------------------------------------------------------- #
def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(DATA_DIR , "train.csv")  
    test_path = os.path.join(DATA_DIR , "test.csv")   
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def align_and_impute(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Align train/test columns, backfill lagged features, and ffill/median impute."""
    lag_map = {
        "lagged_forward_returns": "forward_returns",
        "lagged_risk_free_rate": "risk_free_rate",
        "lagged_market_forward_excess_returns": TARGET_COL,
    }
    for df in (train_df, test_df):
        df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce").astype(int)
    train_df = train_df.sort_values(ID_COL)
    test_df = test_df.sort_values(ID_COL)

    test_feature_cols = [c for c in test_df.columns if c not in {"is_scored"}]
    feature_cols = [c for c in test_feature_cols if c not in {TARGET_COL, ID_COL}]

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
def cross_val_predictions(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_factory,
    params: Dict,
    n_splits: int = 5,
) -> Tuple[np.ndarray, float]:
    df_sorted = df.sort_values(ID_COL).reset_index(drop=True)
    y = df_sorted[TARGET_COL].values
    oof = np.zeros(len(df_sorted))
    scores: List[float] = []
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
        oof[val_idx] = preds
        scores.append(blend_correlation_sharpe(safe_spearman(y_val, preds), sharpe_eval_slice(df_sorted.iloc[val_idx], preds)))
    return oof, float(np.mean(scores))


def tune_lightgbm(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
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
def main():
    train_df, test_df = load_raw()
    clean_train, clean_test, feature_cols = align_and_impute(train_df, test_df)
    print(f"Clean train: {clean_train.shape}, clean test: {clean_test.shape}, features: {len(feature_cols)}")

    # Exclude evaluation-only columns from features
    feature_cols = [c for c in feature_cols if c not in {"forward_returns", "risk_free_rate"}]

    best_lgb = tune_lightgbm(clean_train, feature_cols)
    best_xgb = tune_xgboost(clean_train, feature_cols)
    best_cb = tune_catboost(clean_train, feature_cols)
    print("Best params -> LGBM:", best_lgb)
    print("Best params -> XGB:", best_xgb)
    print("Best params -> CatBoost:", best_cb)

    # OOF for weighting
    lgb_oof, lgb_sharpe = cross_val_predictions(
        clean_train, feature_cols, lambda p: LGBMRegressor(**p), best_lgb
    )
    xgb_oof, xgb_sharpe = cross_val_predictions(
        clean_train, feature_cols, lambda p: XGBRegressor(**p, objective="reg:squarederror"), best_xgb
    )
    cb_oof = np.zeros(len(clean_train))
    cb_scores = []
    sorted_df = clean_train.sort_values(ID_COL).reset_index(drop=True)
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
        cb_oof[val_idx] = preds
        cb_scores.append(blend_correlation_sharpe(safe_spearman(y_val, preds), sharpe_eval_slice(sorted_df.iloc[val_idx], preds)))
    cb_sharpe = float(np.mean(cb_scores))

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
    print(f"Sharpe -> LGBM: {lgb_sharpe:.3f}, XGB: {xgb_sharpe:.3f}, Cat: {cb_sharpe:.3f}")
    print(f"Weights -> LGBM: {lgb_w:.3f}, XGB: {xgb_w:.3f}, Cat: {cb_w:.3f}")

    # Fit full models
    scaler_lgb = StandardScaler()
    scaler_xgb = StandardScaler()
    X_train = clean_train[feature_cols].values
    y_train = clean_train[TARGET_COL].values
    X_train_lgb = scaler_lgb.fit_transform(X_train)
    X_train_xgb = scaler_xgb.fit_transform(X_train)

    lgb_full = LGBMRegressor(**best_lgb).fit(X_train_lgb, y_train)
    xgb_full = XGBRegressor(**best_xgb, objective="reg:squarederror").fit(X_train_xgb, y_train)
    cb_full = CatBoostRegressor(**best_cb).fit(clean_train[feature_cols], y_train, verbose=False)

    # Example inference on test set (row-wise)
    predictions = []
    for _, row in clean_test.sort_values(ID_COL).iterrows():
        X_row = row[feature_cols].values.reshape(1, -1)
        pred_l = lgb_full.predict(scaler_lgb.transform(X_row))
        pred_x = xgb_full.predict(scaler_xgb.transform(X_row))
        pred_c = cb_full.predict(pd.DataFrame(X_row, columns=feature_cols))
        ensemble_pred = float(lgb_w * pred_l + xgb_w * pred_x + cb_w * pred_c)
        predictions.append(convert_ret_to_signal(ensemble_pred))

    submission = pd.DataFrame({ID_COL: clean_test[ID_COL].values, "prediction": predictions})
    submission_path = Path("data") / "04_models" / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")

    # Inference server style predict function (row-wise)
    # def predict_rows(df_like) -> float:
    #     if hasattr(df_like, "to_pandas"):
    #         df_in = df_like.to_pandas()
    #     else:
    #         df_in = pd.DataFrame(df_like)
    #     df_in = df_in.sort_values(ID_COL)
    #     row = df_in.iloc[0]
    #     X_row = row[feature_cols].values.reshape(1, -1)
    #     pred_l = lgb_full.predict(scaler_lgb.transform(X_row))
    #     pred_x = xgb_full.predict(scaler_xgb.transform(X_row))
    #     pred_c = cb_full.predict(pd.DataFrame(X_row, columns=feature_cols))
    #     blended = float(lgb_w * pred_l + xgb_w * pred_x + cb_w * pred_c)
    #     return float(convert_ret_to_signal(blended))

    # inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict_rows)
    # if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    #     inference_server.serve()
    # else:
    #     inference_server.run_local_gateway((str(Path(DATA_DIR).parent),))


if __name__ == "__main__":
    main()
