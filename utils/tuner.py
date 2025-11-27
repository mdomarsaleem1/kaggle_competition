"""Hyperparameter tuner utilities with exploratory helpers."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from IPython.display import Markdown, display
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit

import joblib
from xgboost import XGBRegressor

from .metrics import hull_sharpe_ratio


def describe_dataset(df: pd.DataFrame) -> None:
    """Print summary statistics, memory usage, and descriptive stats."""
    display(Markdown("## 📋 Dataset Overview"))
    print("--- Basic Dimensions & Memory ---")
    num_rows, num_cols = df.shape
    print(f"**Shape (Rows, Columns):** ({num_rows:,}, {num_cols:,})")
    mem_usage = df.memory_usage(deep=True).sum()
    mem_gbs = mem_usage / (1024**2)
    print(f"**Total Memory Usage:** {mem_gbs:.2f} MB")

    print("\n--- Feature Data Types and Counts ---")
    dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
    dtype_counts.columns = ["Data_Type", "Count"]
    print(dtype_counts.to_markdown(index=False))

    display(Markdown("\n## 📊 Descriptive Statistics"))
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

    if numerical_cols:
        display(Markdown("### Numerical Features"))
        num_desc = df[numerical_cols].describe().T
        num_desc["IQR"] = num_desc["75%"] - num_desc["25%"]
        display(num_desc.style.format("{:,.2f}"))
        print(f"Found {len(numerical_cols)} numerical features.")

    if categorical_cols:
        display(Markdown("### Categorical / Object Features"))
        cat_desc = df[categorical_cols].describe().T
        display(cat_desc.style.format({"unique": "{:,}", "freq": "{:,}"}))
        print(f"Found {len(categorical_cols)} categorical/object features.")

    if datetime_cols:
        display(Markdown("### Datetime Features"))
        dt_desc = df[datetime_cols].describe().T
        display(dt_desc)
        print(f"Found {len(datetime_cols)} datetime features.")


def missing_duplicates_analysis(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Visualize missingness and duplicates."""
    print("--- 📊 Missing Data and Duplicates Analysis ---")
    missing_counts = df.isnull().sum()
    missing_summary = pd.DataFrame(
        {
            "Missing_Count": missing_counts,
            "Missing_Percent": 100 * missing_counts / len(df),
        }
    )
    missing_summary = missing_summary[missing_summary["Missing_Count"] > 0]
    missing_summary = missing_summary.sort_values("Missing_Count", ascending=False)

    num_duplicates = df.duplicated().sum()
    print(f"**Duplicate rows found:** {num_duplicates}")

    if missing_summary.empty:
        print("✅ **No missing values found** in the dataset.")
        return pd.DataFrame()

    print(f"\n**Total features with missing values:** {len(missing_summary)}")
    plot_data = missing_summary.head(top_n)

    plt.style.use("ggplot")
    plt.figure(figsize=(12, 6))
    sns.barplot(x=plot_data.index, y="Missing_Count", data=plot_data, palette="viridis")
    for i, count in enumerate(plot_data["Missing_Count"]):
        percent = plot_data["Missing_Percent"].iloc[i]
        plt.text(
            x=i,
            y=count + (df.shape[0] * 0.005),
            s=f"{percent:.1f}%",
            ha="center",
            fontsize=9,
        )
    plt.title(
        f"Top {min(top_n, len(missing_summary))} Features by Missing Values Count "
        f"(Total Rows: {len(df)})",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Feature Name", fontsize=12)
    plt.ylabel("Missing Count", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("\n**Missing Data Summary Table (Top 10):**")
    print(missing_summary.head(10).to_markdown(floatfmt=".2f"))
    return missing_summary


def detect_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 2.5,
    z_threshold: float = 3.0,
    cols: Optional[List[str]] = None,
    summary: bool = True,
):
    """Flag outliers via IQR or Z-score."""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if method not in {"iqr", "zscore"}:
        raise ValueError("method must be 'iqr' or 'zscore'")

    outlier_flags = pd.DataFrame(False, index=df.index, columns=cols)
    for col in cols:
        series = df[col].dropna()
        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outlier_flags[col] = (df[col] < lower) | (df[col] > upper)
        else:
            z_scores = np.abs(stats.zscore(series))
            outlier_flags[col] = z_scores > z_threshold

    if summary:
        summary_df = pd.DataFrame(
            {
                "outlier_count": outlier_flags.sum(),
                "percent_outliers": 100 * outlier_flags.sum() / len(df),
            }
        ).sort_values("percent_outliers", ascending=False)
        print("📊 Outlier Detection Summary:")
        print(summary_df.round(2))
        return outlier_flags, summary_df
    return outlier_flags


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def timer(func):
    """Decorator to time a function and print elapsed time."""

    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"[{now_str()}] START {func.__name__}")
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[{now_str()}] DONE  {func.__name__} (elapsed {elapsed:.1f}s)")
        return result

    return wrapper


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman correlation with NaN guard."""
    corr, _ = spearmanr(y_true, y_pred)
    if np.isnan(corr):
        return -1.0
    return float(corr)


def blend_correlation_sharpe(spearman_score: float, sharpe_score: float) -> float:
    """Dynamically weight Sharpe vs Spearman using correlation strength."""
    spearman_norm = (spearman_score + 1.0) / 2.0  # -> [0, 1]
    sharpe_weight = 0.2 + 0.6 * spearman_norm  # emphasize Sharpe when correlation is strong
    return (1.0 - sharpe_weight) * spearman_score + sharpe_weight * sharpe_score


def time_series_cv_splits(X: pd.DataFrame, n_splits: int = 4):
    """Yield train/val indices for a TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return tscv.split(X)


class ModelTuner:
    """Optuna tuner for LightGBM and CatBoost using Spearman-based CV."""

    def __init__(self, seed: int = 42, n_splits: int = 4):
        self.seed = seed
        self.n_splits = n_splits

    @timer
    def tune_lightgbm(
        self, X: pd.DataFrame, y: pd.Series, n_trials: int = 30, n_jobs: int = 1
    ) -> optuna.study.Study:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            params = {
                "objective": "regression",
                "metric": "rmse",
                "n_estimators": trial.suggest_int("n_estimators", 400, 2600),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "num_leaves": trial.suggest_int("num_leaves", 16, 1000),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "subsample_freq": 1,
                "random_state": self.seed,
                "verbosity": -1,
            }
            fold_scores = []
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            for train_idx, val_idx in tscv.split(X):
                model = LGBMRegressor(**params)
                model.fit(
                    X.iloc[train_idx],
                    y.iloc[train_idx],
                    eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                    eval_metric="rmse",
                    callbacks=[
                        early_stopping(stopping_rounds=100, verbose=False),
                        log_evaluation(period=0),
                    ],
                )
                preds = model.predict(X.iloc[val_idx])
                y_val = y.iloc[val_idx].values
                spearman = safe_spearman(y_val, preds)
                sharpe = hull_sharpe_ratio(y_val, preds)
                fold_scores.append(blend_correlation_sharpe(spearman, sharpe))
            return float(np.mean(fold_scores))

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
        return study

    @timer
    def tune_catboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 30,
        use_pool: bool = False,
        task_type: str = "GPU",
    ) -> optuna.study.Study:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            params = {
                "loss_function": "RMSE",
                "iterations": trial.suggest_int("iterations", 400, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
                "random_seed": self.seed,
                "verbose": False,
            }
            if task_type.upper() == "GPU":
                params["task_type"] = "GPU"
                params["devices"] = "0"

            fold_scores = []
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                if use_pool:
                    pool_train = Pool(X_train, label=y_train)
                    pool_val = Pool(X_val, label=y_val)
                    model = CatBoostRegressor(**params)
                    model.fit(
                        pool_train,
                        eval_set=pool_val,
                        use_best_model=True,
                        early_stopping_rounds=100,
                        verbose=False,
                    )
                else:
                    model = CatBoostRegressor(**params)
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=(X_val, y_val),
                        use_best_model=True,
                        early_stopping_rounds=100,
                        verbose=False,
                    )
                preds = model.predict(X_val)
                y_val_arr = y_val.values
                spearman = safe_spearman(y_val_arr, preds)
                sharpe = hull_sharpe_ratio(y_val_arr, preds)
                fold_scores.append(blend_correlation_sharpe(spearman, sharpe))
            return float(np.mean(fold_scores))

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study

    @timer
    def train_final_lgbm(
        self, X: pd.DataFrame, y: pd.Series, best_params: Dict[str, Any], model_path: str
    ):
        params = dict(best_params)
        params.setdefault("random_state", self.seed)
        params.setdefault("verbosity", -1)
        model = LGBMRegressor(**params)
        model.fit(X, y)
        joblib.dump(model, model_path)
        return model

    @timer
    def train_final_catboost(
        self, X: pd.DataFrame, y: pd.Series, best_params: Dict[str, Any], model_path: str
    ):
        params = dict(best_params)
        params.setdefault("random_seed", self.seed)
        params.setdefault("verbose", False)
        model = CatBoostRegressor(**params)
        model.fit(X, y, use_best_model=False, verbose=False)
        model.save_model(model_path)
        return model

    @timer
    def tune_xgboost(
        self, X: pd.DataFrame, y: pd.Series, n_trials: int = 30
    ) -> optuna.study.Study:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            params = {
                "objective": "reg:squarederror",
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
                "n_estimators": trial.suggest_int("n_estimators", 400, 2000),
                "early_stopping_rounds": 100,
                "tree_method": "hist",
                "random_state": self.seed,
            }
            fold_scores: List[float] = []
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            for train_idx, val_idx in tscv.split(X):
                model = XGBRegressor(**params)
                model.fit(
                    X.iloc[train_idx],
                    y.iloc[train_idx],
                    eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                    verbose=False,
                )
                preds = model.predict(X.iloc[val_idx])
                y_val = y.iloc[val_idx].values
                spearman = safe_spearman(y_val, preds)
                sharpe = hull_sharpe_ratio(y_val, preds)
                fold_scores.append(blend_correlation_sharpe(spearman, sharpe))
            return float(np.mean(fold_scores))

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study
