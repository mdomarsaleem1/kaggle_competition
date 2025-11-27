# %% [markdown]
# ## Load Libraries

# %%
import numpy as np
import pandas as pd 
import os
from typing import Optional, Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import optuna
import warnings
import polars as pl
import time
import math
import gc
import lightgbm as lgb
from scipy.stats import spearmanr
import joblib
import kaggle_evaluation.default_inference_server
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="lightgbm")
from IPython.display import display, Markdown
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor, Pool
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %%
train_df = pd.read_csv('/kaggle/input/hull-tactical-market-prediction/train.csv')
test_df =pd.read_csv('/kaggle/input/hull-tactical-market-prediction/test.csv')

# %%
train_df.head(5)

# %% [markdown]
# ## Functions

# %%
# ======================================================
# Lag Features
# ======================================================
def add_lag_features(df: pd.DataFrame, cols, lags):
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


# ======================================================
# Rolling Window Features
# ======================================================
def add_rolling_features(df: pd.DataFrame, cols, windows):
    for col in cols:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_roll_std_{w}"] = df[col].rolling(w, min_periods=1).std()
            df[f"{col}_roll_min_{w}"] = df[col].rolling(w, min_periods=1).min()
            df[f"{col}_roll_max_{w}"] = df[col].rolling(w, min_periods=1).max()
            df[f"{col}_roll_median_{w}"] = df[col].rolling(w, min_periods=1).median()
    return df


# ======================================================
# Percentage Change Features (Volatility / Momentum)
# ======================================================
def add_pct_change_features(df: pd.DataFrame, cols, periods=[1, 3, 7]):
    for col in cols:
        if col not in df.columns:
            continue
        for p in periods:
            df[f"{col}_pct_change_{p}"] = df[col].pct_change(periods=p)
    return df


# ======================================================
# First Difference
# ======================================================
def add_diff_features(df: pd.DataFrame, cols):
    for col in cols:
        if col in df.columns:
            df[f"{col}_diff_1"] = df[col].diff(1)
            df[f"{col}_diff_7"] = df[col].diff(7)
    return df


# ======================================================
# Exponential Weighted Moving Features
# ======================================================
def add_ewm_features(df: pd.DataFrame, cols, spans=[7, 14, 30]):
    for col in cols:
        if col not in df.columns:
            continue
        for s in spans:
            df[f"{col}_ewm_mean_{s}"] = df[col].ewm(span=s, adjust=False).mean()
            df[f"{col}_ewm_std_{s}"] = df[col].ewm(span=s, adjust=False).std()
    return df


# ======================================================
# Feature Interactions (non-linear relationships)
# ======================================================
def add_interaction_features(df: pd.DataFrame, cols):
    numeric_cols = [c for c in cols if df[c].dtype != "object"]
    
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i+1:]:
            df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
            df[f"{c1}_div_{c2}"] = df[c1] / (df[c2] + 1e-7)
    return df


# ======================================================
# Rolling Normalization (z-score inside windows)
# ======================================================
def add_rolling_normalized_features(df: pd.DataFrame, cols, windows=[14, 30]):
    for col in cols:
        if col not in df.columns:
            continue
        
        for w in windows:
            m = df[col].rolling(w, min_periods=1).mean()
            s = df[col].rolling(w, min_periods=1).std()
            df[f"{col}_roll_zscore_{w}"] = (df[col] - m) / (s + 1e-7)
    return df


# ======================================================
# FEATURE PIPELINE
# ======================================================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # Define primary columns for FE
    cols = TOP_FEATURES_FOR_FE
    
    # --- Base FE ---
    df = add_lag_features(df, cols, LAG_PERIODS)
    df = add_rolling_features(df, cols, ROLLING_WINDOWS)

    # --- Extra FE modules ---
    df = add_pct_change_features(df, cols)
    df = add_diff_features(df, cols)
    df = add_ewm_features(df, cols)
    df = add_interaction_features(df, cols)
    df = add_rolling_normalized_features(df, cols)

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    for c in df.columns:
        if df[c].isnull().any():
            df[c].fillna(df[c].median(), inplace=True)

    return df





# %%
def describe_dataset(df: pd.DataFrame):
    """
    Overview of a DataFrame, including
    dimensions, memory usage, data types, and detailed descriptive statistics.

    Args:
        df (pd.DataFrame): The input DataFrame to describe.
    """
    
    display(Markdown("## 📋 Dataset Overview"))
    print("--- Basic Dimensions & Memory ---")
    
    # shape
    num_rows, num_cols = df.shape
    print(f"**Shape (Rows, Columns):** ({num_rows:,}, {num_cols:,})")
    
    # memory
    mem_usage = df.memory_usage(deep=True).sum()
    mem_gbs = mem_usage / (1024**2)
    print(f"**Total Memory Usage:** {mem_gbs:.2f} MB")
    
    print("\n--- Feature Data Types and Counts ---")
    
    # data types
    dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
    dtype_counts.columns = ['Data_Type', 'Count']
    print(dtype_counts.to_markdown(index=False))

    
    # stats
    display(Markdown("\n## 📊 Descriptive Statistics"))
    
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    # numerical
    if numerical_cols:
        display(Markdown("### Numerical Features"))
        # Use transpose for better readability when many features exist
        num_desc = df[numerical_cols].describe().T
        # Add IQR for a more detailed statistical view
        num_desc['IQR'] = num_desc['75%'] - num_desc['25%']
        display(num_desc.style.format("{:,.2f}"))
        print(f"Found {len(numerical_cols)} numerical features.")

    # categorical
    if categorical_cols:
        display(Markdown("### Categorical / Object Features"))
        # Include top, frequency, and unique count
        cat_desc = df[categorical_cols].describe().T
        display(cat_desc.style.format({"unique": "{:,}", "freq": "{:,}"}))
        print(f"Found {len(categorical_cols)} categorical/object features.")

    # datetime
    if datetime_cols:
        display(Markdown("### Datetime Features"))
        dt_desc = df[datetime_cols].describe().T
        display(dt_desc)
        print(f"Found {len(datetime_cols)} datetime features.")


def missing_duplicates_analysis(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    print("--- 📊 Missing Data and Duplicates Analysis ---")
    # missing summary
    missing_counts = df.isnull().sum()
    missing_summary = pd.DataFrame({
        'Missing_Count': missing_counts,
        'Missing_Percent': 100 * missing_counts / len(df)
    })
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
    missing_summary = missing_summary.sort_values(by='Missing_Count', ascending=False)
    
    
    # check for duplicates
    num_duplicates = df.duplicated().sum()
    print(f"**Duplicate rows found:** {df.duplicated().sum()}")
    
    # visualisation
    if missing_summary.empty:
        print("✅ **No missing values found** in the dataset.")
        return pd.DataFrame()
    
    print(f"\n**Total features with missing values:** {len(missing_summary)}")
    
    # Select the top n features
    plot_data = missing_summary.head(top_n)
    
    plt.style.use('ggplot')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=plot_data.index,
        y='Missing_Count',
        data=plot_data,
        palette='viridis'
    )
    
    # Add percentage labels above the bars
    for i, count in enumerate(plot_data['Missing_Count']):
        percent = plot_data['Missing_Percent'].iloc[i]
        plt.text(
            x=i, 
            y=count + (df.shape[0] * 0.005),
            s=f'{percent:.1f}%',
            ha='center',
            fontsize=9
        )
    
    plt.title(f"Top {min(top_n, len(missing_summary))} Features by Missing Values Count (Total Rows: {len(df)})", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Feature Name", fontsize=12)
    plt.ylabel("Missing Count", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    print("\n**Missing Data Summary Table (Top 10):**")
    print(missing_summary.head(10).to_markdown(floatfmt=".2f"))
    return missing_summary


def detect_outliers(df, method='iqr', threshold=2.5, z_threshold=3.0, cols=None, summary=True):

    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if method not in ['iqr', 'zscore']:
        raise ValueError("method must be 'iqr' or 'zscore'")

    outlier_flags = pd.DataFrame(False, index=df.index, columns=cols)

    for col in cols:
        series = df[col].dropna()

        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_flags[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            outlier_flags[col] = z_scores > z_threshold

    if summary:
        summary_df = pd.DataFrame({
            'outlier_count': outlier_flags.sum(),
            'percent_outliers': 100 * outlier_flags.sum() / len(df)
        }).sort_values('percent_outliers', ascending=False)

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
    """Compute Spearman correlation"""
    corr, _ = spearmanr(y_true, y_pred)
    if np.isnan(corr):
        return -1.0
    return float(corr)

def time_series_cv_splits(X: pd.DataFrame, n_splits: int = 4):
    """Yield train/val indices for a TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return tscv.split(X)

# Plots

def plot_feature_importance_lgbm(model: LGBMRegressor, feature_names: List[str], top_n: int = 20, figsize=(10,8), save_path: Optional[str]=None):
    """Plot top_n LGBM feature importances (gain)."""
    # try booster_.feature_importance to get 'gain' if available
    try:
        imp = model.booster_.feature_importance(importance_type="gain")
    except Exception:
        imp = model.feature_importances_
    imp = np.array(imp)
    indices = np.argsort(imp)[::-1][:top_n]
    names = np.array(feature_names)[indices]
    values = imp[indices]

    plt.figure(figsize=figsize)
    plt.barh(names, values)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance (gain)")
    plt.title(f"LightGBM Top-{top_n} Feature Importances")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_feature_importance_catboost(model: CatBoostRegressor, feature_names: List[str], top_n: int = 20, figsize=(10,8), save_path: Optional[str]=None):
    """Plot top_n CatBoost feature importances (PredictionValuesChange or LossFunctionChange)."""
    try:
        # prediction values change is usually informative
        imp = np.array(model.get_feature_importance(type="PredictionValuesChange"))
    except Exception:
        imp = np.array(model.get_feature_importance())
    indices = np.argsort(imp)[::-1][:top_n]
    names = np.array(feature_names)[indices]
    values = imp[indices]

    plt.figure(figsize=figsize)
    plt.barh(names, values)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"CatBoost Top-{top_n} Feature Importances")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()



# ModelTuner
class ModelTuner:
    def __init__(self, seed: int = 42, n_splits: int = 4):
        self.seed = seed
        self.n_splits = n_splits

    @timer
    def tune_lightgbm(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 30, n_jobs: int = 1) -> optuna.study.Study:
        """Tune LGBM with Optuna optimizing mean Spearman across time-series folds."""
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
                "verbosity": -1
            }

            fold_scores = []
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            for train_idx, val_idx in tscv.split(X):
                model = LGBMRegressor(**params)
                model.fit(
                    X.iloc[train_idx], y.iloc[train_idx],
                    eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                    eval_metric="rmse",
                    callbacks=[early_stopping(stopping_rounds=100, verbose=False), log_evaluation(period=0)]
                )
                preds = model.predict(X.iloc[val_idx])
                fold_scores.append(safe_spearman(y.iloc[val_idx].values, preds))

            return float(np.mean(fold_scores))

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
        return study

    @timer
    def tune_catboost(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 30, use_pool: bool = False, task_type: str = "GPU") -> optuna.study.Study:
        """Tune CatBoost with Optuna optimizing mean Spearman across time-series folds.
           use_pool: if True, wraps data in catboost.Pool.
           task_type: 'CPU' or 'GPU'.
        """
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
                # "rsm": trial.suggest_float("rsm", 0.6, 1.0),
                "random_seed": self.seed,
                "verbose": False,
                # "bootstrap_type": "MVS",
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
                    model.fit(pool_train, eval_set=pool_val, use_best_model=True,
                              early_stopping_rounds=100, verbose=False)
                else:
                    model = CatBoostRegressor(**params)
                    model.fit(X_train, y_train, eval_set=(X_val, y_val),
                              use_best_model=True, early_stopping_rounds=100, verbose=False)

                preds = model.predict(X_val)
                fold_scores.append(safe_spearman(y_val.values, preds))

            return float(np.mean(fold_scores))

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study

    # Final training utilities
    @timer
    def train_final_lgbm(self, X: pd.DataFrame, y: pd.Series, best_params: Dict[str,Any], model_path: str):
        p = dict(best_params)
        p.setdefault("random_state", self.seed)
        p.setdefault("verbosity", -1)
        model = LGBMRegressor(**p)
        model.fit(X, y)  # we rely on best_params having sensible n_estimators
        joblib.dump(model, model_path)
        return model

    @timer
    def train_final_catboost(self, X: pd.DataFrame, y: pd.Series, best_params: Dict[str,Any], model_path: str):
        p = dict(best_params)
        p.setdefault("random_seed", self.seed)
        p.setdefault("verbose", False)
        model = CatBoostRegressor(**p)
        model.fit(X, y, use_best_model=False, verbose=False)
        model.save_model(model_path)
        return model


# %% [markdown]
# ## EDA

# %%
describe_dataset(train_df)

# %%
missing_df_summary = missing_duplicates_analysis(train_df)

# %%
outliers, summary = detect_outliers(train_df, method='iqr')

# %% [markdown]
# ### CONFIG

# %%
TRAIN_PATH = '/kaggle/input/hull-tactical-market-prediction/train.csv'
LOCAL_GATEWAY_PATH = '/kaggle/input/hull-tactical-market-prediction/'

TOP_FEATURES_FOR_FE = ['E1','E10', 'E11', 'E12', 'E13', 'E14', 
                       'E15', 'E16', 'E17', 'E18', 'E19',
                       'E2', 'E20', 'E3', 'E4', 'E5', 'E6', 'E8', 'E9',
                       'S2', 'P9',  'S1', 'S5', 'I2', 'P8',
                       'P10', 'P12', 'P13',
                      'M4', 'M2', 'V5']

LAG_PERIODS = [1, 3, 5, 7, 14, 20]
ROLLING_WINDOWS = [3, 6, 10, 20, 60]

TARGET = 'market_forward_excess_returns'
COLS_TO_DROP = ['forward_returns', 'risk_free_rate', 'excess_return', 'E7', 'V10', 'S3', 'M1', 'M14']
TUNER_SEED = 2
N_TRIALS_LIGHTGBM = 10 
N_TRIALS_CATBOOST = 10
N_SPLITS = 4
USE_CATPOOL = False 
CAT_TASK = "GPU"  

# %% [markdown]
# - Using date_id and TimeSeriesSplit ensures that training and validation sets respect temporal order, preventing data leakage.
# 
# - Optuna allows efficient search of the hyperparameter space using Bayesian optimization. Using Spearman correlation as the optimization metric aligns model objectives with rank-based trading signals.
# 

# %% [markdown]
# ## Training Pipeline

# %%
# Load + preprocess
print("\n--- Loading data ---")
df = pd.read_csv(TRAIN_PATH)
if 'date_id' not in df.columns:
    df['date_id'] = df.index
df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], inplace=True)

print("\n--- Creating features ---")
df_feat = create_features(df)
df_feat.dropna(subset=[TARGET], inplace=True)

FEATURES = [c for c in df_feat.columns if c not in [TARGET, 'date_id']]
X = df_feat[FEATURES]
y = df_feat[TARGET]

print(f"X shape: {X.shape}  y shape: {y.shape}")

tuner = ModelTuner(seed=TUNER_SEED, n_splits=N_SPLITS)

# Tune LightGBM
print("\n--- Tuning LightGBM ---")
study_lgb = tuner.tune_lightgbm(X, y, n_trials=N_TRIALS_LIGHTGBM, n_jobs=1)
print("LGB study best value (mean Spearman):", study_lgb.best_value)
print("LGB best params (trimmed):", {k: study_lgb.best_params[k] for k in study_lgb.best_params})

# Train final LightGBM
print("\n--- Training final LightGBM ---")
final_lgb = tuner.train_final_lgbm(X, y, study_lgb.best_params, "final_lgbm.joblib")
print("Saved final_lgbm.joblib")

# Plot top features for LGBM
print("\n--- LGBM Feature Importance (top 20) ---")
plot_feature_importance_lgbm(final_lgb, FEATURES, top_n=20)

# Tune CatBoost
print("\n--- Tuning CatBoost ---")
study_cb = tuner.tune_catboost(X, y, n_trials=N_TRIALS_CATBOOST, use_pool=USE_CATPOOL, task_type=CAT_TASK)
print("CatBoost study best value (mean Spearman):", study_cb.best_value)
print("CatBoost best params (trimmed):", {k: study_cb.best_params[k] for k in study_cb.best_params})

# Train final CatBoost
print("\n--- Training final CatBoost ---")
final_cb = tuner.train_final_catboost(X, y, study_cb.best_params, "final_catboost.cbm")
print("Saved final_catboost.cbm")

# Plot top features for CatBoost
print("\n--- CatBoost Feature Importance (top 20) ---")
plot_feature_importance_catboost(final_cb, FEATURES, top_n=20)

# Save the features used for inference
joblib.dump(FEATURES, "features_lgbm.joblib")
joblib.dump(FEATURES, "features_cat.joblib")

# Clean up
gc.collect()

# %% [markdown]
# ## Inference

# %%
print("Loading artifacts for inference...")

try:
    # Load LightGBM artifacts
    lgb_model = joblib.load("final_lgbm.joblib")
    lgb_features = joblib.load("features_lgbm.joblib")

    # Load CatBoost artifacts
    cat_model = CatBoostRegressor()
    cat_model.load_model("final_catboost.cbm")
    cat_features = joblib.load("features_cat.joblib")

except Exception as e:
    raise RuntimeError(
        f"Could not load model/features. Ensure training was successful. Error: {e}"
    )

print("Initializing prediction history...")
history_df = pd.read_csv(TRAIN_PATH)

# Clean history
cols_to_drop_hist = [
    col for col in COLS_TO_DROP 
    if col in history_df.columns and col != TARGET
]
history_df.drop(columns=cols_to_drop_hist, inplace=True)

if 'date_id' not in history_df.columns:
    history_df['date_id'] = history_df.index

print("Setup complete. Ready for prediction.")

# -----------------------------
# Helper: Convert return to signal
# -----------------------------
def convert_ret_to_signal(ret: float) -> int:
    """
    Convert predicted return into discrete signal (0,1,2) 
    using a dynamic thresholding based on training distribution.
    """
    arr = np.array([ret], dtype=float)

    q75 = max(0, np.quantile(arr, 0.75))
    if ret <= 0:
        return 0
    elif ret <= q75:
        return 1
    else:
        return 2


# -----------------------------
# Main prediction function
# -----------------------------
def predict(test_df_pl: pl.DataFrame) -> float:
    global history_df

    # Convert to pandas
    test_df_pd = test_df_pl.to_pandas()

    # Assign next date_id
    if 'date_id' not in test_df_pd.columns:
        last_id = history_df['date_id'].max() if not history_df.empty else -1
        test_df_pd['date_id'] = last_id + 1

    # Update historical data
    history_df = pd.concat([history_df, test_df_pd], ignore_index=True)

    # Prepare rolling window slice
    slice_size = max(ROLLING_WINDOWS) + max(LAG_PERIODS) + 5
    window = history_df.tail(slice_size)

    processed = create_features(window)

    # -----------------------------
    # Prepare features for each model
    # -----------------------------
    x_lgb = processed.tail(1)[lgb_features]
    x_cat = processed.tail(1)[cat_features]

    # -----------------------------
    # Make predictions
    # -----------------------------
    pred_lgb = float(lgb_model.predict(x_lgb)[0])
    pred_cat = float(cat_model.predict(x_cat)[0])

    # -----------------------------
    # Ensemble prediction
    # -----------------------------
    blended_pred = 0.6 * pred_lgb + 0.4 * pred_cat

    # Convert to discrete signal
    signal = convert_ret_to_signal(blended_pred)

    gc.collect()
    return float(signal)


# -----------------------------
# Inference server
# -----------------------------
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    print("Serving predictions for the competition...")
    inference_server.serve()
else:
    print("Running local gateway for testing...")
    inference_server.run_local_gateway((LOCAL_GATEWAY_PATH,))

print("Submission script finished.")


# %%


# %%


# %%



