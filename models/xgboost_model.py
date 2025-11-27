"""
XGBoost Time Series Forecasting Model
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Optional, Tuple
import joblib
import json

from utils.metrics import hull_sharpe_xgboost


class XGBoostTimeSeriesModel:
    """XGBoost model for time series forecasting"""

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize XGBoost model

        Args:
            params: XGBoost parameters dictionary
        """
        self.default_params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'random_state': 42,
            'tree_method': 'hist',
            'device': 'cpu'  # Use 'cuda' for GPU
        }

        if params:
            self.default_params.update(params)

        self.model = None
        self.feature_importance = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              verbose: bool = True,
              use_sharpe_metric: bool = False) -> Dict:
        """
        Train XGBoost model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            verbose: Print training progress

        Returns:
            Dictionary with training metrics
        """
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model = xgb.XGBRegressor(**self.default_params)

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric=hull_sharpe_xgboost if use_sharpe_metric else 'rmse',
            verbose=verbose
        )

        self.feature_importance = pd.DataFrame({
            'feature': range(X_train.shape[1]),
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred)
        }

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            metrics['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
            metrics['val_mae'] = mean_absolute_error(y_val, val_pred)

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input features

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def save_model(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save_model(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from file"""
        self.model = xgb.XGBRegressor()
        self.model.load_model(filepath)
        print(f"Model loaded from {filepath}")

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet.")
        return self.feature_importance.head(top_n)

    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                n_trials: int = 100) -> Dict:
        """
        Optimize hyperparameters using Optuna

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of optimization trials

        Returns:
            Best parameters dictionary
        """
        try:
            import optuna
        except ImportError:
            print("Optuna not installed. Install it with: pip install optuna")
            return self.default_params

        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'n_estimators': 1000,
                'early_stopping_rounds': 50,
                'random_state': 42,
                'tree_method': 'hist'
            }

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            return rmse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"Best RMSE: {study.best_value}")
        print(f"Best params: {study.best_params}")

        return study.best_params


class XGBoostMultiStepForecaster:
    """XGBoost model for multi-step time series forecasting"""

    def __init__(self, forecast_horizon: int, params: Optional[Dict] = None):
        """
        Initialize multi-step forecaster

        Args:
            forecast_horizon: Number of steps to forecast
            params: XGBoost parameters
        """
        self.forecast_horizon = forecast_horizon
        self.models = [XGBoostTimeSeriesModel(params) for _ in range(forecast_horizon)]

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train separate models for each forecast step

        Args:
            X_train: Training features
            y_train: Training targets (shape: [samples, forecast_horizon])
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Dictionary with training metrics
        """
        metrics = {}

        for step in range(self.forecast_horizon):
            print(f"\nTraining model for step {step + 1}/{self.forecast_horizon}")

            y_train_step = y_train[:, step] if y_train.ndim > 1 else y_train
            y_val_step = y_val[:, step] if (y_val is not None and y_val.ndim > 1) else y_val

            step_metrics = self.models[step].train(
                X_train, y_train_step,
                X_val, y_val_step,
                verbose=False
            )

            metrics[f'step_{step + 1}'] = step_metrics

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make multi-step predictions

        Args:
            X: Input features

        Returns:
            Predictions array (shape: [samples, forecast_horizon])
        """
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))

        return np.column_stack(predictions)

    def save_models(self, base_filepath: str):
        """Save all models"""
        for i, model in enumerate(self.models):
            model.save_model(f"{base_filepath}_step_{i + 1}.json")

    def load_models(self, base_filepath: str):
        """Load all models"""
        for i, model in enumerate(self.models):
            model.load_model(f"{base_filepath}_step_{i + 1}.json")
