"""
CatBoost Time Series Forecasting Model
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Optional, Tuple


class CatBoostTimeSeriesModel:
    """CatBoost model for time series forecasting"""

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize CatBoost model

        Args:
            params: CatBoost parameters dictionary
        """
        self.default_params = {
            'loss_function': 'RMSE',
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'iterations': 1000,
            'early_stopping_rounds': 50,
            'random_seed': 42,
            'verbose': False,
            'task_type': 'CPU',  # Use 'GPU' for GPU
            'bootstrap_type': 'Bayesian',
            'bagging_temperature': 1,
            'subsample': 0.8,
            'sampling_frequency': 'PerTree',
            'border_count': 128
        }

        if params:
            self.default_params.update(params)

        self.model = None
        self.feature_importance = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              cat_features: Optional[list] = None,
              verbose: bool = True) -> Dict:
        """
        Train CatBoost model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            cat_features: List of categorical feature indices
            verbose: Print training progress

        Returns:
            Dictionary with training metrics
        """
        train_pool = Pool(X_train, y_train, cat_features=cat_features)

        eval_set = None
        if X_val is not None and y_val is not None:
            val_pool = Pool(X_val, y_val, cat_features=cat_features)
            eval_set = val_pool

        params = self.default_params.copy()
        if verbose:
            params['verbose'] = 100
        else:
            params['verbose'] = False

        self.model = CatBoostRegressor(**params)

        self.model.fit(
            train_pool,
            eval_set=eval_set,
            use_best_model=True if eval_set is not None else False
        )

        self.feature_importance = pd.DataFrame({
            'feature': range(X_train.shape[1]),
            'importance': self.model.get_feature_importance()
        }).sort_values('importance', ascending=False)

        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'best_iteration': self.model.best_iteration_ if hasattr(self.model, 'best_iteration_') else params['iterations']
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
        self.model = CatBoostRegressor()
        self.model.load_model(filepath)
        print(f"Model loaded from {filepath}")

    def get_feature_importance(self, top_n: int = 20, importance_type: str = 'FeatureImportance') -> pd.DataFrame:
        """
        Get top N most important features

        Args:
            top_n: Number of top features to return
            importance_type: Type of importance ('FeatureImportance', 'PredictionValuesChange', etc.)

        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet.")
        return self.feature_importance.head(top_n)

    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                cat_features: Optional[list] = None,
                                n_trials: int = 100) -> Dict:
        """
        Optimize hyperparameters using Optuna

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            cat_features: List of categorical feature indices
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
                'loss_function': 'RMSE',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'iterations': 1000,
                'early_stopping_rounds': 50,
                'random_seed': 42,
                'verbose': False,
                'task_type': 'CPU',
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'border_count': trial.suggest_int('border_count', 32, 255)
            }

            train_pool = Pool(X_train, y_train, cat_features=cat_features)
            val_pool = Pool(X_val, y_val, cat_features=cat_features)

            model = CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)

            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            return rmse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"Best RMSE: {study.best_value}")
        print(f"Best params: {study.best_params}")

        return study.best_params


class CatBoostMultiStepForecaster:
    """CatBoost model for multi-step time series forecasting"""

    def __init__(self, forecast_horizon: int, params: Optional[Dict] = None):
        """
        Initialize multi-step forecaster

        Args:
            forecast_horizon: Number of steps to forecast
            params: CatBoost parameters
        """
        self.forecast_horizon = forecast_horizon
        self.models = [CatBoostTimeSeriesModel(params) for _ in range(forecast_horizon)]

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              cat_features: Optional[list] = None) -> Dict:
        """
        Train separate models for each forecast step

        Args:
            X_train: Training features
            y_train: Training targets (shape: [samples, forecast_horizon])
            X_val: Validation features
            y_val: Validation targets
            cat_features: List of categorical feature indices

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
                cat_features=cat_features,
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
            model.save_model(f"{base_filepath}_step_{i + 1}.cbm")

    def load_models(self, base_filepath: str):
        """Load all models"""
        for i, model in enumerate(self.models):
            model.load_model(f"{base_filepath}_step_{i + 1}.cbm")
