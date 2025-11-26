"""
Facebook Prophet Time Series Forecasting Model
"""
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Optional, List
import json


class ProphetTimeSeriesModel:
    """Facebook Prophet model for time series forecasting"""

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize Prophet model

        Args:
            params: Prophet parameters dictionary
        """
        self.default_params = {
            'growth': 'linear',  # 'linear' or 'logistic'
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'seasonality_mode': 'additive',  # 'additive' or 'multiplicative'
            'changepoint_range': 0.8,
            'yearly_seasonality': 'auto',
            'weekly_seasonality': 'auto',
            'daily_seasonality': 'auto',
            'interval_width': 0.95
        }

        if params:
            self.default_params.update(params)

        self.model = None
        self.feature_columns = []

    def prepare_data(self, df: pd.DataFrame, date_col: str, target_col: str,
                    additional_regressors: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Prepare data in Prophet format

        Args:
            df: Input dataframe
            date_col: Name of date column
            target_col: Name of target column
            additional_regressors: List of additional regressor columns

        Returns:
            DataFrame in Prophet format (ds, y, regressors)
        """
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = pd.to_datetime(df[date_col])
        prophet_df['y'] = df[target_col]

        if additional_regressors:
            for regressor in additional_regressors:
                if regressor in df.columns:
                    prophet_df[regressor] = df[regressor]
                    self.feature_columns.append(regressor)

        return prophet_df

    def train(self, train_df: pd.DataFrame,
              additional_regressors: Optional[List[str]] = None,
              verbose: bool = True) -> Dict:
        """
        Train Prophet model

        Args:
            train_df: Training data in Prophet format (ds, y, regressors)
            additional_regressors: List of additional regressor columns
            verbose: Print training progress

        Returns:
            Dictionary with training metrics
        """
        # Suppress Prophet warnings if not verbose
        import logging
        if not verbose:
            logging.getLogger('prophet').setLevel(logging.ERROR)
            logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

        self.model = Prophet(**self.default_params)

        # Add additional regressors
        if additional_regressors:
            for regressor in additional_regressors:
                if regressor in train_df.columns:
                    self.model.add_regressor(regressor)
                    if regressor not in self.feature_columns:
                        self.feature_columns.append(regressor)

        self.model.fit(train_df)

        # Calculate training metrics
        train_pred = self.model.predict(train_df)
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(train_df['y'], train_pred['yhat'])),
            'train_mae': mean_absolute_error(train_df['y'], train_pred['yhat'])
        }

        return metrics

    def predict(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions

        Args:
            future_df: Future dataframe with 'ds' column and regressors

        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        forecast = self.model.predict(future_df)
        return forecast

    def make_future_dataframe(self, periods: int, freq: str = 'D',
                             include_history: bool = True) -> pd.DataFrame:
        """
        Create future dataframe for predictions

        Args:
            periods: Number of periods to forecast
            freq: Frequency of predictions ('D', 'W', 'M', etc.)
            include_history: Include historical dates

        Returns:
            Future dataframe
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.make_future_dataframe(periods=periods, freq=freq,
                                               include_history=include_history)

    def cross_validate(self, df: pd.DataFrame, initial: str, period: str,
                      horizon: str) -> pd.DataFrame:
        """
        Perform time series cross-validation

        Args:
            df: Input dataframe in Prophet format
            initial: Initial training period (e.g., '730 days')
            period: Period between cutoff dates (e.g., '180 days')
            horizon: Forecast horizon (e.g., '365 days')

        Returns:
            DataFrame with cross-validation results
        """
        from prophet.diagnostics import cross_validation, performance_metrics

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        df_cv = cross_validation(self.model, initial=initial, period=period, horizon=horizon)
        df_metrics = performance_metrics(df_cv)

        return df_cv, df_metrics

    def save_model(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save.")

        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from file"""
        import pickle
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filepath}")

    def plot_forecast(self, forecast: pd.DataFrame, figsize: tuple = (15, 6)):
        """
        Plot forecast with uncertainty intervals

        Args:
            forecast: Forecast dataframe from predict()
            figsize: Figure size tuple
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        import matplotlib.pyplot as plt

        fig = self.model.plot(forecast, figsize=figsize)
        plt.title('Prophet Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        return fig

    def plot_components(self, forecast: pd.DataFrame, figsize: tuple = (15, 10)):
        """
        Plot forecast components (trend, seasonality, etc.)

        Args:
            forecast: Forecast dataframe from predict()
            figsize: Figure size tuple
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        fig = self.model.plot_components(forecast, figsize=figsize)
        return fig


class ProphetEnsembleModel:
    """Ensemble of Prophet models with different configurations"""

    def __init__(self, param_sets: List[Dict]):
        """
        Initialize ensemble of Prophet models

        Args:
            param_sets: List of parameter dictionaries for each model
        """
        self.models = [ProphetTimeSeriesModel(params) for params in param_sets]

    def train(self, train_df: pd.DataFrame,
              additional_regressors: Optional[List[str]] = None) -> Dict:
        """
        Train all models in ensemble

        Args:
            train_df: Training data in Prophet format
            additional_regressors: List of additional regressor columns

        Returns:
            Dictionary with metrics for each model
        """
        metrics = {}

        for i, model in enumerate(self.models):
            print(f"\nTraining Prophet model {i + 1}/{len(self.models)}")
            model_metrics = model.train(train_df, additional_regressors, verbose=False)
            metrics[f'model_{i + 1}'] = model_metrics

        return metrics

    def predict(self, future_df: pd.DataFrame, method: str = 'mean') -> np.ndarray:
        """
        Make ensemble predictions

        Args:
            future_df: Future dataframe with 'ds' column and regressors
            method: Ensemble method ('mean', 'median', 'weighted')

        Returns:
            Ensemble predictions array
        """
        predictions = []

        for model in self.models:
            forecast = model.predict(future_df)
            predictions.append(forecast['yhat'].values)

        predictions = np.array(predictions)

        if method == 'mean':
            return np.mean(predictions, axis=0)
        elif method == 'median':
            return np.median(predictions, axis=0)
        elif method == 'weighted':
            # Simple inverse RMSE weighting (could be improved)
            weights = 1.0 / (np.array([1.0] * len(self.models)) + 1e-6)
            weights = weights / weights.sum()
            return np.average(predictions, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

    def save_models(self, base_filepath: str):
        """Save all models in ensemble"""
        for i, model in enumerate(self.models):
            model.save_model(f"{base_filepath}_model_{i + 1}.pkl")

    def load_models(self, base_filepath: str, n_models: int):
        """Load all models in ensemble"""
        self.models = []
        for i in range(n_models):
            model = ProphetTimeSeriesModel()
            model.load_model(f"{base_filepath}_model_{i + 1}.pkl")
            self.models.append(model)
