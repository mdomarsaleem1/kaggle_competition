"""
Ensemble predictions from multiple models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from typing import Dict, List, Optional

from models import (
    XGBoostTimeSeriesModel,
    LightGBMTimeSeriesModel,
    CatBoostTimeSeriesModel,
    ProphetTimeSeriesModel,
    ChronosTimeSeriesModel
)


class ModelEnsemble:
    """Ensemble of trained time series models"""

    def __init__(self, model_dir: str = 'trained_models'):
        """
        Initialize ensemble

        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.preprocessor = None
        self.weights = {}

    def load_models(self, models_to_load: Optional[List[str]] = None):
        """
        Load trained models

        Args:
            models_to_load: List of model names to load (default: all)
        """
        print("Loading models...")

        # Load preprocessor
        preprocessor_path = self.model_dir / 'preprocessor.pkl'
        if preprocessor_path.exists():
            self.preprocessor = joblib.load(preprocessor_path)
            print("Preprocessor loaded")

        if models_to_load is None:
            models_to_load = ['xgboost', 'lightgbm', 'catboost', 'prophet']

        # Load XGBoost
        if 'xgboost' in models_to_load:
            try:
                model = XGBoostTimeSeriesModel()
                model.load_model(str(self.model_dir / 'xgboost_model.json'))
                self.models['xgboost'] = model
                print("XGBoost loaded")
            except Exception as e:
                print(f"Error loading XGBoost: {e}")

        # Load LightGBM
        if 'lightgbm' in models_to_load:
            try:
                model = LightGBMTimeSeriesModel()
                model.load_model(str(self.model_dir / 'lightgbm_model.txt'))
                self.models['lightgbm'] = model
                print("LightGBM loaded")
            except Exception as e:
                print(f"Error loading LightGBM: {e}")

        # Load CatBoost
        if 'catboost' in models_to_load:
            try:
                model = CatBoostTimeSeriesModel()
                model.load_model(str(self.model_dir / 'catboost_model.cbm'))
                self.models['catboost'] = model
                print("CatBoost loaded")
            except Exception as e:
                print(f"Error loading CatBoost: {e}")

        # Load Prophet
        if 'prophet' in models_to_load:
            try:
                model = ProphetTimeSeriesModel()
                model.load_model(str(self.model_dir / 'prophet_model.pkl'))
                self.models['prophet'] = model
                print("Prophet loaded")
            except Exception as e:
                print(f"Error loading Prophet: {e}")

        # Load Chronos (optional)
        if 'chronos' in models_to_load:
            try:
                model = ChronosTimeSeriesModel(model_size='small')
                model.load_model()
                self.models['chronos'] = model
                print("Chronos loaded")
            except Exception as e:
                print(f"Error loading Chronos: {e}")

        print(f"\nLoaded {len(self.models)} models: {list(self.models.keys())}")

    def set_weights(self, weights: Dict[str, float]):
        """
        Set ensemble weights

        Args:
            weights: Dictionary mapping model names to weights
        """
        # Normalize weights
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
        print(f"Ensemble weights: {self.weights}")

    def predict(self, X: np.ndarray, method: str = 'weighted_average') -> np.ndarray:
        """
        Make ensemble predictions

        Args:
            X: Input features
            method: Ensemble method ('weighted_average', 'median', 'mean')

        Returns:
            Ensemble predictions
        """
        predictions = {}

        # Get predictions from tree-based models
        for name in ['xgboost', 'lightgbm', 'catboost']:
            if name in self.models:
                predictions[name] = self.models[name].predict(X)

        if not predictions:
            raise ValueError("No models loaded for prediction")

        # Stack predictions
        pred_array = np.array([predictions[name] for name in predictions.keys()])

        if method == 'mean':
            return np.mean(pred_array, axis=0)
        elif method == 'median':
            return np.median(pred_array, axis=0)
        elif method == 'weighted_average':
            if not self.weights:
                # Use equal weights if not set
                weights = np.ones(len(predictions)) / len(predictions)
            else:
                weights = np.array([self.weights.get(name, 1.0) for name in predictions.keys()])
                weights = weights / weights.sum()

            return np.average(pred_array, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

    def predict_with_prophet(self, future_df: pd.DataFrame) -> np.ndarray:
        """
        Get predictions from Prophet model

        Args:
            future_df: Future dataframe with 'ds' column

        Returns:
            Prophet predictions
        """
        if 'prophet' not in self.models:
            raise ValueError("Prophet model not loaded")

        forecast = self.models['prophet'].predict(future_df)
        return forecast['yhat'].values

    def predict_with_chronos(self, context: np.ndarray, prediction_length: int) -> np.ndarray:
        """
        Get predictions from Chronos model

        Args:
            context: Historical context
            prediction_length: Number of steps to forecast

        Returns:
            Chronos predictions (median)
        """
        if 'chronos' not in self.models:
            raise ValueError("Chronos model not loaded")

        forecasts = self.models['chronos'].predict(
            context=context,
            prediction_length=prediction_length,
            num_samples=20
        )
        return np.median(forecasts, axis=0)

    def create_submission(self, test_df: pd.DataFrame,
                         feature_cols: List[str],
                         submission_file: str = 'submission.csv',
                         id_col: str = 'id',
                         method: str = 'weighted_average') -> pd.DataFrame:
        """
        Create submission file

        Args:
            test_df: Test dataframe
            feature_cols: List of feature column names
            submission_file: Output submission filename
            id_col: ID column name
            method: Ensemble method

        Returns:
            Submission dataframe
        """
        print(f"Creating submission using {method} ensemble...")

        # Prepare features
        X_test = test_df[feature_cols].values

        # Scale features if preprocessor is available
        if self.preprocessor:
            X_test = self.preprocessor.transform(X_test)

        # Make predictions
        predictions = self.predict(X_test, method=method)

        # Create submission dataframe
        submission = pd.DataFrame({
            id_col: test_df[id_col],
            'prediction': predictions
        })

        # Save submission
        submission.to_csv(submission_file, index=False)
        print(f"Submission saved to {submission_file}")

        return submission


def optimize_ensemble_weights(model_dir: str,
                              val_features: np.ndarray,
                              val_targets: np.ndarray) -> Dict[str, float]:
    """
    Optimize ensemble weights using validation data

    Args:
        model_dir: Directory containing trained models
        val_features: Validation features
        val_targets: Validation targets

    Returns:
        Optimized weights dictionary
    """
    from sklearn.metrics import mean_squared_error
    from scipy.optimize import minimize

    ensemble = ModelEnsemble(model_dir)
    ensemble.load_models(['xgboost', 'lightgbm', 'catboost'])

    # Get predictions from each model
    predictions = {}
    for name, model in ensemble.models.items():
        predictions[name] = model.predict(val_features)

    model_names = list(predictions.keys())
    pred_matrix = np.column_stack([predictions[name] for name in model_names])

    def objective(weights):
        """Objective function to minimize"""
        weights = weights / weights.sum()  # Normalize
        ensemble_pred = pred_matrix @ weights
        return mean_squared_error(val_targets, ensemble_pred)

    # Initial weights (equal)
    initial_weights = np.ones(len(model_names)) / len(model_names)

    # Constraints: weights sum to 1 and are non-negative
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    bounds = [(0, 1) for _ in range(len(model_names))]

    # Optimize
    result = minimize(objective, initial_weights,
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)

    optimal_weights = dict(zip(model_names, result.x))

    print("\nOptimal Ensemble Weights:")
    for name, weight in optimal_weights.items():
        print(f"  {name}: {weight:.4f}")

    print(f"\nOptimized RMSE: {np.sqrt(result.fun):.6f}")

    return optimal_weights


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Ensemble predictions')
    parser.add_argument('--model-dir', type=str, default='trained_models',
                       help='Directory containing trained models')
    parser.add_argument('--test-file', type=str, default='data/test.csv',
                       help='Test data file')
    parser.add_argument('--submission-file', type=str, default='submission.csv',
                       help='Output submission file')
    parser.add_argument('--method', type=str, default='weighted_average',
                       choices=['mean', 'median', 'weighted_average'],
                       help='Ensemble method')
    parser.add_argument('--optimize-weights', action='store_true',
                       help='Optimize ensemble weights (requires validation data)')

    args = parser.parse_args()

    # Create ensemble
    ensemble = ModelEnsemble(args.model_dir)
    ensemble.load_models()

    # Load test data
    test_df = pd.read_csv(args.test_file)

    # Create submission
    # Note: You'll need to specify the correct feature columns
    # This is a placeholder - adjust based on your data
    feature_cols = [col for col in test_df.columns if col not in ['id', 'date']]

    submission = ensemble.create_submission(
        test_df,
        feature_cols,
        args.submission_file,
        method=args.method
    )

    print("\nSubmission preview:")
    print(submission.head())
