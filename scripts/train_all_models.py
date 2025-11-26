"""
Train all time series forecasting models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from tqdm import tqdm

from utils.data_utils import TimeSeriesPreprocessor, split_time_series
from models import (
    XGBoostTimeSeriesModel,
    LightGBMTimeSeriesModel,
    CatBoostTimeSeriesModel,
    ProphetTimeSeriesModel,
    ChronosTimeSeriesModel
)


class ModelTrainer:
    """Train and evaluate all models"""

    def __init__(self, data_dir: str = 'data', output_dir: str = 'trained_models'):
        """
        Initialize trainer

        Args:
            data_dir: Directory containing data files
            output_dir: Directory to save trained models
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.preprocessor = TimeSeriesPreprocessor(scaler_type='standard')
        self.models = {}
        self.results = {}

    def load_and_prepare_data(self, train_file: str = 'train.csv',
                             target_col: str = 'target',
                             date_col: str = 'date'):
        """
        Load and prepare data for modeling

        Args:
            train_file: Training data filename
            target_col: Target column name
            date_col: Date column name

        Returns:
            Prepared train and validation sets
        """
        print("Loading data...")
        train_df, _ = self.preprocessor.load_data(
            str(self.data_dir / train_file)
        )

        print(f"Data shape: {train_df.shape}")
        print(f"Date range: {train_df[date_col].min()} to {train_df[date_col].max()}")

        # Create features
        print("\nCreating features...")
        df_features = self.preprocessor.create_all_features(train_df, target_col)

        print(f"Features shape: {df_features.shape}")
        print(f"Number of features: {df_features.shape[1]}")

        # Split data
        print("\nSplitting data...")
        train_data, val_data = split_time_series(df_features, test_size=0.2)

        # Prepare feature columns (exclude date and target)
        feature_cols = [col for col in df_features.columns
                       if col not in [date_col, target_col]]

        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values
        X_val = val_data[feature_cols].values
        y_val = val_data[target_col].values

        # Scale features
        X_train = self.preprocessor.fit_transform(X_train)
        X_val = self.preprocessor.transform(X_val)

        print(f"Train set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")

        # Save preprocessor
        joblib.dump(self.preprocessor, self.output_dir / 'preprocessor.pkl')

        return X_train, y_train, X_val, y_val, train_data, val_data

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        print("\n" + "="*50)
        print("Training XGBoost Model")
        print("="*50)

        model = XGBoostTimeSeriesModel()
        metrics = model.train(X_train, y_train, X_val, y_val, verbose=True)

        print(f"\nXGBoost Results:")
        print(f"Train RMSE: {metrics['train_rmse']:.6f}")
        print(f"Train MAE: {metrics['train_mae']:.6f}")
        print(f"Val RMSE: {metrics['val_rmse']:.6f}")
        print(f"Val MAE: {metrics['val_mae']:.6f}")

        model.save_model(str(self.output_dir / 'xgboost_model.json'))
        self.models['xgboost'] = model
        self.results['xgboost'] = metrics

        # Feature importance
        importance = model.get_feature_importance(top_n=10)
        print(f"\nTop 10 Important Features:")
        print(importance)

        return model, metrics

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        print("\n" + "="*50)
        print("Training LightGBM Model")
        print("="*50)

        model = LightGBMTimeSeriesModel()
        metrics = model.train(X_train, y_train, X_val, y_val, verbose=True)

        print(f"\nLightGBM Results:")
        print(f"Train RMSE: {metrics['train_rmse']:.6f}")
        print(f"Train MAE: {metrics['train_mae']:.6f}")
        print(f"Val RMSE: {metrics['val_rmse']:.6f}")
        print(f"Val MAE: {metrics['val_mae']:.6f}")
        print(f"Best Iteration: {metrics['best_iteration']}")

        model.save_model(str(self.output_dir / 'lightgbm_model.txt'))
        self.models['lightgbm'] = model
        self.results['lightgbm'] = metrics

        return model, metrics

    def train_catboost(self, X_train, y_train, X_val, y_val):
        """Train CatBoost model"""
        print("\n" + "="*50)
        print("Training CatBoost Model")
        print("="*50)

        model = CatBoostTimeSeriesModel()
        metrics = model.train(X_train, y_train, X_val, y_val, verbose=True)

        print(f"\nCatBoost Results:")
        print(f"Train RMSE: {metrics['train_rmse']:.6f}")
        print(f"Train MAE: {metrics['train_mae']:.6f}")
        print(f"Val RMSE: {metrics['val_rmse']:.6f}")
        print(f"Val MAE: {metrics['val_mae']:.6f}")

        model.save_model(str(self.output_dir / 'catboost_model.cbm'))
        self.models['catboost'] = model
        self.results['catboost'] = metrics

        return model, metrics

    def train_prophet(self, train_data, val_data, date_col='date', target_col='target'):
        """Train Prophet model"""
        print("\n" + "="*50)
        print("Training Prophet Model")
        print("="*50)

        model = ProphetTimeSeriesModel()

        # Prepare data for Prophet
        prophet_train = model.prepare_data(train_data, date_col, target_col)
        prophet_val = model.prepare_data(val_data, date_col, target_col)

        # Train model
        metrics = model.train(prophet_train, verbose=True)

        # Validate
        val_forecast = model.predict(prophet_val[['ds']])
        val_rmse = np.sqrt(np.mean((val_forecast['yhat'].values - prophet_val['y'].values)**2))
        val_mae = np.mean(np.abs(val_forecast['yhat'].values - prophet_val['y'].values))

        metrics['val_rmse'] = val_rmse
        metrics['val_mae'] = val_mae

        print(f"\nProphet Results:")
        print(f"Train RMSE: {metrics['train_rmse']:.6f}")
        print(f"Train MAE: {metrics['train_mae']:.6f}")
        print(f"Val RMSE: {metrics['val_rmse']:.6f}")
        print(f"Val MAE: {metrics['val_mae']:.6f}")

        model.save_model(str(self.output_dir / 'prophet_model.pkl'))
        self.models['prophet'] = model
        self.results['prophet'] = metrics

        return model, metrics

    def train_chronos(self, train_data, val_data, target_col='target',
                     model_size='small', context_length=128, prediction_length=30):
        """Train (load) Chronos model"""
        print("\n" + "="*50)
        print("Loading Chronos-2 Model")
        print("="*50)

        try:
            model = ChronosTimeSeriesModel(model_size=model_size)
            model.load_model()

            # Evaluate on validation set
            print("\nEvaluating Chronos on validation set...")
            metrics = model.evaluate(
                val_data,
                context_length=context_length,
                prediction_length=prediction_length,
                target_col=target_col
            )

            print(f"\nChronos Results:")
            print(f"Val RMSE: {metrics['rmse']:.6f}")
            print(f"Val MAE: {metrics['mae']:.6f}")
            print(f"Predictions made: {metrics['num_predictions']}")

            self.models['chronos'] = model
            self.results['chronos'] = metrics

            return model, metrics

        except Exception as e:
            print(f"Error loading Chronos: {e}")
            print("Skipping Chronos model...")
            return None, None

    def train_all(self, train_file: str = 'train.csv',
                 target_col: str = 'target',
                 date_col: str = 'date',
                 include_chronos: bool = False):
        """
        Train all models

        Args:
            train_file: Training data filename
            target_col: Target column name
            date_col: Date column name
            include_chronos: Whether to include Chronos (requires GPU and more time)
        """
        # Load and prepare data
        X_train, y_train, X_val, y_val, train_data, val_data = \
            self.load_and_prepare_data(train_file, target_col, date_col)

        # Train tree-based models
        self.train_xgboost(X_train, y_train, X_val, y_val)
        self.train_lightgbm(X_train, y_train, X_val, y_val)
        self.train_catboost(X_train, y_train, X_val, y_val)

        # Train Prophet
        self.train_prophet(train_data, val_data, date_col, target_col)

        # Train Chronos (optional)
        if include_chronos:
            self.train_chronos(train_data, val_data, target_col)

        # Save results
        self.save_results()

        # Print summary
        self.print_summary()

    def save_results(self):
        """Save training results to JSON"""
        results_file = self.output_dir / 'training_results.json'

        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for model_name, metrics in self.results.items():
            serializable_results[model_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
            }

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to {results_file}")

    def print_summary(self):
        """Print summary of all models"""
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)

        # Create comparison dataframe
        comparison = []
        for model_name, metrics in self.results.items():
            comparison.append({
                'Model': model_name.upper(),
                'Val RMSE': metrics.get('val_rmse', metrics.get('rmse', 'N/A')),
                'Val MAE': metrics.get('val_mae', metrics.get('mae', 'N/A'))
            })

        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('Val RMSE')

        print(comparison_df.to_string(index=False))
        print("="*70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train all time series models')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing data files')
    parser.add_argument('--output-dir', type=str, default='trained_models',
                       help='Directory to save trained models')
    parser.add_argument('--train-file', type=str, default='train.csv',
                       help='Training data filename')
    parser.add_argument('--target-col', type=str, default='target',
                       help='Target column name')
    parser.add_argument('--date-col', type=str, default='date',
                       help='Date column name')
    parser.add_argument('--include-chronos', action='store_true',
                       help='Include Chronos model (requires more resources)')

    args = parser.parse_args()

    trainer = ModelTrainer(args.data_dir, args.output_dir)
    trainer.train_all(
        train_file=args.train_file,
        target_col=args.target_col,
        date_col=args.date_col,
        include_chronos=args.include_chronos
    )
