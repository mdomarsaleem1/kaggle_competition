"""
Nested Ensemble (Stacking) with Meta-Learner
=============================================

This script implements a two-level ensemble:
- Level 0: Base models (XGBoost, LightGBM, CatBoost, Prophet, Chronos)
- Level 1: Meta-learner that learns to combine base predictions dynamically

Key differences from simple ensembling:
1. Context Injection: Feeds [original_features, base_predictions] to meta-learner
2. Dynamic Weighting: Meta-learner adjusts weights per sample based on features
3. Hold-out Calibration: Uses separate validation set to train meta-learner
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb

from models import (
    XGBoostTimeSeriesModel,
    LightGBMTimeSeriesModel,
    CatBoostTimeSeriesModel,
    ProphetTimeSeriesModel,
    ChronosTimeSeriesModel
)
from utils.data_utils import TimeSeriesPreprocessor, split_time_series


class NestedEnsemble:
    """
    Two-level ensemble with meta-learning

    Level 0: Base models make initial predictions
    Level 1: Meta-learner combines predictions dynamically based on context
    """

    def __init__(self, meta_learner_type: str = 'xgboost', use_original_features: bool = True):
        """
        Initialize nested ensemble

        Args:
            meta_learner_type: Type of meta-learner ('xgboost', 'ridge', 'lasso')
            use_original_features: Whether to include original features in meta-learner
        """
        self.meta_learner_type = meta_learner_type
        self.use_original_features = use_original_features

        # Base models (Level 0)
        self.base_models = {}
        self.base_model_names = ['xgboost', 'lightgbm', 'catboost']

        # Meta-learner (Level 1)
        self.meta_learner = None

        # Preprocessor
        self.preprocessor = None

        # Feature names for debugging
        self.feature_names = []
        self.base_prediction_indices = []

    def _create_meta_learner(self):
        """Create meta-learner model"""
        if self.meta_learner_type == 'xgboost':
            return xgb.XGBRegressor(
                objective='reg:squarederror',
                learning_rate=0.01,
                max_depth=3,  # Shallow to avoid overfitting
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif self.meta_learner_type == 'ridge':
            return Ridge(alpha=1.0, random_state=42)
        elif self.meta_learner_type == 'lasso':
            return Lasso(alpha=0.1, random_state=42)
        else:
            raise ValueError(f"Unknown meta-learner type: {self.meta_learner_type}")

    def train_base_models(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Train all base models

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Dictionary with training metrics
        """
        print("\n" + "="*70)
        print("LEVEL 0: Training Base Models")
        print("="*70)

        metrics = {}

        # Train XGBoost
        print("\n[1/3] Training XGBoost...")
        xgb_model = XGBoostTimeSeriesModel()
        xgb_metrics = xgb_model.train(X_train, y_train, X_val, y_val, verbose=False)
        self.base_models['xgboost'] = xgb_model
        metrics['xgboost'] = xgb_metrics
        print(f"  Val RMSE: {xgb_metrics['val_rmse']:.6f}")

        # Train LightGBM
        print("\n[2/3] Training LightGBM...")
        lgb_model = LightGBMTimeSeriesModel()
        lgb_metrics = lgb_model.train(X_train, y_train, X_val, y_val, verbose=False)
        self.base_models['lightgbm'] = lgb_model
        metrics['lightgbm'] = lgb_metrics
        print(f"  Val RMSE: {lgb_metrics['val_rmse']:.6f}")

        # Train CatBoost
        print("\n[3/3] Training CatBoost...")
        cat_model = CatBoostTimeSeriesModel()
        cat_metrics = cat_model.train(X_train, y_train, X_val, y_val, verbose=False)
        self.base_models['catboost'] = cat_model
        metrics['catboost'] = cat_metrics
        print(f"  Val RMSE: {cat_metrics['val_rmse']:.6f}")

        return metrics

    def generate_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions from all base models

        Args:
            X: Input features

        Returns:
            Array of base predictions (shape: [samples, n_base_models])
        """
        base_preds = []

        for model_name in self.base_model_names:
            if model_name in self.base_models:
                pred = self.base_models[model_name].predict(X)
                base_preds.append(pred)

        return np.column_stack(base_preds)

    def create_meta_features(self, X: np.ndarray, base_predictions: np.ndarray) -> np.ndarray:
        """
        Create features for meta-learner

        Args:
            X: Original features
            base_predictions: Predictions from base models

        Returns:
            Combined feature array for meta-learner
        """
        if self.use_original_features:
            # Combine original features with base predictions
            # This allows meta-learner to learn context-dependent weights
            meta_features = np.hstack([X, base_predictions])
        else:
            # Only use base predictions (traditional stacking)
            meta_features = base_predictions

        return meta_features

    def train_meta_learner(self, X_meta_train: np.ndarray, y_meta_train: np.ndarray,
                          X_meta_val: np.ndarray, y_meta_val: np.ndarray) -> Dict:
        """
        Train meta-learner on base model predictions

        Args:
            X_meta_train: Meta-features for training
            y_meta_train: Training targets
            X_meta_val: Meta-features for validation
            y_meta_val: Validation targets

        Returns:
            Dictionary with training metrics
        """
        print("\n" + "="*70)
        print("LEVEL 1: Training Meta-Learner")
        print("="*70)
        print(f"Meta-learner type: {self.meta_learner_type}")
        print(f"Using original features: {self.use_original_features}")
        print(f"Meta-feature dimensions: {X_meta_train.shape[1]}")

        # Create and train meta-learner
        self.meta_learner = self._create_meta_learner()

        if self.meta_learner_type == 'xgboost':
            self.meta_learner.fit(
                X_meta_train, y_meta_train,
                eval_set=[(X_meta_val, y_meta_val)],
                verbose=False
            )
        else:
            self.meta_learner.fit(X_meta_train, y_meta_train)

        # Evaluate
        train_pred = self.meta_learner.predict(X_meta_train)
        val_pred = self.meta_learner.predict(X_meta_val)

        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_meta_train, train_pred)),
            'train_mae': mean_absolute_error(y_meta_train, train_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_meta_val, val_pred)),
            'val_mae': mean_absolute_error(y_meta_val, val_pred)
        }

        print(f"\nMeta-Learner Results:")
        print(f"  Train RMSE: {metrics['train_rmse']:.6f}")
        print(f"  Val RMSE: {metrics['val_rmse']:.6f}")
        print(f"  Val MAE: {metrics['val_mae']:.6f}")

        # Get feature importance if using XGBoost
        if self.meta_learner_type == 'xgboost':
            self._analyze_meta_learner_importance(X_meta_train.shape[1])

        return metrics

    def _analyze_meta_learner_importance(self, n_features: int):
        """Analyze and display meta-learner feature importance"""
        importance = self.meta_learner.feature_importances_

        # Separate base predictions from original features
        n_base_models = len(self.base_model_names)

        if self.use_original_features:
            n_original = n_features - n_base_models

            print("\n" + "-"*70)
            print("Meta-Learner Feature Importance")
            print("-"*70)

            # Base model importance
            print("\nBase Model Predictions:")
            for i, model_name in enumerate(self.base_model_names):
                idx = n_original + i
                print(f"  {model_name:12s}: {importance[idx]:.4f}")

            # Top original features
            original_importance = importance[:n_original]
            top_indices = np.argsort(original_importance)[-10:][::-1]

            print("\nTop 10 Original Features:")
            for idx in top_indices:
                print(f"  Feature {idx:3d}: {original_importance[idx]:.4f}")
        else:
            print("\n" + "-"*70)
            print("Base Model Importance:")
            for i, model_name in enumerate(self.base_model_names):
                print(f"  {model_name:12s}: {importance[i]:.4f}")

    def train_with_holdout(self, X: np.ndarray, y: np.ndarray,
                          val_split: float = 0.3) -> Dict:
        """
        Train nested ensemble with holdout validation

        Split data into:
        - Base training: Train base models
        - Meta training: Train meta-learner
        - Meta validation: Evaluate meta-learner

        Args:
            X: Full feature set
            y: Full target set
            val_split: Fraction to use for meta-learner training/validation

        Returns:
            Dictionary with all metrics
        """
        # Split for base model training vs meta-learner
        split_idx = int(len(X) * (1 - val_split))

        X_base_train = X[:split_idx]
        y_base_train = y[:split_idx]
        X_meta = X[split_idx:]
        y_meta = y[split_idx:]

        # Further split meta data for training and validation
        meta_split_idx = int(len(X_meta) * 0.5)
        X_meta_train = X_meta[:meta_split_idx]
        y_meta_train = y_meta[:meta_split_idx]
        X_meta_val = X_meta[meta_split_idx:]
        y_meta_val = y_meta[meta_split_idx:]

        print(f"Base training set: {len(X_base_train)} samples")
        print(f"Meta training set: {len(X_meta_train)} samples")
        print(f"Meta validation set: {len(X_meta_val)} samples")

        # Train base models on base training set
        base_metrics = self.train_base_models(
            X_base_train, y_base_train,
            X_meta_train, y_meta_train  # Use meta_train as validation
        )

        # Generate base predictions for meta-learner
        base_preds_meta_train = self.generate_base_predictions(X_meta_train)
        base_preds_meta_val = self.generate_base_predictions(X_meta_val)

        # Create meta-features
        X_meta_learner_train = self.create_meta_features(X_meta_train, base_preds_meta_train)
        X_meta_learner_val = self.create_meta_features(X_meta_val, base_preds_meta_val)

        # Train meta-learner
        meta_metrics = self.train_meta_learner(
            X_meta_learner_train, y_meta_train,
            X_meta_learner_val, y_meta_val
        )

        # Compare with simple averaging
        simple_avg_pred = base_preds_meta_val.mean(axis=1)
        simple_avg_rmse = np.sqrt(mean_squared_error(y_meta_val, simple_avg_pred))

        print("\n" + "="*70)
        print("COMPARISON: Meta-Learner vs Simple Average")
        print("="*70)
        print(f"Simple Average RMSE:   {simple_avg_rmse:.6f}")
        print(f"Meta-Learner RMSE:     {meta_metrics['val_rmse']:.6f}")
        print(f"Improvement:           {((simple_avg_rmse - meta_metrics['val_rmse']) / simple_avg_rmse * 100):.2f}%")

        return {
            'base_models': base_metrics,
            'meta_learner': meta_metrics,
            'simple_average_rmse': simple_avg_rmse
        }

    def train_with_cv(self, X: np.ndarray, y: np.ndarray,
                     n_folds: int = 5) -> Dict:
        """
        Train nested ensemble with time series cross-validation

        This creates out-of-fold predictions for training the meta-learner,
        which helps prevent overfitting.

        Args:
            X: Full feature set
            y: Full target set
            n_folds: Number of CV folds

        Returns:
            Dictionary with all metrics
        """
        print("\n" + "="*70)
        print(f"Training with {n_folds}-Fold Time Series Cross-Validation")
        print("="*70)

        # We'll use expanding window (not shuffled)
        fold_size = len(X) // (n_folds + 1)

        # Store out-of-fold predictions
        oof_base_predictions = np.zeros((len(X), len(self.base_model_names)))
        oof_meta_predictions = np.zeros(len(X))

        fold_metrics = []

        for fold in range(n_folds):
            print(f"\n{'='*70}")
            print(f"FOLD {fold + 1}/{n_folds}")
            print(f"{'='*70}")

            # Expanding window split
            train_end = fold_size * (fold + 1)
            val_start = train_end
            val_end = train_end + fold_size

            if val_end > len(X):
                break

            X_train_fold = X[:train_end]
            y_train_fold = y[:train_end]
            X_val_fold = X[val_start:val_end]
            y_val_fold = y[val_start:val_end]

            print(f"Train: 0-{train_end}, Val: {val_start}-{val_end}")

            # Train base models for this fold
            fold_models = {}

            for i, model_name in enumerate(self.base_model_names):
                if model_name == 'xgboost':
                    model = XGBoostTimeSeriesModel()
                elif model_name == 'lightgbm':
                    model = LightGBMTimeSeriesModel()
                elif model_name == 'catboost':
                    model = CatBoostTimeSeriesModel()

                model.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold, verbose=False)
                fold_models[model_name] = model

                # Store out-of-fold predictions
                oof_base_predictions[val_start:val_end, i] = model.predict(X_val_fold)

            print(f"Fold {fold + 1} base models trained")

        # Now train final base models on all data
        print("\n" + "="*70)
        print("Training Final Base Models on Full Data")
        print("="*70)

        split_idx = int(len(X) * 0.8)
        self.train_base_models(X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:])

        # Train meta-learner on out-of-fold predictions
        print("\n" + "="*70)
        print("Training Meta-Learner on Out-of-Fold Predictions")
        print("="*70)

        # Use later folds for meta-learner training
        meta_start_idx = fold_size * 2
        X_meta_train = X[meta_start_idx:split_idx]
        y_meta_train = y[meta_start_idx:split_idx]
        X_meta_val = X[split_idx:]
        y_meta_val = y[split_idx:]

        # Get base predictions
        base_preds_meta_train = oof_base_predictions[meta_start_idx:split_idx]
        base_preds_meta_val = self.generate_base_predictions(X_meta_val)

        # Create meta-features
        X_meta_learner_train = self.create_meta_features(X_meta_train, base_preds_meta_train)
        X_meta_learner_val = self.create_meta_features(X_meta_val, base_preds_meta_val)

        # Train meta-learner
        meta_metrics = self.train_meta_learner(
            X_meta_learner_train, y_meta_train,
            X_meta_learner_val, y_meta_val
        )

        return {'meta_learner': meta_metrics}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using nested ensemble

        Args:
            X: Input features

        Returns:
            Final predictions from meta-learner
        """
        # Get base predictions
        base_predictions = self.generate_base_predictions(X)

        # Create meta-features
        meta_features = self.create_meta_features(X, base_predictions)

        # Meta-learner prediction
        return self.meta_learner.predict(meta_features)

    def save(self, output_dir: str):
        """Save all models"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Save base models
        for name, model in self.base_models.items():
            if name == 'xgboost':
                model.save_model(str(output_path / 'xgboost_base.json'))
            elif name == 'lightgbm':
                model.save_model(str(output_path / 'lightgbm_base.txt'))
            elif name == 'catboost':
                model.save_model(str(output_path / 'catboost_base.cbm'))

        # Save meta-learner
        if self.meta_learner_type == 'xgboost':
            self.meta_learner.save_model(str(output_path / 'meta_learner.json'))
        else:
            joblib.dump(self.meta_learner, output_path / 'meta_learner.pkl')

        # Save configuration
        config = {
            'meta_learner_type': self.meta_learner_type,
            'use_original_features': self.use_original_features,
            'base_model_names': self.base_model_names
        }
        with open(output_path / 'ensemble_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nNested ensemble saved to {output_dir}")

    def load(self, input_dir: str):
        """Load all models"""
        input_path = Path(input_dir)

        # Load configuration
        with open(input_path / 'ensemble_config.json', 'r') as f:
            config = json.load(f)

        self.meta_learner_type = config['meta_learner_type']
        self.use_original_features = config['use_original_features']
        self.base_model_names = config['base_model_names']

        # Load base models
        for name in self.base_model_names:
            if name == 'xgboost':
                model = XGBoostTimeSeriesModel()
                model.load_model(str(input_path / 'xgboost_base.json'))
                self.base_models[name] = model
            elif name == 'lightgbm':
                model = LightGBMTimeSeriesModel()
                model.load_model(str(input_path / 'lightgbm_base.txt'))
                self.base_models[name] = model
            elif name == 'catboost':
                model = CatBoostTimeSeriesModel()
                model.load_model(str(input_path / 'catboost_base.cbm'))
                self.base_models[name] = model

        # Load meta-learner
        if self.meta_learner_type == 'xgboost':
            self.meta_learner = xgb.XGBRegressor()
            self.meta_learner.load_model(str(input_path / 'meta_learner.json'))
        else:
            self.meta_learner = joblib.load(input_path / 'meta_learner.pkl')

        print(f"\nNested ensemble loaded from {input_dir}")


def main():
    """Main training script"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Train nested ensemble with meta-learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with holdout validation
  python scripts/nested_ensemble_predict.py --data-dir data --train-file train.csv

  # Train with cross-validation
  python scripts/nested_ensemble_predict.py --data-dir data --train-file train.csv --use-cv --n-folds 5

  # Use Ridge meta-learner without original features (traditional stacking)
  python scripts/nested_ensemble_predict.py --data-dir data --train-file train.csv --meta-learner ridge --no-original-features
        """
    )

    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing data files')
    parser.add_argument('--train-file', type=str, default='train.csv',
                       help='Training data filename')
    parser.add_argument('--target-col', type=str, default='target',
                       help='Target column name')
    parser.add_argument('--date-col', type=str, default='date',
                       help='Date column name')
    parser.add_argument('--output-dir', type=str, default='nested_ensemble_models',
                       help='Directory to save trained models')
    parser.add_argument('--meta-learner', type=str, default='xgboost',
                       choices=['xgboost', 'ridge', 'lasso'],
                       help='Type of meta-learner')
    parser.add_argument('--no-original-features', action='store_true',
                       help='Do not use original features in meta-learner (traditional stacking)')
    parser.add_argument('--use-cv', action='store_true',
                       help='Use cross-validation instead of holdout')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of CV folds (only if --use-cv is set)')
    parser.add_argument('--val-split', type=float, default=0.3,
                       help='Validation split fraction (only for holdout)')

    args = parser.parse_args()

    # Load and prepare data
    print("="*70)
    print("NESTED ENSEMBLE TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Meta-learner: {args.meta_learner}")
    print(f"  Use original features: {not args.no_original_features}")
    print(f"  Training method: {'Cross-validation' if args.use_cv else 'Holdout'}")

    preprocessor = TimeSeriesPreprocessor(scaler_type='standard')

    print(f"\nLoading data from {args.data_dir}/{args.train_file}...")
    train_df, _ = preprocessor.load_data(str(Path(args.data_dir) / args.train_file))

    print("Creating features...")
    df_features = preprocessor.create_all_features(train_df, args.target_col)

    # Prepare features
    feature_cols = [col for col in df_features.columns
                   if col not in [args.date_col, args.target_col]]

    X = df_features[feature_cols].values
    y = df_features[args.target_col].values

    # Scale features
    X = preprocessor.fit_transform(X)

    print(f"Data shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Create and train nested ensemble
    ensemble = NestedEnsemble(
        meta_learner_type=args.meta_learner,
        use_original_features=not args.no_original_features
    )

    if args.use_cv:
        metrics = ensemble.train_with_cv(X, y, n_folds=args.n_folds)
    else:
        metrics = ensemble.train_with_holdout(X, y, val_split=args.val_split)

    # Save ensemble
    ensemble.save(args.output_dir)

    # Save preprocessor
    joblib.dump(preprocessor, Path(args.output_dir) / 'preprocessor.pkl')

    # Save metrics
    with open(Path(args.output_dir) / 'training_metrics.json', 'w') as f:
        # Convert numpy types for JSON serialization
        serializable_metrics = json.loads(
            json.dumps(metrics, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
        )
        json.dump(serializable_metrics, f, indent=2)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Models saved to: {args.output_dir}")
    print("\nTo make predictions, use:")
    print(f"  python scripts/nested_ensemble_predict.py --mode predict --model-dir {args.output_dir} --test-file data/test.csv")


if __name__ == '__main__':
    main()
