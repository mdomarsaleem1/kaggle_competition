"""
Universal Nested Ensemble: Supports ALL Models
===============================================

This script creates a powerful ensemble that combines:
- Tree-based models: XGBoost, LightGBM, CatBoost
- Statistical models: Prophet
- Foundation models: Chronos-2
- Transformer models: PatchTST, iTransformer, TimesNet

The meta-learner dynamically weighs all models based on context.
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
from sklearn.linear_model import Ridge
import xgboost as xgb
import torch

from models import (
    # Tree-based
    XGBoostTimeSeriesModel,
    LightGBMTimeSeriesModel,
    CatBoostTimeSeriesModel,
    # Transformers
    PatchTSTTimeSeriesModel,
    iTransformerTimeSeriesModel,
    TimesNetTimeSeriesModel,
)
from utils.data_utils import TimeSeriesPreprocessor


class UniversalNestedEnsemble:
    """
    Universal nested ensemble supporting all model types

    Combines:
    - Tree-based models (work on tabular features)
    - Transformer models (work on sequential data)
    """

    def __init__(self,
                 meta_learner_type: str = 'xgboost',
                 use_tree_models: bool = True,
                 use_transformer_models: bool = True,
                 seq_len: int = 96,
                 pred_len: int = 24,
                 transformer_epochs: int = 50,
                 device: Optional[str] = None):
        """
        Initialize universal ensemble

        Args:
            meta_learner_type: Type of meta-learner ('xgboost', 'ridge')
            use_tree_models: Whether to include tree-based models
            use_transformer_models: Whether to include transformer models
            seq_len: Sequence length for transformers
            pred_len: Prediction length
            transformer_epochs: Training epochs for transformers
            device: Device for transformers ('cpu', 'cuda', or None)
        """
        self.meta_learner_type = meta_learner_type
        self.use_tree_models = use_tree_models
        self.use_transformer_models = use_transformer_models
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.transformer_epochs = transformer_epochs
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Models
        self.tree_models = {}
        self.transformer_models = {}
        self.meta_learner = None

        # Model names
        self.tree_model_names = ['xgboost', 'lightgbm', 'catboost'] if use_tree_models else []
        self.transformer_model_names = ['patchtst', 'itransformer', 'timesnet'] if use_transformer_models else []
        self.all_model_names = self.tree_model_names + self.transformer_model_names

        print(f"\nUniversal Nested Ensemble Configuration:")
        print(f"  Tree models: {self.tree_model_names}")
        print(f"  Transformer models: {self.transformer_model_names}")
        print(f"  Meta-learner: {meta_learner_type}")
        print(f"  Device: {self.device}")

    def _create_meta_learner(self):
        """Create meta-learner"""
        if self.meta_learner_type == 'xgboost':
            return xgb.XGBRegressor(
                objective='reg:squarederror',
                learning_rate=0.01,
                max_depth=3,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif self.meta_learner_type == 'ridge':
            return Ridge(alpha=1.0, random_state=42)
        else:
            raise ValueError(f"Unknown meta-learner: {self.meta_learner_type}")

    def train_tree_models(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train tree-based models"""
        if not self.use_tree_models:
            return {}

        print("\n" + "="*70)
        print("LEVEL 0A: Training Tree-Based Models")
        print("="*70)

        metrics = {}

        # XGBoost
        print("\n[1/3] Training XGBoost...")
        xgb_model = XGBoostTimeSeriesModel()
        xgb_metrics = xgb_model.train(X_train, y_train, X_val, y_val, verbose=False)
        self.tree_models['xgboost'] = xgb_model
        metrics['xgboost'] = xgb_metrics
        print(f"  Val RMSE: {xgb_metrics['val_rmse']:.6f}")

        # LightGBM
        print("\n[2/3] Training LightGBM...")
        lgb_model = LightGBMTimeSeriesModel()
        lgb_metrics = lgb_model.train(X_train, y_train, X_val, y_val, verbose=False)
        self.tree_models['lightgbm'] = lgb_model
        metrics['lightgbm'] = lgb_metrics
        print(f"  Val RMSE: {lgb_metrics['val_rmse']:.6f}")

        # CatBoost
        print("\n[3/3] Training CatBoost...")
        cat_model = CatBoostTimeSeriesModel()
        cat_metrics = cat_model.train(X_train, y_train, X_val, y_val, verbose=False)
        self.tree_models['catboost'] = cat_model
        metrics['catboost'] = cat_metrics
        print(f"  Val RMSE: {cat_metrics['val_rmse']:.6f}")

        return metrics

    def train_transformer_models(self, train_data: np.ndarray, val_data: np.ndarray) -> Dict:
        """Train transformer models on sequential data"""
        if not self.use_transformer_models:
            return {}

        print("\n" + "="*70)
        print("LEVEL 0B: Training Transformer Models")
        print("="*70)

        metrics = {}
        n_features = train_data.shape[1] if train_data.ndim > 1 else 1

        # PatchTST
        print("\n[1/3] Training PatchTST...")
        patchtst = PatchTSTTimeSeriesModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            n_features=n_features,
            d_model=128,
            n_heads=8,
            n_layers=3,
            epochs=self.transformer_epochs,
            device=self.device
        )
        patchtst_metrics = patchtst.train(train_data, val_data, verbose=False)
        self.transformer_models['patchtst'] = patchtst
        metrics['patchtst'] = patchtst_metrics
        print(f"  Val RMSE: {patchtst_metrics.get('val_rmse', 'N/A')}")

        # iTransformer
        print("\n[2/3] Training iTransformer...")
        itransformer = iTransformerTimeSeriesModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            n_features=n_features,
            d_model=512,
            n_heads=8,
            n_layers=2,
            epochs=self.transformer_epochs,
            device=self.device
        )
        itransformer_metrics = itransformer.train(train_data, val_data, verbose=False)
        self.transformer_models['itransformer'] = itransformer
        metrics['itransformer'] = itransformer_metrics
        print(f"  Val RMSE: {itransformer_metrics.get('val_rmse', 'N/A')}")

        # TimesNet
        print("\n[3/3] Training TimesNet...")
        timesnet = TimesNetTimeSeriesModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            n_features=n_features,
            d_model=64,
            n_layers=2,
            epochs=self.transformer_epochs,
            device=self.device
        )
        timesnet_metrics = timesnet.train(train_data, val_data, verbose=False)
        self.transformer_models['timesnet'] = timesnet
        metrics['timesnet'] = timesnet_metrics
        print(f"  Val RMSE: {timesnet_metrics.get('val_rmse', 'N/A')}")

        return metrics

    def generate_predictions(self, X_tabular: Optional[np.ndarray],
                           sequential_data: Optional[np.ndarray]) -> np.ndarray:
        """
        Generate predictions from all models

        Args:
            X_tabular: Tabular features for tree models [samples, features]
            sequential_data: Sequential data for transformers [samples, seq_len]

        Returns:
            Predictions array [samples, n_models]
        """
        all_predictions = []

        # Tree model predictions
        if self.use_tree_models and X_tabular is not None:
            for name in self.tree_model_names:
                if name in self.tree_models:
                    pred = self.tree_models[name].predict(X_tabular)
                    all_predictions.append(pred)

        # Transformer predictions
        if self.use_transformer_models and sequential_data is not None:
            for name in self.transformer_model_names:
                if name in self.transformer_models:
                    pred = self.transformer_models[name].predict(sequential_data, return_sequences=False)
                    # If multivariate, take mean or first column
                    if pred.ndim > 1:
                        pred = pred.mean(axis=1) if pred.shape[1] > 1 else pred[:, 0]
                    all_predictions.append(pred)

        return np.column_stack(all_predictions)

    def train_meta_learner(self, X_meta_train: np.ndarray, y_meta_train: np.ndarray,
                          X_meta_val: np.ndarray, y_meta_val: np.ndarray) -> Dict:
        """Train meta-learner"""
        print("\n" + "="*70)
        print("LEVEL 1: Training Meta-Learner on ALL Models")
        print("="*70)
        print(f"Meta-learner type: {self.meta_learner_type}")
        print(f"Number of base models: {X_meta_train.shape[1]}")
        print(f"Models: {', '.join(self.all_model_names)}")

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
        print(f"  Val RMSE: {metrics['val_rmse']:.6f}")
        print(f"  Val MAE: {metrics['val_mae']:.6f}")

        # Feature importance
        if self.meta_learner_type == 'xgboost':
            self._analyze_meta_learner_importance()

        return metrics

    def _analyze_meta_learner_importance(self):
        """Analyze meta-learner feature importance"""
        importance = self.meta_learner.feature_importances_

        print("\n" + "-"*70)
        print("Model Importance in Meta-Learner:")
        print("-"*70)

        for i, model_name in enumerate(self.all_model_names):
            if i < len(importance):
                print(f"  {model_name:15s}: {importance[i]:.4f}")

    def _build_sequential_matrix(self, df: pd.DataFrame, target_col: str, date_col: Optional[str]) -> np.ndarray:
        """Create sequential matrix with numeric date indices."""
        target_values = df[target_col].values.reshape(-1, 1)

        if date_col and date_col in df.columns:
            date_numeric = df[date_col].dt.toordinal().values.reshape(-1, 1)
            return np.hstack([target_values, date_numeric])

        return target_values

    def train(self, full_data: pd.DataFrame, target_col: str = 'target',
             val_split: float = 0.3, preprocessor: Optional[TimeSeriesPreprocessor] = None,
             date_col: str = 'date') -> Dict:
        """
        Train universal ensemble

        Args:
            full_data: Full dataframe with time series
            target_col: Target column name
            val_split: Validation split fraction
            preprocessor: Preprocessor for feature engineering

        Returns:
            Training metrics
        """
        # Split data
        split_idx = int(len(full_data) * (1 - val_split))
        train_df = full_data.iloc[:split_idx]
        val_df = full_data.iloc[split_idx:]

        print(f"\nData split:")
        print(f"  Training: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")

        # Prepare sequential data for transformers using numeric date index
        train_sequential = self._build_sequential_matrix(train_df, target_col, date_col)
        val_sequential = self._build_sequential_matrix(val_df, target_col, date_col)

        # Prepare tabular data for tree models
        X_train_tabular, y_train, X_val_tabular, y_val = None, None, None, None

        if self.use_tree_models and preprocessor:
            # Create features
            df_features = preprocessor.create_all_features(full_data, target_col)

            # Split
            train_features = df_features.iloc[:split_idx]
            val_features = df_features.iloc[split_idx:]

            # Extract features
            feature_cols = [col for col in df_features.columns if col not in ['date', target_col]]
            X_train_tabular = train_features[feature_cols].values
            y_train = train_features[target_col].values
            X_val_tabular = val_features[feature_cols].values
            y_val = val_features[target_col].values

            # Scale
            X_train_tabular = preprocessor.fit_transform(X_train_tabular)
            X_val_tabular = preprocessor.transform(X_val_tabular)

        # Train base models
        tree_metrics = self.train_tree_models(X_train_tabular, y_train, X_val_tabular, y_val) if self.use_tree_models else {}
        transformer_metrics = self.train_transformer_models(train_sequential, val_sequential) if self.use_transformer_models else {}

        # Generate predictions for meta-learner training
        # We need to align transformer and tree predictions
        if self.use_transformer_models:
            # Get transformer predictions on validation set (already returns point forecasts)
            meta_train_size = len(val_sequential) // 2
            meta_train_seq = val_sequential[:meta_train_size]
            meta_val_seq = val_sequential[meta_train_size:]

            # Generate base predictions
            base_preds_train = self.generate_predictions(
                X_val_tabular[:meta_train_size] if X_val_tabular is not None else None,
                meta_train_seq if meta_train_seq is not None else None
            )

            base_preds_val = self.generate_predictions(
                X_val_tabular[meta_train_size:] if X_val_tabular is not None else None,
                meta_val_seq if meta_val_seq is not None else None
            )

            # Targets for meta-learner
            y_meta_train = y_val[:meta_train_size] if y_val is not None else val_sequential[:meta_train_size, 0]
            y_meta_val = y_val[meta_train_size:] if y_val is not None else val_sequential[meta_train_size:, 0]

        else:
            # Only tree models
            meta_train_size = len(y_val) // 2
            base_preds_train = self.generate_predictions(X_val_tabular[:meta_train_size], None)
            base_preds_val = self.generate_predictions(X_val_tabular[meta_train_size:], None)
            y_meta_train = y_val[:meta_train_size]
            y_meta_val = y_val[meta_train_size:]

        # Train meta-learner
        meta_metrics = self.train_meta_learner(
            base_preds_train, y_meta_train,
            base_preds_val, y_meta_val
        )

        # Compare with simple averaging
        simple_avg = base_preds_val.mean(axis=1)
        simple_rmse = np.sqrt(mean_squared_error(y_meta_val, simple_avg))

        print("\n" + "="*70)
        print("COMPARISON: Meta-Learner vs Simple Average")
        print("="*70)
        print(f"Simple Average RMSE:   {simple_rmse:.6f}")
        print(f"Meta-Learner RMSE:     {meta_metrics['val_rmse']:.6f}")
        improvement = ((simple_rmse - meta_metrics['val_rmse']) / simple_rmse * 100)
        print(f"Improvement:           {improvement:+.2f}%")

        return {
            'tree_models': tree_metrics,
            'transformer_models': transformer_metrics,
            'meta_learner': meta_metrics,
            'simple_average_rmse': simple_rmse,
            'improvement_pct': improvement
        }

    def predict(self, X_tabular: Optional[np.ndarray],
               sequential_data: Optional[np.ndarray]) -> np.ndarray:
        """Make predictions using ensemble"""
        # Get base predictions
        base_predictions = self.generate_predictions(X_tabular, sequential_data)

        # Meta-learner prediction
        return self.meta_learner.predict(base_predictions)

    def save(self, output_dir: str):
        """Save all models"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Save tree models
        for name, model in self.tree_models.items():
            if name == 'xgboost':
                model.save_model(str(output_path / 'xgboost_base.json'))
            elif name == 'lightgbm':
                model.save_model(str(output_path / 'lightgbm_base.txt'))
            elif name == 'catboost':
                model.save_model(str(output_path / 'catboost_base.cbm'))

        # Save transformer models
        for name, model in self.transformer_models.items():
            model.save_model(str(output_path / f'{name}_base.pth'))

        # Save meta-learner
        if self.meta_learner_type == 'xgboost':
            self.meta_learner.save_model(str(output_path / 'meta_learner.json'))
        else:
            joblib.dump(self.meta_learner, output_path / 'meta_learner.pkl')

        # Save configuration
        config = {
            'meta_learner_type': self.meta_learner_type,
            'use_tree_models': self.use_tree_models,
            'use_transformer_models': self.use_transformer_models,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'tree_model_names': self.tree_model_names,
            'transformer_model_names': self.transformer_model_names
        }

        with open(output_path / 'ensemble_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nUniversal ensemble saved to {output_dir}")


def main():
    """Main training script"""
    import argparse

    parser = argparse.ArgumentParser(description='Train universal nested ensemble')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--train-file', type=str, default='train.csv')
    parser.add_argument('--target-col', type=str, default='target')
    parser.add_argument('--date-col', type=str, default='date')
    parser.add_argument('--output-dir', type=str, default='universal_ensemble_models')
    parser.add_argument('--meta-learner', type=str, default='xgboost', choices=['xgboost', 'ridge'])
    parser.add_argument('--no-tree-models', action='store_true', help='Exclude tree models')
    parser.add_argument('--no-transformers', action='store_true', help='Exclude transformers')
    parser.add_argument('--seq-len', type=int, default=96, help='Sequence length for transformers')
    parser.add_argument('--pred-len', type=int, default=24, help='Prediction length')
    parser.add_argument('--transformer-epochs', type=int, default=50, help='Transformer training epochs')
    parser.add_argument('--val-split', type=float, default=0.3, help='Validation split')

    args = parser.parse_args()

    print("="*70)
    print("UNIVERSAL NESTED ENSEMBLE TRAINING")
    print("Combining Tree-Based + Transformer Models")
    print("="*70)

    # Load data
    preprocessor = TimeSeriesPreprocessor(scaler_type='standard')
    print(f"\nLoading data from {args.data_dir}/{args.train_file}...")
    train_df, _ = preprocessor.load_data(str(Path(args.data_dir) / args.train_file))

    # Create ensemble
    ensemble = UniversalNestedEnsemble(
        meta_learner_type=args.meta_learner,
        use_tree_models=not args.no_tree_models,
        use_transformer_models=not args.no_transformers,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        transformer_epochs=args.transformer_epochs
    )

    # Train
    metrics = ensemble.train(
        train_df,
        target_col=args.target_col,
        val_split=args.val_split,
        preprocessor=preprocessor if not args.no_tree_models else None,
        date_col=args.date_col
    )

    # Save
    ensemble.save(args.output_dir)
    joblib.dump(preprocessor, Path(args.output_dir) / 'preprocessor.pkl')

    # Save metrics
    with open(Path(args.output_dir) / 'training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Models saved to: {args.output_dir}")
    print(f"Final improvement: {metrics.get('improvement_pct', 0):.2f}%")


if __name__ == '__main__':
    main()
