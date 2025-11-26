"""
Ultimate Meta-Ensemble: ALL Models + Meta-Learning
===================================================

This is the most comprehensive ensemble combining:

Level 0A: Tree-Based Models (3)
├── XGBoost
├── LightGBM
└── CatBoost

Level 0B: Transformer Models (3)
├── PatchTST
├── iTransformer
└── TimesNet

Level 0C: Foundation Models (1)
└── TimesFM (Google's pre-trained foundation model)

Level 0D: Hybrid Models (1)
└── Chronos-PatchTST (with covariate injection)

Level 1: Meta-Learner
└── XGBoost/Ridge (learns to combine ALL base predictions)

Total: 8+ diverse models → 1 powerful prediction
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
    # Foundation
    TimesFMTimeSeriesModel,
    # Hybrid
    HybridChronosPatchTSTModel
)
from utils.data_utils import TimeSeriesPreprocessor


class UltimateMetaEnsemble:
    """
    The ultimate ensemble combining ALL model types with meta-learning

    This ensemble represents the state-of-the-art by combining:
    - Classical ML (tree-based models)
    - Deep Learning (transformers)
    - Foundation models (Chronos-2)
    - Hybrid approaches (covariate injection)
    - Meta-learning (context-aware combination)
    """

    def __init__(self,
                 seq_len: int = 96,
                 pred_len: int = 24,
                 transformer_epochs: int = 50,
                 meta_learner_type: str = 'xgboost',
                 include_tree_models: bool = True,
                 include_transformers: bool = True,
                 include_foundation: bool = True,
                 include_hybrid: bool = True,
                 device: Optional[str] = None):
        """
        Initialize ultimate ensemble

        Args:
            seq_len: Sequence length
            pred_len: Prediction length
            transformer_epochs: Transformer training epochs
            meta_learner_type: 'xgboost' or 'ridge'
            include_tree_models: Include tree-based models
            include_transformers: Include transformer models
            include_foundation: Include foundation models (TimesFM)
            include_hybrid: Include hybrid Chronos-PatchTST
            device: Device for deep learning models
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.transformer_epochs = transformer_epochs
        self.meta_learner_type = meta_learner_type
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Model containers
        self.tree_models = {}
        self.transformer_models = {}
        self.foundation_models = {}
        self.hybrid_model = None
        self.meta_learner = None

        # Configuration
        self.include_tree_models = include_tree_models
        self.include_transformers = include_transformers
        self.include_foundation = include_foundation
        self.include_hybrid = include_hybrid

        # Model names
        self.all_model_names = []
        if include_tree_models:
            self.all_model_names.extend(['xgboost', 'lightgbm', 'catboost'])
        if include_transformers:
            self.all_model_names.extend(['patchtst', 'itransformer', 'timesnet'])
        if include_foundation:
            self.all_model_names.append('timesfm')
        if include_hybrid:
            self.all_model_names.append('hybrid_chronos_patchtst')

        print("\n" + "="*70)
        print("ULTIMATE META-ENSEMBLE")
        print("="*70)
        print(f"Models included ({len(self.all_model_names)}):")
        for i, name in enumerate(self.all_model_names, 1):
            print(f"  {i}. {name}")
        print(f"\nMeta-learner: {meta_learner_type}")
        print(f"Device: {self.device}")
        print("="*70)

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
        if not self.include_tree_models:
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
        """Train transformer models"""
        if not self.include_transformers:
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

    def train_foundation_models(self, train_data: np.ndarray, val_data: np.ndarray) -> Dict:
        """Train foundation models (TimesFM)"""
        if not self.include_foundation:
            return {}

        print("\n" + "="*70)
        print("LEVEL 0C: Training Foundation Models")
        print("="*70)

        metrics = {}

        # TimesFM
        print("\n[1/1] Loading TimesFM (pre-trained)...")
        timesfm = TimesFMTimeSeriesModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            model_size='base',
            device=self.device
        )

        # TimesFM is pre-trained, so we just "load" it (evaluation on val_data)
        timesfm_metrics = timesfm.train(train_data, val_data, verbose=False)
        self.foundation_models['timesfm'] = timesfm
        metrics['timesfm'] = timesfm_metrics
        print(f"  TimesFM loaded (pre-trained foundation model)")
        if 'val_rmse' in timesfm_metrics:
            print(f"  Val RMSE: {timesfm_metrics['val_rmse']:.6f}")

        return metrics

    def train_hybrid_model(self, train_data: np.ndarray, val_data: np.ndarray) -> Dict:
        """Train hybrid Chronos-PatchTST model"""
        if not self.include_hybrid:
            return {}

        print("\n" + "="*70)
        print("LEVEL 0D: Training Hybrid Chronos-PatchTST Model")
        print("="*70)

        n_features = train_data.shape[1] if train_data.ndim > 1 else 1

        self.hybrid_model = HybridChronosPatchTSTModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            n_features=n_features,
            patchtst_epochs=self.transformer_epochs,
            chronos_model_size='small',
            device=self.device
        )

        # Train PatchTST specialist
        hybrid_metrics = self.hybrid_model.train_patchtst(train_data, val_data, verbose=False)

        return {'hybrid_chronos_patchtst': hybrid_metrics}

    def generate_all_predictions(self, X_tabular: Optional[np.ndarray],
                                sequential_data: Optional[np.ndarray]) -> np.ndarray:
        """
        Generate predictions from ALL models

        Args:
            X_tabular: Tabular features for tree models
            sequential_data: Sequential data for transformers

        Returns:
            Predictions array [samples, n_models]
        """
        all_predictions = []

        # Tree model predictions
        if self.include_tree_models and X_tabular is not None:
            for name in ['xgboost', 'lightgbm', 'catboost']:
                if name in self.tree_models:
                    pred = self.tree_models[name].predict(X_tabular)
                    all_predictions.append(pred)

        # Transformer predictions
        if self.include_transformers and sequential_data is not None:
            for name in ['patchtst', 'itransformer', 'timesnet']:
                if name in self.transformer_models:
                    pred = self.transformer_models[name].predict(sequential_data, return_sequences=False)
                    # Handle multivariate
                    if pred.ndim > 1:
                        pred = pred.mean(axis=1) if pred.shape[1] > 1 else pred[:, 0]
                    all_predictions.append(pred)

        # Foundation model predictions
        if self.include_foundation and sequential_data is not None:
            for name in ['timesfm']:
                if name in self.foundation_models:
                    # Use only the last seq_len points for foundation model
                    context = sequential_data[-self.seq_len:]
                    if context.ndim == 1:
                        context = context.reshape(-1, 1)

                    try:
                        timesfm_pred = self.foundation_models[name].predict(context)
                        # Repeat prediction for all samples if needed
                        if len(all_predictions) > 0:
                            target_len = len(all_predictions[0])
                            if len(timesfm_pred) != target_len:
                                timesfm_pred_repeated = np.full(target_len, timesfm_pred[0] if hasattr(timesfm_pred, '__len__') else timesfm_pred)
                                all_predictions.append(timesfm_pred_repeated)
                            else:
                                all_predictions.append(timesfm_pred)
                        else:
                            all_predictions.append(timesfm_pred)
                    except Exception as e:
                        print(f"Warning: TimesFM prediction failed: {e}")

        # Hybrid model predictions
        if self.include_hybrid and self.hybrid_model and sequential_data is not None:
            # Use only the last seq_len points for hybrid model
            context = sequential_data[-self.seq_len:]
            if context.ndim == 1:
                context = context.reshape(-1, 1)

            try:
                hybrid_pred = self.hybrid_model.predict(context)
                # Repeat prediction for all samples (since hybrid gives one prediction)
                hybrid_pred_repeated = np.full(len(all_predictions[0]), hybrid_pred[0] if hasattr(hybrid_pred, '__len__') else hybrid_pred)
                all_predictions.append(hybrid_pred_repeated)
            except Exception as e:
                print(f"Warning: Hybrid prediction failed: {e}")

        if not all_predictions:
            raise ValueError("No predictions generated!")

        return np.column_stack(all_predictions)

    def train_meta_learner(self, base_predictions: np.ndarray, targets: np.ndarray,
                          val_predictions: np.ndarray, val_targets: np.ndarray) -> Dict:
        """Train meta-learner on all base predictions"""
        print("\n" + "="*70)
        print(f"LEVEL 1: Training Meta-Learner on {base_predictions.shape[1]} Models")
        print("="*70)
        print(f"Models: {', '.join(self.all_model_names)}")

        # Create meta-learner
        self.meta_learner = self._create_meta_learner()

        if self.meta_learner_type == 'xgboost':
            self.meta_learner.fit(
                base_predictions, targets,
                eval_set=[(val_predictions, val_targets)],
                verbose=False
            )
        else:
            self.meta_learner.fit(base_predictions, targets)

        # Evaluate
        train_pred = self.meta_learner.predict(base_predictions)
        val_pred = self.meta_learner.predict(val_predictions)

        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(targets, train_pred)),
            'train_mae': mean_absolute_error(targets, train_pred),
            'val_rmse': np.sqrt(mean_squared_error(val_targets, val_pred)),
            'val_mae': mean_absolute_error(val_targets, val_pred)
        }

        print(f"\nMeta-Learner Results:")
        print(f"  Val RMSE: {metrics['val_rmse']:.6f}")
        print(f"  Val MAE: {metrics['val_mae']:.6f}")

        # Feature importance
        if self.meta_learner_type == 'xgboost':
            self._analyze_importance()

        # Compare with simple averaging
        simple_avg = val_predictions.mean(axis=1)
        simple_rmse = np.sqrt(mean_squared_error(val_targets, simple_avg))

        print("\n" + "="*70)
        print("COMPARISON: Meta-Learner vs Simple Average")
        print("="*70)
        print(f"Simple Average RMSE:   {simple_rmse:.6f}")
        print(f"Meta-Learner RMSE:     {metrics['val_rmse']:.6f}")
        improvement = ((simple_rmse - metrics['val_rmse']) / simple_rmse * 100)
        print(f"Improvement:           {improvement:+.2f}%")

        metrics['simple_average_rmse'] = simple_rmse
        metrics['improvement_pct'] = improvement

        return metrics

    def _analyze_importance(self):
        """Analyze meta-learner feature importance"""
        importance = self.meta_learner.feature_importances_

        print("\n" + "-"*70)
        print("Model Importance in Ultimate Meta-Ensemble:")
        print("-"*70)

        for i, model_name in enumerate(self.all_model_names):
            if i < len(importance):
                print(f"  {model_name:25s}: {importance[i]:.4f}")

    def train(self, full_data: pd.DataFrame, target_col: str = 'target',
             val_split: float = 0.3, preprocessor: Optional[TimeSeriesPreprocessor] = None) -> Dict:
        """
        Train the complete ultimate ensemble

        Args:
            full_data: Full dataframe
            target_col: Target column name
            val_split: Validation split
            preprocessor: Preprocessor for tree models

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

        # Prepare sequential data
        train_sequential = train_df[target_col].values
        val_sequential = val_df[target_col].values

        if train_sequential.ndim == 1:
            train_sequential = train_sequential.reshape(-1, 1)
            val_sequential = val_sequential.reshape(-1, 1)

        # Prepare tabular data for tree models
        X_train_tabular, y_train, X_val_tabular, y_val = None, None, None, None

        if self.include_tree_models and preprocessor:
            df_features = preprocessor.create_all_features(full_data, target_col)
            train_features = df_features.iloc[:split_idx]
            val_features = df_features.iloc[split_idx:]

            feature_cols = [col for col in df_features.columns if col not in ['date', target_col]]
            X_train_tabular = train_features[feature_cols].values
            y_train = train_features[target_col].values
            X_val_tabular = val_features[feature_cols].values
            y_val = val_features[target_col].values

            X_train_tabular = preprocessor.fit_transform(X_train_tabular)
            X_val_tabular = preprocessor.transform(X_val_tabular)

        # Train all base models
        tree_metrics = self.train_tree_models(X_train_tabular, y_train, X_val_tabular, y_val)
        transformer_metrics = self.train_transformer_models(train_sequential, val_sequential)
        foundation_metrics = self.train_foundation_models(train_sequential, val_sequential)
        hybrid_metrics = self.train_hybrid_model(train_sequential, val_sequential)

        # Generate base predictions for meta-learner
        # Use second half of validation set for meta-learner training
        meta_train_size = len(val_sequential) // 2
        meta_train_seq = val_sequential[:meta_train_size]
        meta_val_seq = val_sequential[meta_train_size:]

        # Generate predictions
        base_preds_train = self.generate_all_predictions(
            X_val_tabular[:meta_train_size] if X_val_tabular is not None else None,
            meta_train_seq
        )

        base_preds_val = self.generate_all_predictions(
            X_val_tabular[meta_train_size:] if X_val_tabular is not None else None,
            meta_val_seq
        )

        # Targets
        y_meta_train = y_val[:meta_train_size] if y_val is not None else val_sequential[:meta_train_size, 0]
        y_meta_val = y_val[meta_train_size:] if y_val is not None else val_sequential[meta_train_size:, 0]

        # Train meta-learner
        meta_metrics = self.train_meta_learner(
            base_preds_train, y_meta_train,
            base_preds_val, y_meta_val
        )

        return {
            'tree_models': tree_metrics,
            'transformer_models': transformer_metrics,
            'foundation_models': foundation_metrics,
            'hybrid_model': hybrid_metrics,
            'meta_learner': meta_metrics
        }

    def predict(self, X_tabular: Optional[np.ndarray],
               sequential_data: Optional[np.ndarray]) -> np.ndarray:
        """Make predictions using ultimate ensemble"""
        base_predictions = self.generate_all_predictions(X_tabular, sequential_data)
        return self.meta_learner.predict(base_predictions)

    def save(self, output_dir: str):
        """Save all models"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Save tree models
        for name, model in self.tree_models.items():
            if name == 'xgboost':
                model.save_model(str(output_path / 'xgboost_ultimate.json'))
            elif name == 'lightgbm':
                model.save_model(str(output_path / 'lightgbm_ultimate.txt'))
            elif name == 'catboost':
                model.save_model(str(output_path / 'catboost_ultimate.cbm'))

        # Save transformer models
        for name, model in self.transformer_models.items():
            model.save_model(str(output_path / f'{name}_ultimate.pth'))

        # Save foundation models
        for name, model in self.foundation_models.items():
            model.save_model(str(output_path / f'{name}_ultimate_config.json'))

        # Save hybrid model
        if self.hybrid_model:
            self.hybrid_model.save_models(str(output_path / 'hybrid'))

        # Save meta-learner
        if self.meta_learner_type == 'xgboost':
            self.meta_learner.save_model(str(output_path / 'meta_learner_ultimate.json'))
        else:
            joblib.dump(self.meta_learner, output_path / 'meta_learner_ultimate.pkl')

        # Save config
        config = {
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'meta_learner_type': self.meta_learner_type,
            'all_model_names': self.all_model_names
        }

        with open(output_path / 'ultimate_ensemble_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nUltimate ensemble saved to {output_dir}")


def main():
    """Main training script"""
    import argparse

    parser = argparse.ArgumentParser(description='Train ultimate meta-ensemble')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--train-file', type=str, default='train.csv')
    parser.add_argument('--target-col', type=str, default='target')
    parser.add_argument('--output-dir', type=str, default='ultimate_ensemble_models')
    parser.add_argument('--seq-len', type=int, default=96)
    parser.add_argument('--pred-len', type=int, default=24)
    parser.add_argument('--transformer-epochs', type=int, default=50)
    parser.add_argument('--meta-learner', type=str, default='xgboost', choices=['xgboost', 'ridge'])
    parser.add_argument('--no-trees', action='store_true', help='Exclude tree models')
    parser.add_argument('--no-transformers', action='store_true', help='Exclude transformers')
    parser.add_argument('--no-foundation', action='store_true', help='Exclude foundation models')
    parser.add_argument('--no-hybrid', action='store_true', help='Exclude hybrid model')
    parser.add_argument('--val-split', type=float, default=0.3)

    args = parser.parse_args()

    print("="*70)
    print("ULTIMATE META-ENSEMBLE TRAINING")
    print("All Models + Meta-Learning")
    print("="*70)

    # Load data
    preprocessor = TimeSeriesPreprocessor(scaler_type='standard')
    print(f"\nLoading data from {args.data_dir}/{args.train_file}...")
    train_df, _ = preprocessor.load_data(str(Path(args.data_dir) / args.train_file))

    # Create ensemble
    ensemble = UltimateMetaEnsemble(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        transformer_epochs=args.transformer_epochs,
        meta_learner_type=args.meta_learner,
        include_tree_models=not args.no_trees,
        include_transformers=not args.no_transformers,
        include_foundation=not args.no_foundation,
        include_hybrid=not args.no_hybrid
    )

    # Train
    metrics = ensemble.train(
        train_df,
        target_col=args.target_col,
        val_split=args.val_split,
        preprocessor=preprocessor if not args.no_trees else None
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
    meta_improvement = metrics.get('meta_learner', {}).get('improvement_pct', 0)
    print(f"Final improvement: {meta_improvement:.2f}%")


if __name__ == '__main__':
    main()
