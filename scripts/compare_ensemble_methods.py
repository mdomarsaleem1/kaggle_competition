"""
Compare Simple Ensemble vs Nested Ensemble (Stacking)

This script trains both ensemble methods and compares their performance
to demonstrate the benefits of meta-learning.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils.data_utils import TimeSeriesPreprocessor
from models import XGBoostTimeSeriesModel, LightGBMTimeSeriesModel, CatBoostTimeSeriesModel
from nested_ensemble_predict import NestedEnsemble


def generate_synthetic_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic time series data"""
    np.random.seed(seed)

    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    # Create synthetic time series with multiple patterns
    t = np.arange(n_samples)

    # Trend
    trend = 100 + 0.05 * t

    # Seasonality
    yearly_season = 20 * np.sin(2 * np.pi * t / 365.25)
    weekly_season = 5 * np.sin(2 * np.pi * t / 7)

    # Volatility regimes (high/low volatility)
    volatility = np.where(t % 500 < 250, 10, 3)  # Changes every 250 days

    # Noise
    noise = np.random.normal(0, volatility)

    # Target
    target = trend + yearly_season + weekly_season + noise

    df = pd.DataFrame({
        'date': dates,
        'target': target,
        'volatility_regime': (t % 500 < 250).astype(int)  # Hidden feature
    })

    return df


def train_simple_ensemble(X_train, y_train, X_val, y_val):
    """Train base models and compute simple average"""
    print("\n" + "="*70)
    print("SIMPLE ENSEMBLE (Average of Base Models)")
    print("="*70)

    models = {}
    predictions = {}

    # Train XGBoost
    print("\nTraining XGBoost...")
    xgb_model = XGBoostTimeSeriesModel()
    xgb_model.train(X_train, y_train, X_val, y_val, verbose=False)
    models['xgboost'] = xgb_model
    predictions['xgboost'] = xgb_model.predict(X_val)

    # Train LightGBM
    print("Training LightGBM...")
    lgb_model = LightGBMTimeSeriesModel()
    lgb_model.train(X_train, y_train, X_val, y_val, verbose=False)
    models['lightgbm'] = lgb_model
    predictions['lightgbm'] = lgb_model.predict(X_val)

    # Train CatBoost
    print("Training CatBoost...")
    cat_model = CatBoostTimeSeriesModel()
    cat_model.train(X_train, y_train, X_val, y_val, verbose=False)
    models['catboost'] = cat_model
    predictions['catboost'] = cat_model.predict(X_val)

    # Compute simple average
    all_preds = np.column_stack([predictions[name] for name in ['xgboost', 'lightgbm', 'catboost']])
    simple_avg = all_preds.mean(axis=1)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_val, simple_avg))
    mae = mean_absolute_error(y_val, simple_avg)

    print(f"\nSimple Ensemble Results:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")

    return models, predictions, simple_avg, rmse, mae


def train_nested_ensemble(X_train, y_train, X_val, y_val):
    """Train nested ensemble with meta-learning"""
    print("\n" + "="*70)
    print("NESTED ENSEMBLE (Meta-Learning)")
    print("="*70)

    # Combine train and val for nested ensemble training
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    # Train nested ensemble
    ensemble = NestedEnsemble(
        meta_learner_type='xgboost',
        use_original_features=True
    )

    metrics = ensemble.train_with_holdout(X_full, y_full, val_split=0.3)

    return ensemble, metrics


def analyze_by_volatility(y_true, predictions_dict, volatility_regime):
    """Analyze performance by volatility regime"""
    print("\n" + "="*70)
    print("PERFORMANCE BY VOLATILITY REGIME")
    print("="*70)

    for regime in [0, 1]:
        mask = volatility_regime == regime
        regime_name = "Low Volatility" if regime == 0 else "High Volatility"

        print(f"\n{regime_name} (n={mask.sum()}):")
        print("-" * 50)

        for name, preds in predictions_dict.items():
            rmse = np.sqrt(mean_squared_error(y_true[mask], preds[mask]))
            print(f"  {name:20s}: {rmse:.6f}")


def create_visualization(y_true, simple_pred, nested_pred, output_file='comparison.png'):
    """Create comparison visualization"""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Predictions vs Actual
        axes[0, 0].plot(y_true[:200], label='Actual', linewidth=2, alpha=0.8)
        axes[0, 0].plot(simple_pred[:200], label='Simple Ensemble', alpha=0.7)
        axes[0, 0].plot(nested_pred[:200], label='Nested Ensemble', alpha=0.7)
        axes[0, 0].set_title('Predictions vs Actual (First 200 samples)', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Residuals
        simple_residuals = y_true - simple_pred
        nested_residuals = y_true - nested_pred

        axes[0, 1].scatter(simple_pred, simple_residuals, alpha=0.5, label='Simple', s=10)
        axes[0, 1].scatter(nested_pred, nested_residuals, alpha=0.5, label='Nested', s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Residual Plot', fontsize=12)
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residual')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Error distribution
        axes[1, 0].hist(simple_residuals, bins=50, alpha=0.6, label='Simple', edgecolor='black')
        axes[1, 0].hist(nested_residuals, bins=50, alpha=0.6, label='Nested', edgecolor='black')
        axes[1, 0].set_title('Error Distribution', fontsize=12)
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Cumulative error
        simple_cum_error = np.cumsum(np.abs(simple_residuals))
        nested_cum_error = np.cumsum(np.abs(nested_residuals))

        axes[1, 1].plot(simple_cum_error, label='Simple Ensemble', linewidth=2)
        axes[1, 1].plot(nested_cum_error, label='Nested Ensemble', linewidth=2)
        axes[1, 1].set_title('Cumulative Absolute Error', fontsize=12)
        axes[1, 1].set_xlabel('Sample')
        axes[1, 1].set_ylabel('Cumulative |Error|')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {output_file}")

        plt.close()

    except ImportError:
        print("\nMatplotlib not available. Skipping visualization.")


def main():
    """Main comparison"""
    print("="*70)
    print("ENSEMBLE METHOD COMPARISON")
    print("Simple Ensemble vs Nested Ensemble (Stacking)")
    print("="*70)

    # Generate data
    print("\nGenerating synthetic time series data...")
    df = generate_synthetic_data(n_samples=2000)

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Prepare features
    print("\nCreating features...")
    preprocessor = TimeSeriesPreprocessor()
    df_features = preprocessor.create_all_features(df, 'target')

    # Keep volatility regime for analysis
    volatility_regime = df_features['volatility_regime'].values if 'volatility_regime' in df_features.columns else None

    # Prepare data
    feature_cols = [col for col in df_features.columns if col not in ['date', 'target', 'volatility_regime']]
    X = df_features[feature_cols].values
    y = df_features['target'].values

    # Scale
    X = preprocessor.fit_transform(X)

    # Split
    split_idx = int(len(X) * 0.7)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    vol_val = volatility_regime[split_idx:] if volatility_regime is not None else None

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Train simple ensemble
    start_time = time.time()
    simple_models, simple_base_preds, simple_pred, simple_rmse, simple_mae = \
        train_simple_ensemble(X_train, y_train, X_val, y_val)
    simple_time = time.time() - start_time

    # Train nested ensemble
    start_time = time.time()
    nested_ensemble, nested_metrics = train_nested_ensemble(X_train, y_train, X_val, y_val)

    # Get nested ensemble predictions
    nested_pred = nested_ensemble.predict(X_val)
    nested_rmse = np.sqrt(mean_squared_error(y_val, nested_pred))
    nested_mae = mean_absolute_error(y_val, nested_pred)
    nested_time = time.time() - start_time

    # Print comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)

    comparison_df = pd.DataFrame({
        'Method': ['Simple Ensemble', 'Nested Ensemble'],
        'RMSE': [simple_rmse, nested_rmse],
        'MAE': [simple_mae, nested_mae],
        'Training Time (s)': [simple_time, nested_time]
    })

    print("\n" + comparison_df.to_string(index=False))

    # Calculate improvement
    rmse_improvement = (simple_rmse - nested_rmse) / simple_rmse * 100
    mae_improvement = (simple_mae - nested_mae) / simple_mae * 100

    print(f"\n{'='*70}")
    print("IMPROVEMENT")
    print(f"{'='*70}")
    print(f"RMSE Improvement: {rmse_improvement:+.2f}%")
    print(f"MAE Improvement:  {mae_improvement:+.2f}%")

    if rmse_improvement > 0:
        print(f"\n✅ Nested ensemble is BETTER by {rmse_improvement:.2f}%")
    else:
        print(f"\n⚠️  Simple ensemble is better by {-rmse_improvement:.2f}%")

    # Analyze by volatility if available
    if vol_val is not None:
        all_predictions = {
            'XGBoost': simple_base_preds['xgboost'],
            'LightGBM': simple_base_preds['lightgbm'],
            'CatBoost': simple_base_preds['catboost'],
            'Simple Ensemble': simple_pred,
            'Nested Ensemble': nested_pred
        }
        analyze_by_volatility(y_val, all_predictions, vol_val)

    # Create visualization
    create_visualization(y_val, simple_pred, nested_pred)

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The nested ensemble uses meta-learning to dynamically combine base models.
Key advantages:
1. Context-aware weighting (adapts to different scenarios)
2. Learns when each model excels
3. Typically achieves 1-5% improvement over simple averaging

Use nested ensemble when:
- You have sufficient data (>5000 samples)
- Different models excel in different scenarios
- You want to maximize performance
    """)


if __name__ == '__main__':
    main()
