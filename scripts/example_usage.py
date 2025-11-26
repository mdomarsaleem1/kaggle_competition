"""
Example usage of all time series models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import all models
from models import (
    XGBoostTimeSeriesModel,
    LightGBMTimeSeriesModel,
    CatBoostTimeSeriesModel,
    ProphetTimeSeriesModel,
    ChronosTimeSeriesModel
)
from utils.data_utils import TimeSeriesPreprocessor


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample time series data for demonstration"""
    np.random.seed(42)

    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    # Create synthetic time series with trend and seasonality
    trend = np.linspace(100, 200, n_samples)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
    noise = np.random.normal(0, 5, n_samples)

    target = trend + seasonality + noise

    df = pd.DataFrame({
        'date': dates,
        'target': target
    })

    return df


def example_xgboost():
    """Example: XGBoost Time Series Model"""
    print("\n" + "="*70)
    print("EXAMPLE: XGBoost Time Series Model")
    print("="*70)

    # Generate data
    df = generate_sample_data(1000)

    # Prepare data
    preprocessor = TimeSeriesPreprocessor()
    df_features = preprocessor.create_all_features(df, 'target')

    # Split
    train_size = int(len(df_features) * 0.8)
    train_df = df_features.iloc[:train_size]
    test_df = df_features.iloc[train_size:]

    feature_cols = [col for col in df_features.columns if col not in ['date', 'target']]
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values

    # Train model
    model = XGBoostTimeSeriesModel()
    metrics = model.train(X_train, y_train, X_test, y_test)

    print(f"\nResults:")
    print(f"  Val RMSE: {metrics['val_rmse']:.4f}")
    print(f"  Val MAE: {metrics['val_mae']:.4f}")

    # Make predictions
    predictions = model.predict(X_test)
    print(f"\nSample predictions: {predictions[:5]}")

    return model


def example_lightgbm():
    """Example: LightGBM Time Series Model"""
    print("\n" + "="*70)
    print("EXAMPLE: LightGBM Time Series Model")
    print("="*70)

    # Generate data
    df = generate_sample_data(1000)

    # Prepare data
    preprocessor = TimeSeriesPreprocessor()
    df_features = preprocessor.create_all_features(df, 'target')

    # Split
    train_size = int(len(df_features) * 0.8)
    train_df = df_features.iloc[:train_size]
    test_df = df_features.iloc[train_size:]

    feature_cols = [col for col in df_features.columns if col not in ['date', 'target']]
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values

    # Train model
    model = LightGBMTimeSeriesModel()
    metrics = model.train(X_train, y_train, X_test, y_test, verbose=False)

    print(f"\nResults:")
    print(f"  Val RMSE: {metrics['val_rmse']:.4f}")
    print(f"  Val MAE: {metrics['val_mae']:.4f}")
    print(f"  Best Iteration: {metrics['best_iteration']}")

    return model


def example_catboost():
    """Example: CatBoost Time Series Model"""
    print("\n" + "="*70)
    print("EXAMPLE: CatBoost Time Series Model")
    print("="*70)

    # Generate data
    df = generate_sample_data(1000)

    # Prepare data
    preprocessor = TimeSeriesPreprocessor()
    df_features = preprocessor.create_all_features(df, 'target')

    # Split
    train_size = int(len(df_features) * 0.8)
    train_df = df_features.iloc[:train_size]
    test_df = df_features.iloc[train_size:]

    feature_cols = [col for col in df_features.columns if col not in ['date', 'target']]
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values

    # Train model
    model = CatBoostTimeSeriesModel()
    metrics = model.train(X_train, y_train, X_test, y_test, verbose=False)

    print(f"\nResults:")
    print(f"  Val RMSE: {metrics['val_rmse']:.4f}")
    print(f"  Val MAE: {metrics['val_mae']:.4f}")

    return model


def example_prophet():
    """Example: Facebook Prophet Model"""
    print("\n" + "="*70)
    print("EXAMPLE: Facebook Prophet Model")
    print("="*70)

    # Generate data
    df = generate_sample_data(1000)

    # Split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # Prepare data for Prophet
    model = ProphetTimeSeriesModel()
    prophet_train = model.prepare_data(train_df, 'date', 'target')
    prophet_test = model.prepare_data(test_df, 'date', 'target')

    # Train model
    print("Training Prophet (this may take a moment)...")
    metrics = model.train(prophet_train, verbose=False)

    # Predict
    forecast = model.predict(prophet_test[['ds']])

    # Evaluate
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    val_rmse = np.sqrt(mean_squared_error(prophet_test['y'], forecast['yhat']))
    val_mae = mean_absolute_error(prophet_test['y'], forecast['yhat'])

    print(f"\nResults:")
    print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"  Val RMSE: {val_rmse:.4f}")
    print(f"  Val MAE: {val_mae:.4f}")

    return model


def example_chronos():
    """Example: Chronos-2 Foundation Model"""
    print("\n" + "="*70)
    print("EXAMPLE: Chronos-2 Foundation Model")
    print("="*70)

    try:
        # Generate data
        df = generate_sample_data(500)

        # Split
        train_size = int(len(df) * 0.8)
        context = df['target'].values[:train_size]
        actual = df['target'].values[train_size:]

        # Load model
        print("Loading Chronos model (this may take a moment)...")
        model = ChronosTimeSeriesModel(model_size='tiny')  # Use 'tiny' for faster demo
        model.load_model()

        # Predict
        context_window = context[-128:]  # Use last 128 points as context
        prediction_length = 30

        print(f"Making predictions for {prediction_length} steps...")
        forecasts = model.predict(
            context=context_window,
            prediction_length=prediction_length,
            num_samples=20
        )

        # Get median forecast
        median_forecast = np.median(forecasts, axis=0)

        # Evaluate
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        actual_trimmed = actual[:prediction_length]
        rmse = np.sqrt(mean_squared_error(actual_trimmed, median_forecast))
        mae = mean_absolute_error(actual_trimmed, median_forecast)

        print(f"\nResults:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  Forecast samples shape: {forecasts.shape}")

        return model

    except Exception as e:
        print(f"\nError with Chronos: {e}")
        print("Chronos requires additional dependencies. Install with:")
        print("pip install git+https://github.com/amazon-science/chronos-forecasting.git")
        return None


def run_all_examples():
    """Run all examples"""
    print("\n" + "="*70)
    print("TIME SERIES FORECASTING MODELS - EXAMPLE USAGE")
    print("="*70)

    # Run examples
    example_xgboost()
    example_lightgbm()
    example_catboost()
    example_prophet()
    example_chronos()

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == '__main__':
    run_all_examples()
