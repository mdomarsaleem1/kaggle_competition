"""
Make predictions using trained nested ensemble
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import argparse

# Import the NestedEnsemble class
from nested_ensemble_predict import NestedEnsemble
from utils.data_utils import TimeSeriesPreprocessor


def make_predictions(model_dir: str, test_file: str, output_file: str,
                    target_col: str = 'target', date_col: str = 'date',
                    id_col: str = 'id'):
    """
    Make predictions using trained nested ensemble

    Args:
        model_dir: Directory containing trained models
        test_file: Path to test data CSV
        output_file: Path to save predictions
        target_col: Name of target column
        date_col: Name of date column
        id_col: Name of ID column
    """
    print("="*70)
    print("NESTED ENSEMBLE PREDICTION")
    print("="*70)

    # Load ensemble
    print(f"\nLoading ensemble from {model_dir}...")
    ensemble = NestedEnsemble()
    ensemble.load(model_dir)

    # Load preprocessor
    print("Loading preprocessor...")
    preprocessor = joblib.load(Path(model_dir) / 'preprocessor.pkl')

    # Load test data
    print(f"\nLoading test data from {test_file}...")
    test_df = pd.read_csv(test_file)
    print(f"Test data shape: {test_df.shape}")

    # Check if we need to create features or if they already exist
    if target_col in test_df.columns:
        # Test data has target - create features
        print("Creating features from test data...")
        df_features = preprocessor.create_all_features(test_df, target_col)

        # Prepare features
        feature_cols = [col for col in df_features.columns
                       if col not in [date_col, target_col]]
        X_test = df_features[feature_cols].values

        # Check if we have ID column
        if id_col in df_features.columns:
            ids = df_features[id_col].values
        else:
            ids = np.arange(len(df_features))

    else:
        # Test data doesn't have target - assume features are already prepared
        print("Using existing features from test data...")
        feature_cols = [col for col in test_df.columns
                       if col not in [date_col, id_col]]
        X_test = test_df[feature_cols].values

        if id_col in test_df.columns:
            ids = test_df[id_col].values
        else:
            ids = np.arange(len(test_df))

    # Scale features
    print("Scaling features...")
    X_test = preprocessor.transform(X_test)

    print(f"Test features shape: {X_test.shape}")

    # Make predictions
    print("\nGenerating predictions...")
    predictions = ensemble.predict(X_test)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")

    # Create submission DataFrame
    submission = pd.DataFrame({
        id_col: ids,
        'prediction': predictions
    })

    # Save predictions
    submission.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")

    # Show preview
    print("\nPrediction preview:")
    print(submission.head(10))

    # Show statistics
    print("\nPrediction Statistics:")
    print(submission['prediction'].describe())

    return submission


def main():
    parser = argparse.ArgumentParser(description='Make predictions with nested ensemble')

    parser.add_argument('--model-dir', type=str, required=True,
                       help='Directory containing trained ensemble models')
    parser.add_argument('--test-file', type=str, required=True,
                       help='Path to test data CSV')
    parser.add_argument('--output-file', type=str, default='nested_ensemble_submission.csv',
                       help='Path to save predictions')
    parser.add_argument('--target-col', type=str, default='target',
                       help='Target column name (if present in test data)')
    parser.add_argument('--date-col', type=str, default='date',
                       help='Date column name')
    parser.add_argument('--id-col', type=str, default='id',
                       help='ID column name')

    args = parser.parse_args()

    make_predictions(
        model_dir=args.model_dir,
        test_file=args.test_file,
        output_file=args.output_file,
        target_col=args.target_col,
        date_col=args.date_col,
        id_col=args.id_col
    )


if __name__ == '__main__':
    main()
