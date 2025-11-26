"""End-to-end submission workflow mirroring the public Hull starter notebook.

The script loads train/test data, optionally evaluates a LightGBM baseline with
chronological cross-validation, and writes a submission file ready for Kaggle.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.lightgbm_model import LightGBMTimeSeriesModel
from utils import create_chronological_folds


def _prepare_features(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    target_col: str,
    date_col: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Split features/targets and align preprocessing across datasets."""
    excluded_cols = {target_col}
    if date_col:
        excluded_cols.add(date_col)

    feature_cols = [col for col in train_df.columns if col not in excluded_cols]
    train_features = train_df[feature_cols].copy()
    eval_features = eval_df[feature_cols].copy()

    numeric_cols = train_features.select_dtypes(include=[np.number]).columns
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    for col in numeric_cols:
        median = train_features[col].median()
        train_features[col] = train_features[col].fillna(median)
        eval_features[col] = eval_features[col].fillna(median)

    for col in categorical_cols:
        mode = train_features[col].mode()
        fill_value = mode.iloc[0] if not mode.empty else ""
        train_features[col] = train_features[col].fillna(fill_value)
        eval_features[col] = eval_features[col].fillna(fill_value)

    return train_features, eval_features, feature_cols


def _parse_dates(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    if date_col and date_col in df.columns:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
    return df


def run_cross_validation(
    df: pd.DataFrame,
    target_col: str,
    date_col: Optional[str],
    n_splits: int,
) -> pd.DataFrame:
    """Evaluate the baseline model with chronological cross-validation."""
    folds = create_chronological_folds(df, n_splits=n_splits, date_col=date_col)
    results = []

    for fold in folds:
        train_df = fold.train
        val_df = fold.test

        X_train, X_val, _ = _prepare_features(train_df, val_df, target_col, date_col)
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values

        model = LightGBMTimeSeriesModel()
        model.train(X_train.values, y_train, verbose=False)
        preds = model.predict(X_val.values)

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)

        results.append(
            {
                "fold": fold.fold_id,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "train_range": fold.train_range,
                "val_range": fold.test_range,
                "rmse": rmse,
                "mae": mae,
            }
        )

    return pd.DataFrame(results)


def train_full_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    date_col: Optional[str],
) -> Tuple[pd.Series, pd.DataFrame]:
    """Train the LightGBM baseline on full data and predict the test set."""
    X_train, X_test, _ = _prepare_features(train_df, test_df, target_col, date_col)
    y_train = train_df[target_col].values

    model = LightGBMTimeSeriesModel()
    model.train(X_train.values, y_train, verbose=False)
    predictions = pd.Series(model.predict(X_test.values))

    return predictions, X_test


def build_submission(
    test_df: pd.DataFrame,
    predictions: pd.Series,
    output_file: Path,
    id_col: Optional[str],
    target_col: str,
) -> pd.DataFrame:
    """Create and save the submission dataframe."""
    if id_col and id_col in test_df.columns:
        identifiers = test_df[id_col]
    else:
        identifiers = pd.RangeIndex(start=0, stop=len(test_df), step=1)

    submission = pd.DataFrame({
        id_col if id_col else "row_id": identifiers,
        target_col: predictions,
    })

    submission.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline submission script for Hull Tactical competition")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing train and test CSVs")
    parser.add_argument("--train-file", type=str, default="train.csv", help="Training filename")
    parser.add_argument("--test-file", type=str, default="test.csv", help="Test filename")
    parser.add_argument("--target-col", type=str, default="target", help="Target column name")
    parser.add_argument("--date-col", type=str, default="date", help="Optional date column name")
    parser.add_argument("--id-col", type=str, default="row_id", help="Identifier column for submission")
    parser.add_argument("--output-file", type=str, default="submission.csv", help="Output submission filename")
    parser.add_argument("--run-cv", action="store_true", help="Run 10-fold chronological CV before training on full data")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of chronological CV splits to generate")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / args.train_file
    test_path = data_dir / args.test_file

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = _parse_dates(train_df, args.date_col)
    test_df = _parse_dates(test_df, args.date_col)

    if args.run_cv:
        cv_results = run_cross_validation(train_df, args.target_col, args.date_col, args.n_splits)
        print("\nChronological CV summary:")
        print(cv_results)
        print(f"\nMean RMSE: {cv_results['rmse'].mean():.6f}, Mean MAE: {cv_results['mae'].mean():.6f}")

    predictions, test_features = train_full_model(train_df, test_df, args.target_col, args.date_col)
    build_submission(test_df.loc[test_features.index], predictions, Path(args.output_file), args.id_col, args.target_col)


if __name__ == "__main__":
    main()
