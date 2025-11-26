"""Chronological cross-validation helpers for time series models."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .metrics import averaged_metric, hull_sharpe_ratio

@dataclass
class TimeSeriesFold:
    """Container for a single time series cross-validation fold."""

    fold_id: int
    train: pd.DataFrame
    test: pd.DataFrame
    train_range: str
    test_range: str


def _sort_by_date(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    if date_col and date_col in df.columns:
        sorted_df = df.copy()
        sorted_df[date_col] = pd.to_datetime(sorted_df[date_col])
        return sorted_df.sort_values(date_col).reset_index(drop=True)
    return df.reset_index(drop=True)


def create_chronological_folds(
    df: pd.DataFrame,
    n_splits: int = 10,
    date_col: Optional[str] = "date",
    test_size: Optional[int] = None,
) -> List[TimeSeriesFold]:
    """Generate chronological cross-validation folds.

    Args:
        df: Input dataframe containing the full training data.
        n_splits: Number of folds to create.
        date_col: Name of the date column to use for sorting. If ``None`` the
            existing order of ``df`` is preserved.
        test_size: Optional fixed test size passed to ``TimeSeriesSplit``. If
            ``None`` the test window is inferred by scikit-learn.

    Returns:
        A list of :class:`TimeSeriesFold` instances with chronologically ordered
        train and test splits.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for cross-validation")

    sorted_df = _sort_by_date(df, date_col)
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    folds: List[TimeSeriesFold] = []
    for fold_id, (train_idx, test_idx) in enumerate(tscv.split(sorted_df), start=1):
        train_df = sorted_df.iloc[train_idx].copy()
        test_df = sorted_df.iloc[test_idx].copy()

        train_range = "N/A"
        test_range = "N/A"
        if date_col and date_col in sorted_df.columns:
            train_range = f"{train_df[date_col].min()} -> {train_df[date_col].max()}"
            test_range = f"{test_df[date_col].min()} -> {test_df[date_col].max()}"

        folds.append(
            TimeSeriesFold(
                fold_id=fold_id,
                train=train_df,
                test=test_df,
                train_range=train_range,
                test_range=test_range,
            )
        )

    return folds


def save_folds_to_disk(folds: List[TimeSeriesFold], output_dir: str) -> None:
    """Persist cross-validation folds to disk.

    Each fold is saved as ``train_fold_{i}.csv`` and ``test_fold_{i}.csv`` where
    ``i`` starts at 1.

    Args:
        folds: The folds returned by :func:`create_chronological_folds`.
        output_dir: Directory where the CSVs should be written.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        train_file = output_path / f"train_fold_{fold.fold_id}.csv"
        test_file = output_path / f"test_fold_{fold.fold_id}.csv"
        fold.train.to_csv(train_file, index=False)
        fold.test.to_csv(test_file, index=False)

        print(
            f"Saved fold {fold.fold_id}: train -> {train_file.name} "
            f"({fold.train_range}), test -> {test_file.name} ({fold.test_range})"
        )


def evaluate_time_series_model(
    X: pd.DataFrame,
    y: pd.Series,
    estimator_factory: Callable[[], object],
    n_splits: int = 10,
    date_col: Optional[str] = "date",
    metric: Callable[[Sequence[float], Sequence[float]], float] = hull_sharpe_ratio,
    test_size: Optional[int] = None,
) -> Tuple[float, List[float]]:
    """Run centralized chronological cross-validation for a model.

    Args:
        X: Feature matrix sorted by ``date_col``.
        y: Target vector aligned with ``X``.
        estimator_factory: Zero-argument callable returning an unfitted estimator.
        n_splits: Number of chronological folds (defaults to 10).
        date_col: Optional date column name used to sort the data.
        metric: Scoring function where higher is better.
        test_size: Optional explicit test window size for ``TimeSeriesSplit``.

    Returns:
        Tuple of the averaged metric across folds and the list of per-fold scores.
    """

    df = X.copy()
    df["__target__"] = y.values
    sorted_df = _sort_by_date(df, date_col)

    features = sorted_df.drop(columns=["__target__"])
    target = sorted_df["__target__"]

    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    fold_scores: List[float] = []

    for train_idx, test_idx in splitter.split(features):
        estimator = estimator_factory()
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

        estimator.fit(X_train, y_train)
        preds = estimator.predict(X_test)
        fold_scores.append(metric(y_test, preds))

    return averaged_metric(fold_scores), fold_scores
