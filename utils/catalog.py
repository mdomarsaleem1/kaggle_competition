"""Lightweight data catalog inspired by Kedro's dataset registry."""
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def _resolve_path(root: Path, relative_path: str) -> Path:
    resolved = root / relative_path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def default_catalog(root: Optional[Path] = None) -> Dict[str, Path]:
    """Return the default project data catalog paths.

    The layout follows a cookie-cutter/Kedro-inspired convention:
    - ``01_raw``: immutable source data
    - ``02_interim``: feature-ready intermediate artifacts
    - ``03_processed``: model-ready datasets
    - ``04_models``: persisted model artifacts and submissions
    """
    base = root or DEFAULT_DATA_ROOT
    return {
        "train_raw": _resolve_path(base, "01_raw/train.csv"),
        "test_raw": _resolve_path(base, "01_raw/test.csv"),
        "clean_train": _resolve_path(base, "02_interim/clean_train.parquet"),
        "folds_dir": _resolve_path(base, "02_interim/folds"),
        "model_matrix": _resolve_path(base, "03_processed/model_matrix.parquet"),
        "oof_predictions": _resolve_path(base, "03_processed/oof_predictions.parquet"),
        "submission": _resolve_path(base, "04_models/submission.csv"),
    }


def load_table(path: Path, **read_kwargs) -> pd.DataFrame:
    """Load a dataframe from disk with sensible defaults for CSV/Parquet."""
    if path.suffix == ".csv":
        return pd.read_csv(path, **read_kwargs)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, **read_kwargs)
    raise ValueError(f"Unsupported extension for {path}")


def save_table(df: pd.DataFrame, path: Path, **write_kwargs) -> None:
    """Persist a dataframe using the extension to choose the writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False, **write_kwargs)
    elif path.suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False, **write_kwargs)
    else:
        raise ValueError(f"Unsupported extension for {path}")
