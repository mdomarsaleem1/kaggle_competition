"""
Data loading and preprocessing utilities for Hull Tactical Market Prediction
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .catalog import DEFAULT_DATA_ROOT


class TimeSeriesPreprocessor:
    """Preprocessor for time series data"""

    def __init__(self, scaler_type='standard'):
        """
        Initialize preprocessor

        Args:
            scaler_type: Type of scaler ('standard', 'minmax', or None)
        """
        self.scaler_type = scaler_type
        self.scaler = None
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()

    def load_data(
        self,
        train_path: str,
        test_path: Optional[str] = None,
        date_is_numeric: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load training and test data

        Args:
            train_path: Path to training CSV
            test_path: Path to test CSV (optional)
            date_is_numeric: If True, date column is numeric ID; if False, parse as datetime

        Returns:
            Tuple of (train_df, test_df)
        """
        train_file = Path(train_path)
        test_file = Path(test_path) if test_path else None

        if not train_file.is_absolute():
            train_file = DEFAULT_DATA_ROOT / train_file
        if test_file and not test_file.is_absolute():
            test_file = DEFAULT_DATA_ROOT / test_file

        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file) if test_file else None

        # Handle date column
        if 'date' in train_df.columns:
            if date_is_numeric:
                # Date is a numeric ID - ensure it's integer type
                train_df['date'] = train_df['date'].astype(int)
                train_df = train_df.sort_values('date').reset_index(drop=True)
            else:
                # Date is actual datetime - parse it
                train_df['date'] = pd.to_datetime(train_df['date'])
                train_df = train_df.sort_values('date').reset_index(drop=True)

        if test_df is not None and 'date' in test_df.columns:
            if date_is_numeric:
                test_df['date'] = test_df['date'].astype(int)
                test_df = test_df.sort_values('date').reset_index(drop=True)
            else:
                test_df['date'] = pd.to_datetime(test_df['date'])
                test_df = test_df.sort_values('date').reset_index(drop=True)

        return train_df, test_df

    def create_lag_features(self, df: pd.DataFrame, target_col: str, lags: List[int]) -> pd.DataFrame:
        """
        Create lag features for time series

        Args:
            df: Input dataframe
            target_col: Name of target column
            lags: List of lag values to create

        Returns:
            DataFrame with lag features
        """
        df_copy = df.copy()
        for lag in lags:
            df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)
        return df_copy

    def create_rolling_features(self, df: pd.DataFrame, target_col: str, windows: List[int]) -> pd.DataFrame:
        """
        Create rolling window features

        Args:
            df: Input dataframe
            target_col: Name of target column
            windows: List of window sizes

        Returns:
            DataFrame with rolling features
        """
        df_copy = df.copy()
        for window in windows:
            df_copy[f'{target_col}_rolling_mean_{window}'] = df_copy[target_col].rolling(window=window).mean()
            df_copy[f'{target_col}_rolling_std_{window}'] = df_copy[target_col].rolling(window=window).std()
            df_copy[f'{target_col}_rolling_min_{window}'] = df_copy[target_col].rolling(window=window).min()
            df_copy[f'{target_col}_rolling_max_{window}'] = df_copy[target_col].rolling(window=window).max()
        return df_copy

    def create_time_features(self, df: pd.DataFrame, date_col: str = 'date',
                            date_is_numeric: bool = True) -> pd.DataFrame:
        """
        Create time-based features

        Args:
            df: Input dataframe
            date_col: Name of date column
            date_is_numeric: If True, date is numeric ID; if False, it's datetime

        Returns:
            DataFrame with time features
        """
        df_copy = df.copy()
        if date_col in df_copy.columns:
            if date_is_numeric:
                # Date is a numeric ID - create cyclic and positional features
                # Normalize date to [0, 1] range for the dataset
                date_min = df_copy[date_col].min()
                date_max = df_copy[date_col].max()
                date_range = date_max - date_min

                if date_range > 0:
                    # Normalized date position
                    df_copy['date_normalized'] = (df_copy[date_col] - date_min) / date_range

                    # Cyclic features assuming different periodicities
                    # Weekly pattern (assuming data has weekly periodicity)
                    df_copy['day_of_week_sin'] = np.sin(2 * np.pi * df_copy[date_col] / 7)
                    df_copy['day_of_week_cos'] = np.cos(2 * np.pi * df_copy[date_col] / 7)

                    # Monthly pattern (assuming ~30 day months)
                    df_copy['day_of_month_sin'] = np.sin(2 * np.pi * df_copy[date_col] / 30)
                    df_copy['day_of_month_cos'] = np.cos(2 * np.pi * df_copy[date_col] / 30)

                    # Yearly pattern (assuming ~365 days)
                    df_copy['day_of_year_sin'] = np.sin(2 * np.pi * df_copy[date_col] / 365)
                    df_copy['day_of_year_cos'] = np.cos(2 * np.pi * df_copy[date_col] / 365)

                    # Quarter of year (assuming ~90 day quarters)
                    df_copy['quarter_sin'] = np.sin(2 * np.pi * df_copy[date_col] / 90)
                    df_copy['quarter_cos'] = np.cos(2 * np.pi * df_copy[date_col] / 90)
                else:
                    # Single date value - set all to 0
                    df_copy['date_normalized'] = 0
                    for col in ['day_of_week', 'day_of_month', 'day_of_year', 'quarter']:
                        df_copy[f'{col}_sin'] = 0
                        df_copy[f'{col}_cos'] = 0
            else:
                # Date is datetime - extract standard datetime features
                df_copy['year'] = df_copy[date_col].dt.year
                df_copy['month'] = df_copy[date_col].dt.month
                df_copy['day'] = df_copy[date_col].dt.day
                df_copy['dayofweek'] = df_copy[date_col].dt.dayofweek
                df_copy['quarter'] = df_copy[date_col].dt.quarter
                df_copy['dayofyear'] = df_copy[date_col].dt.dayofyear
                df_copy['weekofyear'] = df_copy[date_col].dt.isocalendar().week
        return df_copy

    def create_all_features(self, df: pd.DataFrame, target_col: str,
                           lags: List[int] = [1, 2, 3, 5, 7, 14, 21, 30],
                           windows: List[int] = [7, 14, 30, 60],
                           date_is_numeric: bool = True) -> pd.DataFrame:
        """
        Create all features for time series modeling

        Args:
            df: Input dataframe
            target_col: Name of target column
            lags: List of lag values
            windows: List of rolling window sizes
            date_is_numeric: If True, date column is numeric ID; if False, it's datetime

        Returns:
            DataFrame with all features
        """
        df_features = df.copy()

        # Time features
        df_features = self.create_time_features(df_features, date_is_numeric=date_is_numeric)

        # Lag features
        df_features = self.create_lag_features(df_features, target_col, lags)

        # Rolling features
        df_features = self.create_rolling_features(df_features, target_col, windows)

        # Drop rows with NaN values from feature engineering
        df_features = df_features.dropna().reset_index(drop=True)

        return df_features

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform features"""
        if self.scaler is not None:
            return self.scaler.fit_transform(X)
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler"""
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform features"""
        if self.scaler is not None:
            return self.scaler.inverse_transform(X)
        return X


def create_sequences(data: np.ndarray, seq_length: int, forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series forecasting

    Args:
        data: Input time series data
        seq_length: Length of input sequences
        forecast_horizon: Number of steps to forecast

    Returns:
        Tuple of (X, y) sequences
    """
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + forecast_horizon])
    return np.array(X), np.array(y)


def split_time_series(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and validation sets

    Args:
        df: Input dataframe
        test_size: Proportion of data to use for validation

    Returns:
        Tuple of (train_df, val_df)
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    return train_df, val_df
