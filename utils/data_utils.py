"""
Data loading and preprocessing utilities for Hull Tactical Market Prediction
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, List


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

    def load_data(self, train_path: str, test_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load training and test data

        Args:
            train_path: Path to training CSV
            test_path: Path to test CSV (optional)

        Returns:
            Tuple of (train_df, test_df)
        """
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if test_path else None

        # Parse date column if exists
        if 'date' in train_df.columns:
            train_df['date'] = pd.to_datetime(train_df['date'])
            train_df = train_df.sort_values('date').reset_index(drop=True)

        if test_df is not None and 'date' in test_df.columns:
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

    def create_time_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Create time-based features

        Args:
            df: Input dataframe
            date_col: Name of date column

        Returns:
            DataFrame with time features
        """
        df_copy = df.copy()
        if date_col in df_copy.columns:
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
                           windows: List[int] = [7, 14, 30, 60]) -> pd.DataFrame:
        """
        Create all features for time series modeling

        Args:
            df: Input dataframe
            target_col: Name of target column
            lags: List of lag values
            windows: List of rolling window sizes

        Returns:
            DataFrame with all features
        """
        df_features = df.copy()

        # Time features
        df_features = self.create_time_features(df_features)

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
