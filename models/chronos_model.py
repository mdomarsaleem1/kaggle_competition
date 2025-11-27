"""
Chronos-2: Universal Time Series Forecasting Foundation Model
Amazon's pre-trained time series forecasting model
"""
import numpy as np
import pandas as pd
import torch
from typing import Optional, List, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ChronosTimeSeriesModel:
    """Chronos-2 foundation model for time series forecasting"""

    def __init__(self, model_size: str = 'small', device: Optional[str] = None):
        """
        Initialize Chronos-2 model

        Args:
            model_size: Model size ('tiny', 'mini', 'small', 'base', 'large')
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        self.model_size = model_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.pipeline = None

        # Model sizes and their HuggingFace paths
        self.model_paths = {
            'tiny': 'amazon/chronos-t5-tiny',
            'mini': 'amazon/chronos-t5-mini',
            'small': 'amazon/chronos-t5-small',
            'base': 'amazon/chronos-t5-base',
            'large': 'amazon/chronos-t5-large'
        }

    def load_model(self):
        """Load pre-trained Chronos model"""
        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError(
                "Chronos not installed. Install it with:\n"
                "pip install git+https://github.com/amazon-science/chronos-forecasting.git"
            )

        model_path = self.model_paths.get(self.model_size)
        if not model_path:
            raise ValueError(f"Unknown model size: {self.model_size}. "
                           f"Choose from {list(self.model_paths.keys())}")

        print(f"Loading Chronos-2 model: {model_path}")
        print(f"Device: {self.device}")

        self.pipeline = ChronosPipeline.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32
        )

        print("Model loaded successfully")

    def predict(self, context: Union[np.ndarray, List[float], pd.Series],
                prediction_length: int,
                num_samples: int = 20,
                temperature: float = 1.0,
                top_k: Optional[int] = 50,
                top_p: Optional[float] = 1.0) -> np.ndarray:
        """
        Generate forecasts using Chronos

        Args:
            context: Historical time series data
            prediction_length: Number of steps to forecast
            num_samples: Number of sample trajectories to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Array of forecasts with shape (num_samples, prediction_length)
        """
        if self.pipeline is None:
            self.load_model()

        # Convert input to tensor
        if isinstance(context, pd.Series):
            context = context.values
        elif isinstance(context, list):
            context = np.array(context)

        context_tensor = torch.tensor(context, dtype=torch.float32)

        # Generate forecasts
        forecast = self.pipeline.predict(
            context_tensor,
            prediction_length=prediction_length,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        return forecast.numpy()

    def predict_quantiles(self, context: Union[np.ndarray, List[float], pd.Series],
                         prediction_length: int,
                         quantiles: List[float] = [0.1, 0.5, 0.9],
                         num_samples: int = 100) -> pd.DataFrame:
        """
        Generate quantile forecasts

        Args:
            context: Historical time series data
            prediction_length: Number of steps to forecast
            quantiles: List of quantiles to compute
            num_samples: Number of samples for quantile estimation

        Returns:
            DataFrame with quantile forecasts
        """
        forecasts = self.predict(
            context=context,
            prediction_length=prediction_length,
            num_samples=num_samples
        )

        # Compute quantiles
        quantile_forecasts = {}
        for q in quantiles:
            quantile_forecasts[f'q{int(q*100)}'] = np.quantile(forecasts, q, axis=0)

        # Add mean forecast
        quantile_forecasts['mean'] = forecasts.mean(axis=0)

        return pd.DataFrame(quantile_forecasts)

    def batch_predict(self, contexts: List[Union[np.ndarray, List[float]]],
                     prediction_length: int,
                     num_samples: int = 20,
                     batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate forecasts for multiple time series

        Args:
            contexts: List of historical time series
            prediction_length: Number of steps to forecast
            num_samples: Number of sample trajectories per series
            batch_size: Batch size for processing

        Returns:
            List of forecast arrays
        """
        if self.pipeline is None:
            self.load_model()

        all_forecasts = []

        for i in range(0, len(contexts), batch_size):
            batch = contexts[i:i + batch_size]

            # Convert to tensors
            batch_tensors = [torch.tensor(ctx, dtype=torch.float32) for ctx in batch]

            # Generate forecasts
            batch_forecasts = self.pipeline.predict(
                batch_tensors,
                prediction_length=prediction_length,
                num_samples=num_samples
            )

            all_forecasts.extend([f.numpy() for f in batch_forecasts])

        return all_forecasts

    def evaluate(self, test_data: pd.DataFrame,
                context_length: int,
                prediction_length: int,
                target_col: str = 'value') -> dict:
        """
        Evaluate model on test data

        Args:
            test_data: Test dataframe
            context_length: Length of context window
            prediction_length: Length of forecast horizon
            target_col: Name of target column

        Returns:
            Dictionary with evaluation metrics
        """
        if self.pipeline is None:
            self.load_model()

        values = test_data[target_col].values
        predictions = []
        actuals = []

        # Rolling window evaluation
        for i in range(context_length, len(values) - prediction_length + 1):
            context = values[i - context_length:i]
            actual = values[i:i + prediction_length]

            # Get median forecast
            forecast_samples = self.predict(
                context=context,
                prediction_length=prediction_length,
                num_samples=20
            )
            forecast = np.median(forecast_samples, axis=0)

            predictions.append(forecast)
            actuals.append(actual)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals.flatten(), predictions.flatten()))
        mae = mean_absolute_error(actuals.flatten(), predictions.flatten())

        # Calculate per-horizon metrics
        horizon_rmse = [
            np.sqrt(mean_squared_error(actuals[:, h], predictions[:, h]))
            for h in range(prediction_length)
        ]

        return {
            'rmse': rmse,
            'mae': mae,
            'horizon_rmse': horizon_rmse,
            'num_predictions': len(predictions)
        }


class ChronosEnsemble:
    """Ensemble of multiple Chronos models"""

    def __init__(self, model_sizes: List[str] = ['tiny', 'small', 'base']):
        """
        Initialize ensemble of Chronos models

        Args:
            model_sizes: List of model sizes to ensemble
        """
        self.models = [ChronosTimeSeriesModel(size) for size in model_sizes]
        self.model_sizes = model_sizes

    def load_models(self):
        """Load all models in ensemble"""
        for i, model in enumerate(self.models):
            print(f"\nLoading model {i+1}/{len(self.models)}: {self.model_sizes[i]}")
            model.load_model()

    def predict(self, context: Union[np.ndarray, List[float], pd.Series],
                prediction_length: int,
                num_samples: int = 20,
                method: str = 'mean') -> np.ndarray:
        """
        Generate ensemble forecasts

        Args:
            context: Historical time series data
            prediction_length: Number of steps to forecast
            num_samples: Number of sample trajectories per model
            method: Ensemble method ('mean', 'median', 'weighted')

        Returns:
            Ensemble forecast
        """
        all_forecasts = []

        for model in self.models:
            forecasts = model.predict(
                context=context,
                prediction_length=prediction_length,
                num_samples=num_samples
            )
            # Take median of samples for each model
            all_forecasts.append(np.median(forecasts, axis=0))

        all_forecasts = np.array(all_forecasts)

        if method == 'mean':
            return np.mean(all_forecasts, axis=0)
        elif method == 'median':
            return np.median(all_forecasts, axis=0)
        elif method == 'weighted':
            # Equal weights (could be improved with model-specific weights)
            weights = np.ones(len(self.models)) / len(self.models)
            return np.average(all_forecasts, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")


class ChronosFineTuner:
    """Fine-tune Chronos model on custom data"""

    def __init__(self, model_size: str = 'small', device: Optional[str] = None):
        """
        Initialize Chronos fine-tuner

        Args:
            model_size: Model size to fine-tune
            device: Device to use
        """
        self.base_model = ChronosTimeSeriesModel(model_size, device)
        print("Note: Fine-tuning Chronos requires additional setup.")
        print("See: https://github.com/amazon-science/chronos-forecasting")

    def fine_tune(self, train_data: List[np.ndarray],
                 val_data: Optional[List[np.ndarray]] = None,
                 epochs: int = 10,
                 learning_rate: float = 1e-4,
                 batch_size: int = 32):
        """
        Fine-tune Chronos model (placeholder - requires additional setup)

        Args:
            train_data: List of training time series
            val_data: List of validation time series
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
        """
        raise NotImplementedError(
            "Fine-tuning Chronos requires additional setup. "
            "Please refer to the official Chronos documentation:\n"
            "https://github.com/amazon-science/chronos-forecasting"
        )
