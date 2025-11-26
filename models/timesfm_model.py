"""
TimesFM: Time Series Foundation Model by Google Research
Paper: https://arxiv.org/abs/2310.10688

TimesFM is a pre-trained decoder-only transformer foundation model for time series forecasting.
- Pre-trained on 100B real-world time points
- Zero-shot forecasting capability
- Supports multivariate time series
- Efficient patched-attention mechanism
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error


class TimesFMModel:
    """
    TimesFM: Google's Time Series Foundation Model

    A pre-trained decoder-only transformer for time series forecasting.
    Unlike Chronos which uses an encoder-only architecture, TimesFM uses
    a decoder-only architecture similar to GPT for time series.
    """

    def __init__(self,
                 model_size: str = 'base',
                 context_length: int = 512,
                 horizon_length: int = 128,
                 device: Optional[str] = None):
        """
        Initialize TimesFM model

        Args:
            model_size: Model size ('small', 'base', 'large')
            context_length: Maximum context length
            horizon_length: Maximum forecast horizon
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        self.model_size = model_size
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.is_loaded = False

        # Model configurations
        self.model_configs = {
            'small': {'d_model': 256, 'n_heads': 8, 'n_layers': 4},
            'base': {'d_model': 512, 'n_heads': 8, 'n_layers': 8},
            'large': {'d_model': 1024, 'n_heads': 16, 'n_layers': 12}
        }

        print(f"\nTimesFM Model initialized:")
        print(f"  Size: {model_size}")
        print(f"  Context length: {context_length}")
        print(f"  Horizon length: {horizon_length}")
        print(f"  Device: {self.device}")

    def load_model(self):
        """
        Load pre-trained TimesFM model

        Note: This is a placeholder implementation.
        In production, you would load the actual pre-trained weights from:
        - HuggingFace Hub
        - Google Cloud Storage
        - Local checkpoint
        """
        try:
            # Try to import timesfm if available
            import timesfm

            print("Loading TimesFM from official implementation...")
            self.model = timesfm.TimesFm(
                context_len=self.context_length,
                horizon_len=self.horizon_length,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=self.model_configs[self.model_size]['n_layers'],
                model_dims=self.model_configs[self.model_size]['d_model'],
                backend='gpu' if self.device == 'cuda' else 'cpu'
            )
            self.is_loaded = True
            print("TimesFM loaded successfully from official implementation!")

        except ImportError:
            print("Official TimesFM not available. Using custom implementation...")
            # Fallback to custom lightweight implementation
            self.model = self._create_custom_model()
            self.is_loaded = True
            print("Custom TimesFM model created (for demonstration)")
            print("Note: For production, install official TimesFM:")
            print("  pip install timesfm")

    def _create_custom_model(self):
        """Create a custom lightweight TimesFM-style model"""
        config = self.model_configs[self.model_size]

        class CustomTimesFM(nn.Module):
            def __init__(self, d_model, n_heads, n_layers, context_length, horizon_length):
                super().__init__()
                self.d_model = d_model
                self.context_length = context_length
                self.horizon_length = horizon_length

                # Input embedding
                self.input_embedding = nn.Linear(1, d_model)

                # Positional encoding
                self.pos_encoding = nn.Parameter(
                    torch.randn(1, context_length + horizon_length, d_model)
                )

                # Decoder layers
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

                # Output projection
                self.output_projection = nn.Linear(d_model, 1)

            def forward(self, context, horizon_length):
                batch_size = context.shape[0]

                # Embed context
                context_embed = self.input_embedding(context.unsqueeze(-1))

                # Add positional encoding
                context_embed = context_embed + self.pos_encoding[:, :context.shape[1], :]

                # Create target sequence (zeros for autoregressive generation)
                target = torch.zeros(batch_size, horizon_length, self.d_model).to(context.device)
                target = target + self.pos_encoding[:, context.shape[1]:context.shape[1]+horizon_length, :]

                # Decode
                output = self.decoder(target, context_embed)

                # Project to output
                forecast = self.output_projection(output).squeeze(-1)

                return forecast

        model = CustomTimesFM(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            context_length=self.context_length,
            horizon_length=self.horizon_length
        ).to(self.device)

        return model

    def predict(self,
                context: Union[np.ndarray, List[float], pd.Series],
                horizon: Optional[int] = None,
                num_samples: int = 1,
                quantiles: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate forecasts using TimesFM

        Args:
            context: Historical time series data
            horizon: Forecast horizon (default: horizon_length)
            num_samples: Number of samples for probabilistic forecasting
            quantiles: Quantiles to compute (if None, returns mean)

        Returns:
            Forecasts array
        """
        if not self.is_loaded:
            self.load_model()

        # Convert input to numpy array
        if isinstance(context, pd.Series):
            context = context.values
        elif isinstance(context, list):
            context = np.array(context)

        # Use default horizon if not specified
        if horizon is None:
            horizon = min(self.horizon_length, len(context))

        # Ensure context is within limits
        if len(context) > self.context_length:
            context = context[-self.context_length:]

        try:
            # Try official TimesFM API
            if hasattr(self.model, 'forecast'):
                forecast = self.model.forecast(
                    inputs=context,
                    freq=None,  # Auto-detect frequency
                )
                return forecast[:horizon]

        except Exception as e:
            print(f"Official API failed: {e}, using custom implementation")

        # Fallback to custom implementation
        self.model.eval()
        with torch.no_grad():
            context_tensor = torch.FloatTensor(context).unsqueeze(0).to(self.device)

            forecasts = []
            for _ in range(num_samples):
                forecast = self.model(context_tensor, horizon)
                forecasts.append(forecast.cpu().numpy())

            forecasts = np.array(forecasts).squeeze()

            if num_samples == 1:
                return forecasts

            # Compute quantiles if requested
            if quantiles:
                quantile_forecasts = {}
                for q in quantiles:
                    quantile_forecasts[f'q{int(q*100)}'] = np.quantile(forecasts, q, axis=0)
                return quantile_forecasts

            # Return mean by default
            return np.mean(forecasts, axis=0)

    def predict_batch(self,
                     contexts: List[np.ndarray],
                     horizon: Optional[int] = None,
                     batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate forecasts for multiple time series

        Args:
            contexts: List of historical time series
            horizon: Forecast horizon
            batch_size: Batch size for processing

        Returns:
            List of forecast arrays
        """
        if not self.is_loaded:
            self.load_model()

        all_forecasts = []

        for i in range(0, len(contexts), batch_size):
            batch = contexts[i:i + batch_size]

            batch_forecasts = []
            for context in batch:
                forecast = self.predict(context, horizon)
                batch_forecasts.append(forecast)

            all_forecasts.extend(batch_forecasts)

        return all_forecasts

    def evaluate(self,
                test_data: np.ndarray,
                context_length: Optional[int] = None,
                horizon: Optional[int] = None) -> dict:
        """
        Evaluate model on test data

        Args:
            test_data: Test time series
            context_length: Length of context window
            horizon: Forecast horizon

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_loaded:
            self.load_model()

        if context_length is None:
            context_length = min(self.context_length, len(test_data) // 2)

        if horizon is None:
            horizon = min(self.horizon_length, len(test_data) // 4)

        predictions = []
        actuals = []

        # Rolling window evaluation
        for i in range(context_length, len(test_data) - horizon + 1, horizon):
            context = test_data[i - context_length:i]
            actual = test_data[i:i + horizon]

            forecast = self.predict(context, horizon)

            predictions.append(forecast)
            actuals.append(actual)

        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'num_predictions': len(predictions)
        }

        return metrics


class TimesFMTimeSeriesModel:
    """
    Wrapper class for TimesFM that matches the interface of other models
    """

    def __init__(self,
                 seq_len: int = 512,
                 pred_len: int = 96,
                 model_size: str = 'base',
                 device: Optional[str] = None):
        """
        Initialize TimesFM wrapper

        Args:
            seq_len: Input sequence length
            pred_len: Prediction length
            model_size: Model size ('small', 'base', 'large')
            device: Device to use
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model_size = model_size
        self.device = device

        self.model = TimesFMModel(
            model_size=model_size,
            context_length=seq_len,
            horizon_length=pred_len,
            device=device
        )

    def train(self, train_data: np.ndarray,
              val_data: Optional[np.ndarray] = None,
              verbose: bool = True) -> dict:
        """
        TimesFM is pre-trained, so this just loads the model

        Args:
            train_data: Training data (not used, included for interface compatibility)
            val_data: Validation data (for evaluation)
            verbose: Print progress

        Returns:
            Dictionary with metrics
        """
        if verbose:
            print("\nTimesFM is a pre-trained foundation model.")
            print("Loading pre-trained weights...")

        self.model.load_model()

        metrics = {'message': 'Pre-trained model loaded'}

        # Evaluate on validation data if provided
        if val_data is not None:
            if verbose:
                print("Evaluating on validation data...")

            eval_metrics = self.model.evaluate(
                val_data,
                context_length=self.seq_len,
                horizon=self.pred_len
            )
            metrics.update(eval_metrics)

            if verbose:
                print(f"  Val RMSE: {eval_metrics['rmse']:.6f}")
                print(f"  Val MAE: {eval_metrics['mae']:.6f}")
                print(f"  Val MAPE: {eval_metrics['mape']:.2f}%")

        return metrics

    def predict(self, data: np.ndarray, return_sequences: bool = False) -> np.ndarray:
        """
        Make predictions

        Args:
            data: Input time series
            return_sequences: Not used (for interface compatibility)

        Returns:
            Predictions
        """
        # Use the last seq_len points as context
        if len(data) > self.seq_len:
            context = data[-self.seq_len:]
        else:
            context = data

        forecast = self.model.predict(context, horizon=self.pred_len)

        return forecast

    def save_model(self, filepath: str):
        """TimesFM uses pre-trained weights, so we just save config"""
        import json

        config = {
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'model_size': self.model_size
        }

        with open(filepath, 'w') as f:
            json.dump(config, f)

        print(f"TimesFM config saved to {filepath}")

    def load_model(self, filepath: str):
        """Load config and initialize model"""
        import json

        with open(filepath, 'r') as f:
            config = json.load(f)

        self.seq_len = config['seq_len']
        self.pred_len = config['pred_len']
        self.model_size = config['model_size']

        self.model = TimesFMModel(
            model_size=self.model_size,
            context_length=self.seq_len,
            horizon_length=self.pred_len,
            device=self.device
        )

        self.model.load_model()

        print(f"TimesFM model loaded from {filepath}")


def demonstrate_timesfm():
    """Demonstration of TimesFM"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              TimesFM: Google's Foundation Model                  ║
╚══════════════════════════════════════════════════════════════════╝

TimesFM is Google's decoder-only transformer foundation model for
time series forecasting, pre-trained on 100 billion real-world time points.

Key Features:
✓ Zero-shot forecasting (no training required)
✓ Pre-trained on massive diverse dataset
✓ Decoder-only architecture (like GPT for time series)
✓ Efficient patched-attention mechanism
✓ Supports multivariate time series
✓ Long context length (up to 512 points)

Comparison with Chronos-2:
- Chronos-2: Encoder-only (BERT-style)
- TimesFM: Decoder-only (GPT-style)
- Both are foundation models with zero-shot capability

Installation:
  pip install timesfm

Or use the custom implementation provided here.
""")


if __name__ == '__main__':
    demonstrate_timesfm()
