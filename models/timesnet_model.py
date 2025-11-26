"""
TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
Paper: https://arxiv.org/abs/2210.02186 (ICLR 2023)

Key innovation: Transform 1D time series into 2D tensors to capture temporal variations
- Uses FFT to find multiple periods in the time series
- Reshapes 1D series into 2D based on periods
- Applies 2D convolutions to capture intraperiod and interperiod variations
"""
import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft
from typing import Optional, Tuple, List
from sklearn.metrics import mean_squared_error, mean_absolute_error


class TimesBlock(nn.Module):
    """
    TimesBlock: Core building block that transforms 1D to 2D and back

    Steps:
    1. FFT to find dominant periods
    2. Reshape time series into 2D based on periods
    3. Apply 2D inception block
    4. Aggregate multi-period results
    """

    def __init__(self, d_model: int, d_ff: int, num_kernels: int = 6, top_k: int = 5):
        super().__init__()

        self.seq_len = None
        self.top_k = top_k
        self.d_model = d_model
        self.num_kernels = num_kernels

        # Parameter for learnable aggregation
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_ff, kernel_size=(1, 1)),
            nn.GELU(),
            nn.Conv2d(d_ff, d_model, kernel_size=(1, 1))
        )

        # Inception block for 2D convolution
        self.inception = InceptionBlock(d_model, num_kernels)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        self.seq_len = seq_len

        # Find top-k periods using FFT
        period_list, period_weight = self._fft_for_period(x)

        # Process each period
        res = []
        for i in range(self.top_k):
            period = period_list[i]

            # Padding
            if seq_len % period != 0:
                length = ((seq_len // period) + 1) * period
                padding = torch.zeros([batch_size, (length - seq_len), d_model]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = seq_len
                out = x

            # Reshape to 2D: [batch, d_model, period, length // period]
            out = out.reshape(batch_size, length // period, period, d_model)
            out = out.permute(0, 3, 2, 1).contiguous()  # [batch, d_model, period, num_periods]

            # 2D convolution
            out = self.inception(out)

            # Reshape back to 1D
            out = out.permute(0, 3, 2, 1).contiguous()
            out = out.reshape(batch_size, -1, d_model)
            out = out[:, :seq_len, :]

            res.append(out)

        # Aggregate results from different periods
        res = torch.stack(res, dim=-1)  # [batch, seq_len, d_model, top_k]

        # Weighted aggregation
        period_weight = torch.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, seq_len, d_model, 1)
        res = torch.sum(res * period_weight, -1)  # [batch, seq_len, d_model]

        # Residual connection
        res = res + x

        return res

    def _fft_for_period(self, x):
        """
        Use FFT to find dominant periods

        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            period_list: List of periods
            period_weight: Weights for periods
        """
        # FFT
        xf = fft.rfft(x, dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0  # Remove DC component

        # Find top-k frequencies
        _, top_list = torch.topk(frequency_list, self.top_k)
        top_list = top_list.detach().cpu().numpy()

        # Convert frequency to period
        period_list = [max(1, self.seq_len // freq) for freq in top_list]
        period_weight = frequency_list[top_list]

        return period_list, period_weight


class InceptionBlock(nn.Module):
    """Inception block for multi-scale 2D convolution"""

    def __init__(self, in_channels: int, num_kernels: int = 6):
        super().__init__()

        self.num_kernels = num_kernels

        # Multiple kernel sizes for multi-scale feature extraction
        self.conv_list = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 2 * i + 1), padding=(0, i))
            for i in range(1, num_kernels + 1)
        ])

    def forward(self, x):
        """
        Args:
            x: [batch, channels, height, width]
        """
        res_list = []
        for conv in self.conv_list:
            res_list.append(conv(x))

        # Max pooling aggregation
        res = torch.stack(res_list, dim=-1)
        res = torch.max(res, dim=-1)[0]

        return res


class TimesNetModel(nn.Module):
    """
    TimesNet: Complete model for time series forecasting

    Architecture:
    1. Embedding layer
    2. Multiple TimesBlocks
    3. Projection head
    """

    def __init__(self,
                 seq_len: int,
                 pred_len: int,
                 n_features: int,
                 d_model: int = 64,
                 d_ff: int = 128,
                 n_layers: int = 2,
                 num_kernels: int = 6,
                 top_k: int = 5,
                 dropout: float = 0.1):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len

        # Embedding
        self.embedding = nn.Linear(n_features, d_model)

        # TimesBlocks
        self.blocks = nn.ModuleList([
            TimesBlock(d_model, d_ff, num_kernels, top_k)
            for _ in range(n_layers)
        ])

        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)

        # Projection
        self.projection = nn.Linear(d_model, n_features)
        self.pred_projection = nn.Linear(seq_len, pred_len)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, n_features]
        Returns:
            forecast: [batch_size, pred_len, n_features]
        """
        # Embedding
        x = self.embedding(x)  # [batch, seq_len, d_model]

        # TimesBlocks
        for block in self.blocks:
            x = block(x)

        # Layer norm
        x = self.layer_norm(x)

        # Project back to features
        x = self.projection(x)  # [batch, seq_len, n_features]

        # Transpose and project to prediction length
        x = x.transpose(1, 2)  # [batch, n_features, seq_len]
        forecast = self.pred_projection(x)  # [batch, n_features, pred_len]
        forecast = forecast.transpose(1, 2)  # [batch, pred_len, n_features]

        return forecast


class TimesNetTimeSeriesModel:
    """Wrapper for TimesNet model"""

    def __init__(self,
                 seq_len: int = 96,
                 pred_len: int = 24,
                 n_features: int = 1,
                 d_model: int = 64,
                 d_ff: int = 128,
                 n_layers: int = 2,
                 num_kernels: int = 6,
                 top_k: int = 5,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 epochs: int = 100,
                 patience: int = 10,
                 device: Optional[str] = None):
        """
        Initialize TimesNet model

        Args:
            seq_len: Input sequence length
            pred_len: Prediction length
            n_features: Number of features
            d_model: Model dimension
            d_ff: Feed-forward dimension
            n_layers: Number of TimesBlocks
            num_kernels: Number of kernels in inception block
            top_k: Number of top periods to use
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Maximum training epochs
            patience: Early stopping patience
            device: Device to use
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        self.model = TimesNetModel(
            seq_len=seq_len,
            pred_len=pred_len,
            n_features=n_features,
            d_model=d_model,
            d_ff=d_ff,
            n_layers=n_layers,
            num_kernels=num_kernels,
            top_k=top_k,
            dropout=dropout
        ).to(self.device)

        self.optimizer = None
        self.criterion = nn.MSELoss()

    def _create_sequences(self, data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create input-output sequences"""
        X, y = [], []

        for i in range(len(data) - self.seq_len - self.pred_len + 1):
            X.append(data[i:i + self.seq_len])
            y.append(data[i + self.seq_len:i + self.seq_len + self.pred_len])

        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

    def train(self, train_data: np.ndarray,
              val_data: Optional[np.ndarray] = None,
              verbose: bool = True) -> dict:
        """Train TimesNet model"""
        # Reshape data if needed
        if train_data.ndim == 1:
            train_data = train_data.reshape(-1, 1)

        self.n_features = train_data.shape[1]

        # Create sequences
        X_train, y_train = self._create_sequences(train_data)

        if val_data is not None:
            if val_data.ndim == 1:
                val_data = val_data.reshape(-1, 1)
            X_val, y_val = self._create_sequences(val_data)
        else:
            X_val, y_val = None, None

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_device = X_val.to(self.device)
                    y_val_device = y_val.to(self.device)
                    val_pred = self.model(X_val_device)
                    val_loss = self.criterion(val_pred, y_val_device).item()
                    val_losses.append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.epochs} - "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.6f}")

        # Calculate final metrics
        metrics = {
            'train_loss': train_losses[-1],
            'epochs_trained': len(train_losses)
        }

        if val_data is not None:
            metrics['val_loss'] = val_losses[-1]
            val_pred_np = val_pred.cpu().numpy()
            y_val_np = y_val.cpu().numpy()
            metrics['val_rmse'] = np.sqrt(mean_squared_error(y_val_np.flatten(), val_pred_np.flatten()))
            metrics['val_mae'] = mean_absolute_error(y_val_np.flatten(), val_pred_np.flatten())

        return metrics

    def predict(self, data: np.ndarray, return_sequences: bool = False) -> np.ndarray:
        """Make predictions"""
        self.model.eval()

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Create sequences
        X, _ = self._create_sequences(data)
        X = X.to(self.device)

        with torch.no_grad():
            predictions = self.model(X)

        predictions = predictions.cpu().numpy()

        if return_sequences:
            return predictions
        else:
            return predictions[:, 0, :]

    def save_model(self, filepath: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': {
                'seq_len': self.seq_len,
                'pred_len': self.pred_len,
                'n_features': self.n_features
            }
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")
