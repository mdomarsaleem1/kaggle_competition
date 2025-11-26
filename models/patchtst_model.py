"""
PatchTST: A Time Series is Worth 64 Words
Paper: https://arxiv.org/abs/2211.14730 (ICLR 2023)

PatchTST is a Transformer-based model that:
1. Segments time series into patches
2. Uses channel-independence (treats each channel separately)
3. Applies self-attention on patches
4. Achieves SOTA performance on long-term forecasting
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from sklearn.metrics import mean_squared_error, mean_absolute_error


class PatchEmbedding(nn.Module):
    """Convert time series into patches and embed them"""

    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        # Linear projection of flattened patches
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, n_features]
        Returns:
            patches: [batch_size, n_patches, d_model]
        """
        batch_size, seq_len, n_features = x.shape

        # Create patches using unfold
        # x: [batch_size, n_features, seq_len]
        x = x.transpose(1, 2)

        # Unfold: [batch_size, n_features, n_patches, patch_len]
        n_patches = (seq_len - self.patch_len) // self.stride + 1
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)

        # Reshape: [batch_size * n_features, n_patches, patch_len]
        patches = patches.reshape(-1, n_patches, self.patch_len)

        # Embed: [batch_size * n_features, n_patches, d_model]
        patches = self.value_embedding(patches)
        patches = self.dropout(patches)

        return patches, n_features


class PatchTSTEncoder(nn.Module):
    """Transformer encoder for PatchTST"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


class PatchTSTModel(nn.Module):
    """
    PatchTST: Transformer model for time series forecasting

    Architecture:
    1. Patch embedding: Convert time series to patches
    2. Positional encoding: Add position information
    3. Transformer encoder: Self-attention on patches
    4. Prediction head: Project to forecast horizon
    """

    def __init__(self,
                 seq_len: int,
                 pred_len: int,
                 n_features: int = 1,
                 patch_len: int = 16,
                 stride: int = 8,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 channel_independence: bool = True):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.channel_independence = channel_independence

        # Patch embedding
        self.patch_embedding = PatchEmbedding(patch_len, stride, d_model, dropout)

        # Calculate number of patches
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.n_patches, d_model)
        )

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            PatchTSTEncoder(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Prediction head
        self.head = nn.Linear(d_model * self.n_patches, pred_len)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, n_features]
        Returns:
            forecast: [batch_size, pred_len, n_features]
        """
        batch_size = x.shape[0]

        # Patch embedding: [batch_size * n_features, n_patches, d_model]
        x, n_features = self.patch_embedding(x)

        # Add positional encoding
        x = x + self.positional_encoding

        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # Flatten patches: [batch_size * n_features, n_patches * d_model]
        x = x.reshape(batch_size * n_features, -1)

        # Prediction head: [batch_size * n_features, pred_len]
        forecast = self.head(x)

        # Reshape to [batch_size, pred_len, n_features]
        forecast = forecast.reshape(batch_size, n_features, self.pred_len)
        forecast = forecast.transpose(1, 2)

        return forecast


class PatchTSTTimeSeriesModel:
    """Wrapper for PatchTST model"""

    def __init__(self,
                 seq_len: int = 96,
                 pred_len: int = 24,
                 n_features: int = 1,
                 patch_len: int = 16,
                 stride: int = 8,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-4,
                 batch_size: int = 32,
                 epochs: int = 100,
                 patience: int = 10,
                 device: Optional[str] = None):
        """
        Initialize PatchTST model

        Args:
            seq_len: Input sequence length
            pred_len: Prediction length
            n_features: Number of features
            patch_len: Length of each patch
            stride: Stride for patch extraction
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Maximum training epochs
            patience: Early stopping patience
            device: Device to use ('cpu', 'cuda', or None for auto)
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
        self.model = PatchTSTModel(
            seq_len=seq_len,
            pred_len=pred_len,
            n_features=n_features,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
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
        """
        Train PatchTST model

        Args:
            train_data: Training time series [samples, features]
            val_data: Validation time series
            verbose: Print training progress

        Returns:
            Dictionary with training metrics
        """
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
        """
        Make predictions

        Args:
            data: Input time series [samples, features]
            return_sequences: If True, return full sequences; else return point forecasts

        Returns:
            Predictions
        """
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
            # Return only the first forecast from each sequence
            return predictions[:, 0, :]

    def save_model(self, filepath: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': {
                'seq_len': self.seq_len,
                'pred_len': self.pred_len,
                'n_features': self.n_features,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'patience': self.patience,
                'learning_rate': self.learning_rate
            }
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Model loaded from {filepath}")
