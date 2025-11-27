# %% [markdown]
# # PatchTST: Patch-based Time Series Transformer
# 
# ## 📚 Overview
# 
# **PatchTST** ("A Time Series is Worth 64 Words") is a state-of-the-art transformer model from ICLR 2023.
# 
# ### Key Innovation
# - Treats time series as **patches** (similar to Vision Transformers)
# - Reduces sequence length by 8x → more efficient training
# - Achieves **SOTA performance** on long-term forecasting benchmarks
# 
# ### Architecture
# ```
# Time Series [96] → Patches [12 x 8] → Transformer → Predictions [24]
# ```
# 
# ### Performance
# - **+40%** improvement vs standard Transformer
# - **+15%** improvement vs previous SOTA
# - Works well on both univariate and multivariate data

# %% [markdown]
# ## 🔧 Setup

# %%
import sys
import os
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from models import PatchTSTTimeSeriesModel
from utils.data_utils import TimeSeriesPreprocessor

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# %% [markdown]
# ## 📊 Load and Prepare Data

# %%
# Load your data
# Expected format: CSV with 'date' column and target column
data_path = '../data/train.csv'

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
else:
    print(f"Data file not found at {data_path}")
    print("Creating synthetic data for demonstration...")
    
    # Create synthetic time series
    n_points = 1000
    dates = pd.date_range('2020-01-01', periods=n_points, freq='D')
    
    # Trend + Seasonality + Noise
    trend = np.linspace(100, 200, n_points)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_points) / 365)
    noise = np.random.normal(0, 5, n_points)
    
    df = pd.DataFrame({
        'date': dates,
        'target': trend + seasonality + noise
    })
    print(f"Created synthetic data with shape: {df.shape}")
    print(df.head())

# %% [markdown]
# ## 📈 Visualize the Data

# %%
plt.figure(figsize=(15, 5))
plt.plot(df['target'].values)
plt.title('Time Series Data', fontsize=14, fontweight='bold')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## 🔄 Prepare Data for PatchTST

# %%
# Extract target values
target_col = 'target'
data = df[target_col].values

# Reshape for model (samples, features)
if data.ndim == 1:
    data = data.reshape(-1, 1)

print(f"Data shape: {data.shape}")

# Train/validation split (80/20)
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
val_data = data[split_idx:]

print(f"Train data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")

# %% [markdown]
# ## 🏗️ Initialize PatchTST Model
# 
# ### Hyperparameters
# - **seq_len**: Input sequence length (e.g., 96 = ~3 months of daily data)
# - **pred_len**: Forecast horizon (e.g., 24 = 24 days)
# - **patch_len**: Length of each patch (default: 16)
# - **stride**: Stride between patches (default: 8)
# - **d_model**: Dimension of model (default: 128)
# - **n_heads**: Number of attention heads (default: 8)
# - **n_layers**: Number of transformer layers (default: 3)

# %%
# Model configuration
seq_len = 96      # Input window: 96 time steps
pred_len = 24     # Forecast: 24 time steps ahead
n_features = data.shape[1]  # Number of features (1 for univariate)

# Initialize PatchTST
model = PatchTSTTimeSeriesModel(
    seq_len=seq_len,
    pred_len=pred_len,
    n_features=n_features,
    patch_len=16,        # Each patch is 16 time steps
    stride=8,            # 50% overlap between patches
    d_model=128,         # Model dimension
    n_heads=8,           # 8 attention heads
    n_layers=3,          # 3 transformer layers
    d_ff=256,            # Feed-forward dimension
    dropout=0.1,
    epochs=50,           # Training epochs
    batch_size=32,
    learning_rate=0.001,
    device=device
)

print(f"\nPatchTST Model Initialized")
print(f"Input: {seq_len} time steps")
print(f"Output: {pred_len} time steps")
print(f"Patches: {(seq_len - 16) // 8 + 1} patches of length 16")

# %% [markdown]
# ## 🎓 Train the Model

# %%
# Train PatchTST
print("Training PatchTST...")
metrics = model.train(train_data, val_data, verbose=True)

print("\n" + "="*50)
print("Training Complete!")
print("="*50)
print(f"Final Validation RMSE: {metrics.get('val_rmse', 'N/A'):.4f}")
print(f"Final Validation MAE: {metrics.get('val_mae', 'N/A'):.4f}")

# %% [markdown]
# ## 📊 Visualize Training History

# %%
if 'train_losses' in metrics and 'val_losses' in metrics:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training and validation loss
    ax1.plot(metrics['train_losses'], label='Training Loss', linewidth=2)
    ax1.plot(metrics['val_losses'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training History', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Learning curve
    best_epoch = np.argmin(metrics['val_losses'])
    ax2.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch}')
    ax2.plot(metrics['val_losses'], linewidth=2, color='orange')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Curve', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 🔮 Make Predictions

# %%
# Make predictions on validation data
predictions = model.predict(val_data, return_sequences=False)

print(f"Predictions shape: {predictions.shape}")
print(f"First 5 predictions: {predictions[:5]}")

# %% [markdown]
# ## 📈 Visualize Predictions

# %%
# Get ground truth values
# For each prediction, the ground truth is pred_len steps ahead
n_samples = min(len(predictions), len(val_data) - pred_len)
ground_truth = []
for i in range(n_samples):
    ground_truth.append(val_data[i + pred_len, 0])
ground_truth = np.array(ground_truth)

# Truncate predictions if needed
predictions_plot = predictions[:n_samples]

# Plot
plt.figure(figsize=(15, 6))

# Plot ground truth and predictions
plt.plot(ground_truth, label='Ground Truth', linewidth=2, alpha=0.7)
plt.plot(predictions_plot, label='PatchTST Predictions', linewidth=2, alpha=0.7)

plt.title(f'PatchTST Forecasts (Horizon: {pred_len} steps)', fontsize=14, fontweight='bold')
plt.xlabel('Sample', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate error metrics
mae = np.mean(np.abs(ground_truth - predictions_plot))
rmse = np.sqrt(np.mean((ground_truth - predictions_plot) ** 2))
mape = np.mean(np.abs((ground_truth - predictions_plot) / (ground_truth + 1e-8))) * 100

print(f"\nPrediction Metrics:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

# %% [markdown]
# ## 🔍 Detailed Forecast Example

# %%
# Take a specific window and forecast
test_idx = 0
context = val_data[test_idx:test_idx + seq_len]

# Make forecast
forecast = model.predict(context.reshape(1, -1, 1))
if forecast.ndim > 1:
    forecast = forecast[0]

# Get actual future values
actual_future = val_data[test_idx + seq_len:test_idx + seq_len + pred_len, 0]

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Historical context
ax1.plot(range(seq_len), context[:, 0], label='Historical Context', linewidth=2)
ax1.axvline(seq_len - 1, color='red', linestyle='--', alpha=0.5, label='Forecast Start')
ax1.set_title('Input Context (Historical Data)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time Step', fontsize=12)
ax1.set_ylabel('Value', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Forecast vs actual
forecast_range = range(seq_len, seq_len + pred_len)
ax2.plot(range(seq_len), context[:, 0], label='Historical Context', linewidth=2, alpha=0.5)
ax2.plot(forecast_range, actual_future, 'g-', label='Actual Future', linewidth=2, marker='o')
ax2.plot(forecast_range, forecast, 'r--', label='PatchTST Forecast', linewidth=2, marker='s')
ax2.axvline(seq_len - 1, color='red', linestyle='--', alpha=0.5)
ax2.set_title('Forecast vs Actual', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time Step', fontsize=12)
ax2.set_ylabel('Value', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Forecast error
forecast_mae = np.mean(np.abs(actual_future - forecast))
print(f"\nForecast MAE: {forecast_mae:.4f}")

# %% [markdown]
# ## 💾 Save the Model

# %%
# Save model
model_path = '../trained_models/patchtst_model.pth'
os.makedirs('../trained_models', exist_ok=True)

model.save_model(model_path)
print(f"Model saved to {model_path}")

# %% [markdown]
# ## 🔄 Load and Use Saved Model

# %%
# Load saved model
loaded_model = PatchTSTTimeSeriesModel(
    seq_len=seq_len,
    pred_len=pred_len,
    n_features=n_features,
    device=device
)

loaded_model.load_model(model_path)
print("Model loaded successfully!")

# Make predictions with loaded model
loaded_predictions = loaded_model.predict(context.reshape(1, -1, 1))
print(f"\nPredictions from loaded model: {loaded_predictions[0][:5]}...")

# %% [markdown]
# ## 🎯 Key Takeaways
# 
# ### PatchTST Advantages
# 1. **Efficiency**: 8x faster training than standard Transformers
# 2. **Performance**: SOTA results on long-term forecasting
# 3. **Flexibility**: Works with both univariate and multivariate data
# 4. **Scalability**: Can handle long sequences efficiently
# 
# ### When to Use PatchTST
# - ✅ Long-term forecasting (24+ steps ahead)
# - ✅ High-frequency data (hourly, daily)
# - ✅ When you have sufficient training data (1000+ samples)
# - ✅ When computational efficiency matters
# 
# ### Hyperparameter Tuning Tips
# - **patch_len**: Larger patches → fewer patches → faster training
# - **stride**: Smaller stride → more patches → better accuracy (but slower)
# - **d_model**: Larger model → more capacity (but needs more data)
# - **n_layers**: 3-5 layers usually optimal
# 
# ### Next Steps
# 1. Try different patch configurations
# 2. Experiment with multivariate data
# 3. Compare with other models (iTransformer, TimesNet)
# 4. Use in ensemble with other models


