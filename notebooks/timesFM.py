# %% [markdown]
# # TimesFM: Google's Time Series Foundation Model
# 
# ## 📚 Overview
# 
# **TimesFM** is Google's decoder-only transformer foundation model for time series forecasting.
# 
# ### Key Features
# - **Pre-trained on 100 billion time points** from real-world data
# - **Decoder-only architecture** (GPT-style for time series)
# - **Zero-shot forecasting** capability
# - **Long context length** (up to 512 time steps)
# - **Efficient patched-attention** mechanism
# 
# ### TimesFM vs Chronos-2
# 
# | Feature | TimesFM | Chronos-2 |
# |---------|---------|----------|
# | Architecture | Decoder-only (GPT-style) | Encoder-only (BERT-style) |
# | Pre-training | 100B time points | 100K+ time series |
# | Approach | Autoregressive generation | Masked modeling |
# | Context Length | Up to 512 | Variable |
# 
# ### When to Use TimesFM
# - ✅ Limited training data (zero-shot capability)
# - ✅ New domains without historical patterns
# - ✅ Need for uncertainty quantification
# - ✅ Quick baseline without training

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

from models import TimesFMTimeSeriesModel

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Check if official TimesFM is available
try:
    import timesfm
    print("✅ Official TimesFM package available")
except ImportError:
    print("⚠️  Official TimesFM not found - using custom fallback implementation")
    print("   Install with: pip install timesfm")

# %% [markdown]
# ## 📊 Load and Prepare Data

# %%
# Load your data
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
    
    # Create synthetic time series with multiple patterns
    n_points = 1000
    dates = pd.date_range('2020-01-01', periods=n_points, freq='D')
    
    # Complex pattern: Trend + Multiple seasonalities + Noise
    trend = np.linspace(100, 200, n_points)
    yearly_seasonality = 30 * np.sin(2 * np.pi * np.arange(n_points) / 365)
    monthly_seasonality = 10 * np.sin(2 * np.pi * np.arange(n_points) / 30)
    weekly_seasonality = 5 * np.sin(2 * np.pi * np.arange(n_points) / 7)
    noise = np.random.normal(0, 5, n_points)
    
    df = pd.DataFrame({
        'date': dates,
        'target': trend + yearly_seasonality + monthly_seasonality + weekly_seasonality + noise
    })
    print(f"Created synthetic data with shape: {df.shape}")
    print(df.head())

# %% [markdown]
# ## 📈 Visualize the Data

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Full time series
ax1.plot(df['target'].values, linewidth=1.5)
ax1.set_title('Complete Time Series', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Value')
ax1.grid(True, alpha=0.3)

# Last 200 points (zoomed in)
ax2.plot(df['target'].values[-200:], linewidth=2, color='orange')
ax2.set_title('Last 200 Time Steps (Zoomed)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Value')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 🔄 Prepare Data for TimesFM

# %%
# Extract target values
target_col = 'target'
data = df[target_col].values

# Reshape for model
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
# ## 🏗️ Initialize TimesFM Model
# 
# ### Model Sizes
# - **small**: ~256M dimension, 4 layers (fastest)
# - **base**: ~512M dimension, 8 layers (balanced) ⭐ Default
# - **large**: ~1024M dimension, 12 layers (best performance)
# 
# ### Key Parameters
# - **seq_len**: Context length (how much history to use)
# - **pred_len**: Forecast horizon (how far to predict)
# - **model_size**: Model capacity

# %%
# Model configuration
seq_len = 512     # TimesFM supports up to 512 context length
pred_len = 96     # Forecast 96 steps ahead

# Initialize TimesFM
model = TimesFMTimeSeriesModel(
    seq_len=seq_len,
    pred_len=pred_len,
    model_size='base',   # 'small', 'base', or 'large'
    device=device
)

print(f"\nTimesFM Model Initialized")
print(f"Model size: base")
print(f"Context length: {seq_len} time steps")
print(f"Forecast horizon: {pred_len} time steps")
print(f"\n⭐ TimesFM is pre-trained - no training required!")

# %% [markdown]
# ## 🎓 "Train" the Model
# 
# **Note:** TimesFM is pre-trained, so this step just evaluates the model on your data.
# It doesn't actually train - it assesses zero-shot performance!

# %%
# Evaluate TimesFM (zero-shot)
print("Evaluating TimesFM on your data...")
print("(This is zero-shot evaluation - no training!)")

metrics = model.train(train_data, val_data, verbose=True)

print("\n" + "="*50)
print("Zero-Shot Evaluation Complete!")
print("="*50)
if 'val_rmse' in metrics:
    print(f"Validation RMSE: {metrics['val_rmse']:.4f}")
if 'val_mae' in metrics:
    print(f"Validation MAE: {metrics['val_mae']:.4f}")

# %% [markdown]
# ## 🔮 Make Zero-Shot Predictions

# %%
# Make predictions on validation data
print("Making zero-shot predictions...")
predictions = model.predict(val_data[:seq_len])

print(f"\nPredictions shape: {predictions.shape}")
print(f"First 10 predictions: {predictions[:10]}")

# %% [markdown]
# ## 📈 Visualize Zero-Shot Forecast

# %%
# Take a window from validation data
test_idx = 0
context = val_data[test_idx:test_idx + seq_len]

# Make forecast
forecast = model.predict(context)
if forecast.ndim > 1 and forecast.shape[0] == 1:
    forecast = forecast[0]

# Get actual future values
if test_idx + seq_len + pred_len <= len(val_data):
    actual_future = val_data[test_idx + seq_len:test_idx + seq_len + pred_len, 0]
else:
    actual_future = val_data[test_idx + seq_len:, 0]
    # Pad if necessary
    if len(actual_future) < pred_len:
        forecast = forecast[:len(actual_future)]

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Historical context
ax1.plot(range(len(context)), context[:, 0], label='Historical Context', linewidth=2, color='blue')
ax1.axvline(len(context) - 1, color='red', linestyle='--', alpha=0.5, label='Forecast Start')
ax1.set_title('Input Context for TimesFM', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time Step', fontsize=12)
ax1.set_ylabel('Value', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Zero-shot forecast vs actual
forecast_range = range(len(context), len(context) + len(actual_future))
ax2.plot(range(len(context)), context[:, 0], label='Historical Context', linewidth=2, alpha=0.3, color='blue')
ax2.plot(forecast_range, actual_future, 'g-', label='Actual Future', linewidth=2.5, marker='o', markersize=4)
ax2.plot(forecast_range, forecast[:len(actual_future)], 'r--', label='TimesFM Zero-Shot Forecast', linewidth=2.5, marker='s', markersize=4)
ax2.axvline(len(context) - 1, color='red', linestyle='--', alpha=0.5)
ax2.fill_between(forecast_range, actual_future, forecast[:len(actual_future)], alpha=0.2, color='orange')
ax2.set_title('TimesFM Zero-Shot Forecast vs Actual', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time Step', fontsize=12)
ax2.set_ylabel('Value', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate error
mae = np.mean(np.abs(actual_future - forecast[:len(actual_future)]))
rmse = np.sqrt(np.mean((actual_future - forecast[:len(actual_future)]) ** 2))
print(f"\nZero-Shot Forecast Metrics:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# %% [markdown]
# ## 🔬 Compare Different Context Lengths

# %%
# Test different context lengths
context_lengths = [64, 128, 256, 512]
results = {}

print("Testing different context lengths...")
for ctx_len in context_lengths:
    if ctx_len > len(val_data):
        continue
    
    # Create model with different context length
    temp_model = TimesFMTimeSeriesModel(
        seq_len=ctx_len,
        pred_len=pred_len,
        model_size='base',
        device=device
    )
    
    # Make prediction
    context = val_data[:ctx_len]
    forecast = temp_model.predict(context)
    
    # Calculate error
    actual = val_data[ctx_len:ctx_len + pred_len, 0]
    if len(actual) > 0:
        forecast_trimmed = forecast[:len(actual)] if forecast.ndim == 1 else forecast[0, :len(actual)]
        mae = np.mean(np.abs(actual - forecast_trimmed))
        results[ctx_len] = mae
        print(f"Context length {ctx_len:3d}: MAE = {mae:.4f}")

# Plot results
if results:
    plt.figure(figsize=(10, 6))
    plt.plot(list(results.keys()), list(results.values()), marker='o', linewidth=2, markersize=10)
    plt.xlabel('Context Length', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('Impact of Context Length on Forecast Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    best_ctx = min(results, key=results.get)
    print(f"\n✅ Best context length: {best_ctx} (MAE: {results[best_ctx]:.4f})")

# %% [markdown]
# ## 💾 Save Model Configuration

# %%
# Save model configuration
model_path = '../trained_models/timesfm_config.json'
os.makedirs('../trained_models', exist_ok=True)

model.save_model(model_path)
print(f"Model configuration saved to {model_path}")
print("\nNote: TimesFM is pre-trained, so we only save configuration.")
print("The actual model weights are loaded from the pre-trained checkpoint.")

# %% [markdown]
# ## 🔄 Load Model Configuration

# %%
# Load model
loaded_model = TimesFMTimeSeriesModel(
    seq_len=seq_len,
    pred_len=pred_len,
    model_size='base',
    device=device
)

loaded_model.load_model(model_path)
print("Model configuration loaded successfully!")

# Test loaded model
test_forecast = loaded_model.predict(val_data[:seq_len])
print(f"\nTest prediction from loaded model: {test_forecast[:5]}...")

# %% [markdown]
# ## 🆚 Compare TimesFM with Simple Baseline

# %%
# Simple baseline: Naive forecast (repeat last value)
context = val_data[:seq_len]
naive_forecast = np.full(pred_len, context[-1, 0])

# TimesFM forecast
timesfm_forecast = model.predict(context)
if timesfm_forecast.ndim > 1:
    timesfm_forecast = timesfm_forecast[0]

# Actual values
actual = val_data[seq_len:seq_len + pred_len, 0]
if len(actual) < pred_len:
    naive_forecast = naive_forecast[:len(actual)]
    timesfm_forecast = timesfm_forecast[:len(actual)]

# Calculate errors
naive_mae = np.mean(np.abs(actual - naive_forecast))
timesfm_mae = np.mean(np.abs(actual - timesfm_forecast))

# Plot comparison
plt.figure(figsize=(15, 6))
forecast_range = range(len(actual))

plt.plot(forecast_range, actual, 'g-', label='Actual', linewidth=2.5, marker='o', markersize=5)
plt.plot(forecast_range, naive_forecast, 'gray', linestyle=':', label=f'Naive (MAE: {naive_mae:.2f})', linewidth=2)
plt.plot(forecast_range, timesfm_forecast, 'r--', label=f'TimesFM (MAE: {timesfm_mae:.2f})', linewidth=2.5, marker='s', markersize=4)

plt.title('TimesFM vs Naive Baseline', fontsize=14, fontweight='bold')
plt.xlabel('Forecast Step', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

improvement = ((naive_mae - timesfm_mae) / naive_mae) * 100
print(f"\n📊 Comparison Results:")
print(f"Naive Baseline MAE: {naive_mae:.4f}")
print(f"TimesFM MAE: {timesfm_mae:.4f}")
print(f"\n✅ TimesFM improvement: {improvement:.1f}% better than naive baseline")

# %% [markdown]
# ## 🎯 Key Takeaways
# 
# ### TimesFM Advantages
# 1. **Zero-Shot**: No training required - works out of the box!
# 2. **Pre-trained**: Learned from 100B time points across diverse domains
# 3. **Fast**: Immediate predictions without waiting for training
# 4. **Versatile**: Works on various time series types
# 5. **Long Context**: Can use up to 512 historical points
# 
# ### When to Use TimesFM
# - ✅ **Limited data**: When you have few training samples
# - ✅ **Quick baseline**: Need fast results without training
# - ✅ **New domains**: No domain-specific historical data
# - ✅ **Transfer learning**: Leverage world knowledge
# - ✅ **Ensemble member**: Combine with domain-specific models
# 
# ### TimesFM vs Other Models
# 
# | Model | Training Needed | Best For |
# |-------|----------------|----------|
# | TimesFM | ❌ No (pre-trained) | Zero-shot, quick baseline |
# | Chronos-2 | ❌ No (pre-trained) | Similar to TimesFM |
# | PatchTST | ✅ Yes | Dataset-specific patterns |
# | iTransformer | ✅ Yes | Multivariate relationships |
# | XGBoost | ✅ Yes | Tabular features |
# 
# ### Tips for Best Results
# 1. **Context length**: Longer is often better (up to 512)
# 2. **Normalization**: TimesFM handles various scales well
# 3. **Ensemble**: Combine with task-specific models
# 4. **Model size**: Start with 'base', use 'large' for better accuracy
# 
# ### Next Steps
# 1. Try different model sizes (small/base/large)
# 2. Experiment with context lengths
# 3. Compare with other foundation models (Chronos-2)
# 4. Use in ensemble with domain-specific models
# 5. Evaluate on multiple datasets


