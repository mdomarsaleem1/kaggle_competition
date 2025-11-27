
# %% [markdown]
# # TimesNet: 2D Temporal Variation Modeling
# 
# ## 📚 Overview
# 
# **TimesNet** ("Temporal 2D-Variation Modeling for General Time Series Analysis") from ICLR 2023.
# 
# ### Revolutionary Innovation: 1D→2D Transformation
# 
# **Key Idea**: Transform 1D time series into 2D tensors to capture complex temporal patterns!
# 
# ```
# 1D Time Series [365 days]
#         ↓ (using FFT to find periods)
# 2D Tensor [52 weeks × 7 days]
#         ↓ (2D convolutions)
# Capture intraperiod + interperiod variations
# ```
# 
# ### Key Features
# - **Automatic period detection** using FFT
# - **2D convolutions** capture both intra-period and inter-period variations
# - **Multi-periodicity**: Handles multiple seasonal patterns simultaneously
# - **SOTA performance** on 5 mainstream time series tasks
# 
# ### When to Use TimesNet
# - ✅ **Multiple periodicities**: Daily, weekly, monthly patterns
# - ✅ **Complex seasonality**: Nested seasonal patterns
# - ✅ **Long-range dependencies**: Patterns across different time scales
# - ✅ **Real-world data**: Often has multiple periods

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
from scipy import fft

from models import TimesNetTimeSeriesModel

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# %% [markdown]
# ## 📊 Create Data with Multiple Periodicities

# %%
# Load your data or create synthetic data with multiple periods
data_path = '../data/train.csv'

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(df.head())
    
    # Extract target
    target_col = [col for col in df.columns if col != 'date'][0]
    data = df[target_col].values
else:
    print(f"Data file not found at {data_path}")
    print("Creating synthetic data with MULTIPLE periodicities...")
    
    # Create data with multiple seasonal patterns
    n_points = 1000
    t = np.arange(n_points)
    
    # Multiple periodic components
    trend = np.linspace(100, 200, n_points)
    yearly = 30 * np.sin(2 * np.pi * t / 365)        # Yearly seasonality
    monthly = 15 * np.sin(2 * np.pi * t / 30)        # Monthly seasonality
    weekly = 8 * np.sin(2 * np.pi * t / 7)           # Weekly seasonality
    daily = 3 * np.sin(2 * np.pi * t / 1)            # Daily variation
    noise = np.random.normal(0, 5, n_points)
    
    data = trend + yearly + monthly + weekly + daily + noise
    
    print(f"\nCreated data with:")
    print(f"  - Yearly seasonality (period=365)")
    print(f"  - Monthly seasonality (period=30)")
    print(f"  - Weekly seasonality (period=7)")
    print(f"  - Daily variation (period=1)")

# Reshape for model
if data.ndim == 1:
    data = data.reshape(-1, 1)

print(f"\nFinal data shape: {data.shape}")

# %% [markdown]
# ## 📈 Visualize Data and Detect Periods with FFT

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Time series
ax1.plot(data[:, 0], linewidth=1.5)
ax1.set_title('Time Series with Multiple Periodicities', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Value')
ax1.grid(True, alpha=0.3)

# FFT to detect periods (like TimesNet does!)
signal = data[:, 0]
fft_vals = fft.rfft(signal)
power = np.abs(fft_vals) ** 2
freqs = fft.rfftfreq(len(signal))

# Find top periods
top_k = 5
top_indices = np.argsort(power)[-top_k-1:-1][::-1]  # Skip DC component
top_freqs = freqs[top_indices]
top_periods = 1 / (top_freqs + 1e-8)

# Plot power spectrum
ax2.plot(freqs[1:len(power)//2], power[1:len(power)//2], linewidth=2)
ax2.scatter(top_freqs, power[top_indices], color='red', s=100, zorder=5, 
            label=f'Top {top_k} Frequencies')
ax2.set_title('Frequency Domain (FFT Power Spectrum)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Power')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n🔍 Detected Top {top_k} Periods (via FFT):")
for i, (freq, period) in enumerate(zip(top_freqs, top_periods)):
    if period < len(signal):
        print(f"  {i+1}. Period: {period:.1f} time steps (freq: {freq:.4f})")

print("\n💡 TimesNet will use these periods to create 2D tensors!")

# %% [markdown]
# ## 🔄 Prepare Data

# %%
# Train/validation split
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
val_data = data[split_idx:]

print(f"Train data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")

# %% [markdown]
# ## 🏗️ Initialize TimesNet Model
# 
# ### How TimesNet Works
# 
# 1. **FFT Analysis**: Find dominant periods in the data
# 2. **1D→2D Transform**: Reshape time series based on detected periods
#    ```
#    [365 steps] → [52 × 7] for weekly pattern
#    [365 steps] → [12 × 30] for monthly pattern
#    ```
# 3. **2D Convolutions**: Capture both:
#    - **Intra-period**: Variations within a period (e.g., Mon vs Fri)
#    - **Inter-period**: Variations across periods (e.g., Week 1 vs Week 52)
# 4. **2D→1D Transform**: Back to time series
# 
# ### Hyperparameters
# - **top_k**: Number of top periods to use (default: 5)
# - **d_model**: Model dimension
# - **n_layers**: Number of TimesBlock layers (each handles multi-periodicity)

# %%
# Model configuration
seq_len = 96       # Input window
pred_len = 24      # Forecast horizon
n_features = data.shape[1]

# Initialize TimesNet
model = TimesNetTimeSeriesModel(
    seq_len=seq_len,
    pred_len=pred_len,
    n_features=n_features,
    d_model=64,          # Model dimension (smaller is often fine)
    n_layers=2,          # Number of TimesBlocks
    top_k=5,             # Use top 5 detected periods
    d_ff=256,
    dropout=0.1,
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    device=device
)

print(f"\nTimesNet Model Initialized")
print(f"Input: {seq_len} time steps")
print(f"Output: {pred_len} time steps")
print(f"\nTimesNet will:")
print(f"  1. Use FFT to find top {5} periods")
print(f"  2. Transform 1D → 2D for each period")
print(f"  3. Apply 2D convolutions")
print(f"  4. Transform back 2D → 1D")

# %% [markdown]
# ## 🎓 Train the Model

# %%
# Train TimesNet
print("Training TimesNet...")
print("Learning multi-periodic patterns!\n")

metrics = model.train(train_data, val_data, verbose=True)

print("\n" + "="*50)
print("Training Complete!")
print("="*50)
if 'val_rmse' in metrics:
    print(f"Validation RMSE: {metrics['val_rmse']:.4f}")
if 'val_mae' in metrics:
    print(f"Validation MAE: {metrics['val_mae']:.4f}")

# %% [markdown]
# ## 📊 Visualize Training History

# %%
if 'train_losses' in metrics and 'val_losses' in metrics:
    plt.figure(figsize=(12, 5))
    
    plt.plot(metrics['train_losses'], label='Training Loss', linewidth=2)
    plt.plot(metrics['val_losses'], label='Validation Loss', linewidth=2)
    
    best_epoch = np.argmin(metrics['val_losses'])
    plt.axvline(best_epoch, color='r', linestyle='--', alpha=0.5,
                label=f'Best Epoch: {best_epoch}')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('TimesNet Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 🔮 Make Predictions

# %%
# Make predictions
predictions = model.predict(val_data, return_sequences=False)

print(f"Predictions shape: {predictions.shape}")
print(f"First 10 predictions: {predictions[:10, 0]}")

# %% [markdown]
# ## 📈 Visualize Forecasts

# %%
# Detailed forecast example
test_idx = 0
context = val_data[test_idx:test_idx + seq_len]

# Make forecast
forecast = model.predict(context.reshape(1, seq_len, n_features))
if forecast.ndim > 2:
    forecast = forecast[0]
if forecast.ndim > 1:
    forecast = forecast[:, 0]

# Get actual future
actual_future = val_data[test_idx + seq_len:test_idx + seq_len + pred_len, 0]
if len(actual_future) < pred_len:
    forecast = forecast[:len(actual_future)]

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Historical context
ax1.plot(range(seq_len), context[:, 0], label='Historical Context', linewidth=2, color='blue')
ax1.axvline(seq_len - 1, color='red', linestyle='--', alpha=0.5, label='Forecast Start')
ax1.set_title('Input Context', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Value')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Forecast vs actual
forecast_range = range(seq_len, seq_len + len(actual_future))
ax2.plot(range(seq_len), context[:, 0], label='Historical', linewidth=2, alpha=0.3, color='blue')
ax2.plot(forecast_range, actual_future, 'g-', label='Actual Future', 
         linewidth=2.5, marker='o', markersize=5)
ax2.plot(forecast_range, forecast, 'r--', label='TimesNet Forecast',
         linewidth=2.5, marker='s', markersize=4)
ax2.axvline(seq_len - 1, color='red', linestyle='--', alpha=0.5)
ax2.fill_between(forecast_range, actual_future, forecast, alpha=0.2, color='orange')
ax2.set_title('TimesNet Forecast (Multi-Periodic Modeling)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Value')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate error
mae = np.mean(np.abs(actual_future - forecast))
rmse = np.sqrt(np.mean((actual_future - forecast) ** 2))
print(f"\nForecast Metrics:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# %% [markdown]
# ## 🔬 Analyze How TimesNet Captures Different Periods

# %%
# Demonstrate 1D→2D transformation for different periods
sample_sequence = context[:, 0]

# Simulate what TimesNet does internally
periods_to_show = [7, 12, 24]  # Example periods

fig, axes = plt.subplots(1, len(periods_to_show), figsize=(15, 4))

for idx, period in enumerate(periods_to_show):
    # Reshape to 2D based on period
    seq_len_available = len(sample_sequence)
    n_periods = seq_len_available // period
    
    if n_periods > 1:
        # Trim to fit period exactly
        trimmed = sample_sequence[:n_periods * period]
        reshaped_2d = trimmed.reshape(n_periods, period)
        
        # Visualize as heatmap
        im = axes[idx].imshow(reshaped_2d, aspect='auto', cmap='viridis')
        axes[idx].set_title(f'Period={period}\n({n_periods}×{period})', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Intra-period')
        axes[idx].set_ylabel('Inter-period')
        plt.colorbar(im, ax=axes[idx])

plt.suptitle('TimesNet 1D→2D Transformation for Different Periods', 
             fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.show()

print("\n💡 Interpretation:")
print("  - Each column: Intra-period variation (e.g., Mon vs Tue)")
print("  - Each row: Inter-period variation (e.g., Week 1 vs Week 2)")
print("  - 2D convolutions capture both patterns simultaneously!")

# %% [markdown]
# ## 💾 Save the Model

# %%
# Save model
model_path = '../trained_models/timesnet_model.pth'
os.makedirs('../trained_models', exist_ok=True)

model.save_model(model_path)
print(f"Model saved to {model_path}")

# %% [markdown]
# ## 🎯 Key Takeaways
# 
# ### TimesNet Innovation
# 1. **1D→2D Transform**: Converts time series to 2D tensors based on detected periods
# 2. **FFT-based period detection**: Automatically finds dominant periodicities
# 3. **2D Convolutions**: Captures both intra-period and inter-period variations
# 4. **Multi-periodicity**: Handles multiple seasonal patterns simultaneously
# 
# ### TimesNet vs Other Transformers
# 
# | Model | Key Innovation | Best For |
# |-------|---------------|----------|
# | Standard Transformer | Self-attention | General sequences |
# | PatchTST | Patches | Long-term forecasting |
# | iTransformer | Inverted attention | Multivariate |
# | TimesNet | 1D→2D transform | Multi-periodic data |
# 
# ### When to Use TimesNet
# - ✅ **Multiple periodicities**: Daily, weekly, monthly, yearly patterns
# - ✅ **Complex seasonality**: Nested seasonal patterns
# - ✅ **Real-world data**: Often has multiple time scales
# - ✅ **Unknown periods**: FFT automatically detects them
# - ✅ **Long-range patterns**: Captures dependencies across periods
# 
# ### When NOT to Use TimesNet
# - ❌ **No periodicity**: Random walk, pure noise
# - ❌ **Single simple period**: Overkill for simple seasonality
# - ❌ **Very short sequences**: Need enough data to detect periods
# 
# ### Real-World Applications
# - **Retail**: Daily sales with weekly and monthly patterns
# - **Energy**: Hourly demand with daily/weekly/seasonal cycles
# - **Web Traffic**: Hourly visits with daily/weekly patterns
# - **Weather**: Temperature with daily/seasonal cycles
# - **Finance**: Trading volume with intraday/weekly patterns
# 
# ### Hyperparameter Tips
# 1. **top_k**: 5 usually good (captures main periods)
# 2. **d_model**: 32-64 often sufficient
# 3. **n_layers**: 2-3 TimesBlocks
# 4. **seq_len**: Longer = better period detection
# 
# ### Comparison with Traditional Methods
# 
# **Seasonal ARIMA:**
# - ❌ Manual period specification
# - ❌ Single periodicity
# - ❌ Linear assumptions
# 
# **TimesNet:**
# - ✅ Automatic period detection
# - ✅ Multiple periodicities
# - ✅ Non-linear modeling
# 
# ### Next Steps
# 1. Try with your periodic data
# 2. Analyze detected periods (are they meaningful?)
# 3. Compare with PatchTST and iTransformer
# 4. Combine in ensemble for best results
# 5. Experiment with different top_k values


