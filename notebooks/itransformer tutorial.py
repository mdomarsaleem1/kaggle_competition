
# %% [markdown]
# # iTransformer: Inverted Transformer for Time Series
# 
# ## 📚 Overview
# 
# **iTransformer** ("Inverted Transformers Are Effective for Time Series Forecasting") from ICLR 2024.
# 
# ### Revolutionary Innovation: Inverted Attention
# 
# **Traditional Transformers:**
# ```
# [Time1, Time2, Time3, ...] → Attention across TIME
# ```
# 
# **iTransformer:**
# ```
# [Var1, Var2, Var3, ...] → Attention across VARIATES
# ```
# 
# ### Key Advantages
# - **Variate-centric**: Each variable's full time series becomes a token
# - **Efficiency**: O(n_variates²) vs O(seq_len²)
# - **Multivariate**: Excels at capturing cross-variable dependencies
# - **Better generalization**: More robust to distribution shifts
# 
# ### When to Use iTransformer
# - ✅ **Multivariate data**: Multiple correlated time series
# - ✅ **Cross-variable relationships**: Features interact
# - ✅ **Long sequences**: Efficient on long time series
# - ✅ **Distribution shift**: Robust to changing patterns

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

from models import iTransformerTimeSeriesModel

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# %% [markdown]
# ## 📊 Create Multivariate Data
# 
# iTransformer shines with **multivariate** time series where variables are correlated.

# %%
# Load your data or create synthetic multivariate data
data_path = '../data/train.csv'

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Extract all numeric columns (multiple variates)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        data = df[numeric_cols].values
        print(f"\nUsing {len(numeric_cols)} variates: {numeric_cols}")
    else:
        # Create additional variates if only 1 column
        print("\nOnly 1 numeric column found, creating correlated variates...")
        target = df[numeric_cols[0]].values
        var2 = target * 1.2 + np.random.normal(0, 5, len(target))
        var3 = target * 0.8 + np.random.normal(0, 3, len(target))
        data = np.column_stack([target, var2, var3])
        print(f"Created 3 correlated variates")
else:
    print(f"Data file not found at {data_path}")
    print("Creating synthetic multivariate data...")
    
    # Create correlated multivariate time series
    n_points = 1000
    n_variates = 5
    
    # Base patterns
    t = np.arange(n_points)
    trend = np.linspace(100, 200, n_points)
    seasonality = 20 * np.sin(2 * np.pi * t / 365)
    
    # Create correlated variates
    data = []
    for i in range(n_variates):
        # Each variate is correlated but with different weights
        weight_trend = 0.8 + 0.4 * i / n_variates
        weight_season = 1.2 - 0.4 * i / n_variates
        noise = np.random.normal(0, 5, n_points)
        
        variate = weight_trend * trend + weight_season * seasonality + noise
        data.append(variate)
    
    data = np.column_stack(data)
    print(f"Created {n_variates} correlated variates")

print(f"\nFinal data shape: {data.shape} (samples, variates)")

# %% [markdown]
# ## 📈 Visualize Multivariate Data

# %%
n_variates = data.shape[1]

fig, axes = plt.subplots(n_variates, 1, figsize=(15, 3*n_variates))
if n_variates == 1:
    axes = [axes]

for i in range(n_variates):
    axes[i].plot(data[:, i], linewidth=1.5)
    axes[i].set_title(f'Variate {i+1}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Value')
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time Step')
plt.suptitle('Multivariate Time Series (All Variates)', fontsize=14, fontweight='bold', y=1.0)
plt.tight_layout()
plt.show()

# Show correlation between variates
plt.figure(figsize=(10, 8))
import seaborn as sns
correlation_matrix = np.corrcoef(data.T)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            xticklabels=[f'Var{i+1}' for i in range(n_variates)],
            yticklabels=[f'Var{i+1}' for i in range(n_variates)])
plt.title('Correlation Between Variates', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n📊 Correlation Analysis:")
print("High correlation means variates influence each other")
print("→ iTransformer can learn these cross-variate relationships!")

# %% [markdown]
# ## 🔄 Prepare Data for iTransformer

# %%
# Train/validation split
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
val_data = data[split_idx:]

print(f"Train data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")
print(f"\nNumber of variates: {data.shape[1]}")

# %% [markdown]
# ## 🏗️ Initialize iTransformer Model
# 
# ### Key Difference from Standard Transformer
# 
# **Standard Transformer:**
# - Input shape: [batch, seq_len, features]
# - Attention: Across time steps
# - Each time step is a token
# 
# **iTransformer:**
# - Input shape: [batch, features, seq_len] ← **Inverted!**
# - Attention: Across variates/features
# - Each variate's time series is a token
# 
# ### Hyperparameters
# - **seq_len**: Input window
# - **pred_len**: Forecast horizon  
# - **n_features**: Number of variates (variables)
# - **d_model**: Embedding dimension (larger = more capacity)
# - **n_heads**: Attention heads
# - **n_layers**: Transformer layers

# %%
# Model configuration
seq_len = 96       # Input window
pred_len = 24      # Forecast horizon
n_features = data.shape[1]

# Initialize iTransformer
model = iTransformerTimeSeriesModel(
    seq_len=seq_len,
    pred_len=pred_len,
    n_features=n_features,  # Each variate becomes a token!
    d_model=512,            # Embedding dimension (can be large since n_features usually small)
    n_heads=8,
    n_layers=2,             # 2-3 layers usually sufficient
    d_ff=2048,
    dropout=0.1,
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    device=device
)

print(f"\niTransformer Model Initialized")
print(f"Input: {seq_len} time steps × {n_features} variates")
print(f"Output: {pred_len} time steps × {n_features} variates")
print(f"\nAttention mechanism: Across {n_features} variates (not time!)")
print(f"Complexity: O({n_features}²) instead of O({seq_len}²)")

# %% [markdown]
# ## 🎓 Train the Model

# %%
# Train iTransformer
print("Training iTransformer...")
print("This learns cross-variate dependencies!\n")

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
    plt.title('iTransformer Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 🔮 Make Multivariate Predictions

# %%
# Make predictions
predictions = model.predict(val_data, return_sequences=False)

print(f"Predictions shape: {predictions.shape}")
print(f"(samples, features) = ({predictions.shape[0]}, {predictions.shape[1]})")
print(f"\nFirst prediction (all variates): {predictions[0]}")

# %% [markdown]
# ## 📈 Visualize Multivariate Forecasts

# %%
# Take a specific window
test_idx = 0
context = val_data[test_idx:test_idx + seq_len]

# Make forecast for all variates
forecast = model.predict(context.reshape(1, seq_len, n_features))
if forecast.ndim == 3:
    forecast = forecast[0]  # Remove batch dimension

# Get actual future values
actual_future = val_data[test_idx + seq_len:test_idx + seq_len + pred_len]
if len(actual_future) < pred_len:
    forecast = forecast[:len(actual_future)]

# Plot each variate
fig, axes = plt.subplots(n_features, 1, figsize=(15, 4*n_features))
if n_features == 1:
    axes = [axes]

for i in range(n_features):
    # Historical context
    axes[i].plot(range(seq_len), context[:, i], 
                 label='Historical', linewidth=2, alpha=0.5, color='blue')
    
    # Forecast vs actual
    forecast_range = range(seq_len, seq_len + len(actual_future))
    axes[i].plot(forecast_range, actual_future[:, i], 'g-',
                 label='Actual', linewidth=2.5, marker='o', markersize=5)
    axes[i].plot(forecast_range, forecast[:, i], 'r--',
                 label='Forecast', linewidth=2.5, marker='s', markersize=4)
    
    axes[i].axvline(seq_len - 1, color='black', linestyle='--', alpha=0.3)
    axes[i].set_title(f'Variate {i+1} - Forecast', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Value')
    axes[i].legend(fontsize=10)
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time Step')
plt.suptitle('iTransformer: Multivariate Forecasts', fontsize=14, fontweight='bold', y=1.0)
plt.tight_layout()
plt.show()

# Calculate error per variate
print("\n📊 Per-Variate Forecast Accuracy:")
for i in range(n_features):
    mae = np.mean(np.abs(actual_future[:, i] - forecast[:, i]))
    print(f"Variate {i+1} MAE: {mae:.4f}")

# %% [markdown]
# ## 🔬 Analyze Cross-Variate Attention
# 
# One of iTransformer's superpowers: learning which variates influence each other!

# %%
# Make predictions for multiple samples
n_samples = min(100, len(val_data) - seq_len - pred_len)
errors_per_variate = []

for i in range(n_samples):
    context = val_data[i:i + seq_len]
    forecast = model.predict(context.reshape(1, seq_len, n_features))
    if forecast.ndim == 3:
        forecast = forecast[0]
    
    actual = val_data[i + seq_len:i + seq_len + pred_len]
    if len(actual) == pred_len:
        errors = np.abs(actual - forecast)
        errors_per_variate.append(errors.mean(axis=0))

errors_per_variate = np.array(errors_per_variate)

# Visualize average error per variate
plt.figure(figsize=(12, 6))
mean_errors = errors_per_variate.mean(axis=0)
std_errors = errors_per_variate.std(axis=0)

x = np.arange(n_features)
plt.bar(x, mean_errors, yerr=std_errors, capsize=5, alpha=0.7, color='steelblue')
plt.xlabel('Variate', fontsize=12)
plt.ylabel('Mean Absolute Error', fontsize=12)
plt.title('iTransformer Performance per Variate', fontsize=14, fontweight='bold')
plt.xticks(x, [f'Var {i+1}' for i in range(n_features)])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print("\n💡 Interpretation:")
print("Lower error = better prediction for that variate")
print("→ Some variates may be easier to predict due to stronger cross-variate relationships")

# %% [markdown]
# ## 💾 Save the Model

# %%
# Save model
model_path = '../trained_models/itransformer_model.pth'
os.makedirs('../trained_models', exist_ok=True)

model.save_model(model_path)
print(f"Model saved to {model_path}")

# %% [markdown]
# ## 🎯 Key Takeaways
# 
# ### iTransformer Innovation
# 1. **Inverted Attention**: Attends across variates, not time
# 2. **Variate-Centric**: Each variable's time series is a token
# 3. **Efficient**: O(n_variates²) complexity
# 4. **Cross-Dependencies**: Learns how variables influence each other
# 
# ### iTransformer vs PatchTST vs Standard Transformer
# 
# | Model | Attention | Best For | Complexity |
# |-------|-----------|----------|------------|
# | Standard Transformer | Time steps | Univariate | O(seq_len²) |
# | PatchTST | Patches | Long sequences | O(n_patches²) |
# | iTransformer | Variates | Multivariate | O(n_variates²) |
# 
# ### When to Use iTransformer
# - ✅ **Multivariate data**: 2+ correlated time series
# - ✅ **Cross-variable relationships**: Variables influence each other
# - ✅ **Many variates**: Efficient even with 50+ variables
# - ✅ **Distribution shift**: More robust than standard transformers
# - ✅ **Feature-rich**: When you have many related measurements
# 
# ### When NOT to Use iTransformer
# - ❌ **Univariate data**: Use PatchTST or TimesNet instead
# - ❌ **Independent variates**: If variables don't correlate
# - ❌ **Very few variates** (1-2): Overhead not worth it
# 
# ### Hyperparameter Tips
# 1. **d_model**: Can be large (512-1024) since n_features usually small
# 2. **n_layers**: 2-3 layers sufficient (more may overfit)
# 3. **n_heads**: 8 heads works well
# 4. **Learning rate**: 0.0001-0.001
# 
# ### Real-World Applications
# - **Finance**: Multiple stock prices, economic indicators
# - **Energy**: Multiple sensor readings from power grid
# - **Weather**: Temperature, humidity, pressure across locations
# - **Healthcare**: Multiple vital signs (heart rate, BP, temp)
# - **IoT**: Multiple sensor measurements
# 
# ### Next Steps
# 1. Try with your multivariate data
# 2. Compare with univariate models (PatchTST)
# 3. Analyze which variates are most predictable
# 4. Combine with other models in ensemble
# 5. Experiment with different numbers of variates


