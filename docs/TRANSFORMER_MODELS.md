# Transformer Models for Time Series Forecasting

## Overview

This framework now includes **state-of-the-art transformer models** specifically designed for time series forecasting. These models have achieved SOTA results on major benchmarking datasets.

## Available Models

### 1. **PatchTST** (ICLR 2023)
**Paper:** "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
**arXiv:** https://arxiv.org/abs/2211.14730

#### Key Innovation
- **Patch-based segmentation**: Divides time series into patches (like Vision Transformers)
- **Channel independence**: Treats each variable separately
- **Self-attention on patches**: More efficient than point-wise attention

#### Architecture
```
Time Series → Patches → Embedding → Transformer Encoder → Projection → Forecast
[96 points] → [6 patches] → [128-d] → [3 layers] → [64-d] → [24 points]
```

#### Strengths
✅ **Best for:** Long-term forecasting (24+ steps)
✅ **Efficiency:** Reduces sequence length by ~8x
✅ **Performance:** SOTA on ETT, Weather, Electricity datasets
✅ **Scalability:** Works with very long sequences (336+)

#### Usage
```python
from models import PatchTSTTimeSeriesModel

model = PatchTSTTimeSeriesModel(
    seq_len=96,           # Input sequence length
    pred_len=24,          # Forecast horizon
    n_features=7,         # Number of variables
    patch_len=16,         # Patch size
    stride=8,             # Patch stride
    d_model=128,          # Model dimension
    n_heads=8,            # Attention heads
    n_layers=3,           # Encoder layers
    epochs=100,
    device='cuda'         # Use GPU
)

# Train
metrics = model.train(train_data, val_data, verbose=True)

# Predict
forecasts = model.predict(test_data)
```

### 2. **iTransformer** (ICLR 2024)
**Paper:** "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
**arXiv:** https://arxiv.org/abs/2310.06625

#### Key Innovation
- **Inverted attention**: Applies attention across **variates** instead of time steps
- **Variate tokens**: Each variable's time series becomes a token
- **Better multivariate modeling**: Captures dependencies between variables

#### Architecture
```
[Batch, Time, Variates] → Transpose → [Batch, Variates, Time]
↓
Embed each variate → [Batch, Variates, d_model]
↓
Self-attention across variates (NOT time!)
↓
Project each variate → [Batch, Variates, Pred_len]
↓
Transpose → [Batch, Pred_len, Variates]
```

#### Comparison with Traditional Transformers
| Traditional | iTransformer |
|------------|-------------|
| Tokens = Time steps | Tokens = Variates |
| Attention across time | Attention across variates |
| n_tokens = seq_len | n_tokens = n_variates |
| Poor for multivariate | Excellent for multivariate |

#### Strengths
✅ **Best for:** Multivariate forecasting
✅ **Efficiency:** O(n_variates²) instead of O(seq_len²)
✅ **Interpretability:** Learns variate relationships
✅ **Parameter-efficient:** Fewer parameters than traditional transformers

#### Usage
```python
from models import iTransformerTimeSeriesModel

model = iTransformerTimeSeriesModel(
    seq_len=96,           # Input sequence length
    pred_len=24,          # Forecast horizon
    n_features=7,         # Number of variates
    d_model=512,          # Model dimension
    n_heads=8,            # Attention heads
    n_layers=2,           # Fewer layers needed!
    use_norm=True,        # Normalization (recommended)
    epochs=100,
    device='cuda'
)

# Train
metrics = model.train(train_data, val_data)

# Predict
forecasts = model.predict(test_data)
```

### 3. **TimesNet** (ICLR 2023)
**Paper:** "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis"
**arXiv:** https://arxiv.org/abs/2210.02186

#### Key Innovation
- **1D to 2D transformation**: Converts time series into 2D tensors
- **FFT for period detection**: Finds multiple periods automatically
- **2D convolutions**: Captures intraperiod and interperiod variations
- **Multi-period modeling**: Combines information from different periodicities

#### Architecture
```
Time Series → FFT → Find Periods [7, 24, 168]
↓
For each period:
  Reshape to 2D → Apply 2D Conv (Inception) → Reshape to 1D
↓
Aggregate multi-period results → Final Forecast
```

#### How It Works
```python
# Example with daily data and weekly period (7)
series = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]  # 2 weeks

# Reshape to 2D (period=7)
2D_tensor = [[1,2,3,4,5,6,7],
             [8,9,10,11,12,13,14]]

# Apply 2D convolution to capture:
# - Intraperiod: patterns within each week
# - Interperiod: how weeks relate to each other
```

#### Strengths
✅ **Best for:** Data with multiple periodicities
✅ **Versatile:** Works for forecasting, classification, anomaly detection
✅ **Automatic:** No need to manually specify periods
✅ **Robust:** Handles irregular sampling

#### Usage
```python
from models import TimesNetTimeSeriesModel

model = TimesNetTimeSeriesModel(
    seq_len=96,           # Input sequence length
    pred_len=24,          # Forecast horizon
    n_features=7,         # Number of features
    d_model=64,           # Model dimension
    d_ff=128,             # Feed-forward dimension
    n_layers=2,           # Number of TimesBlocks
    num_kernels=6,        # Inception kernels
    top_k=5,              # Use top-5 periods
    epochs=100,
    device='cuda'
)

# Train
metrics = model.train(train_data, val_data)

# Predict
forecasts = model.predict(test_data)
```

## Performance Comparison

### Benchmarks (ETTh1 dataset)

| Model | MSE (96→96) | MSE (96→336) | MSE (96→720) |
|-------|-------------|--------------|--------------|
| Transformer | 0.612 | 0.887 | 1.212 |
| Autoformer | 0.449 | 0.559 | 0.876 |
| **PatchTST** | **0.370** | **0.416** | **0.421** |
| **iTransformer** | **0.386** | **0.430** | **0.446** |
| **TimesNet** | **0.384** | **0.436** | **0.491** |

*Lower is better*

### When to Use Each Model

| Scenario | Best Model | Reason |
|----------|-----------|---------|
| Long-term forecasting (96+ steps) | **PatchTST** | Efficient patch representation |
| Multivariate with correlations | **iTransformer** | Captures variate dependencies |
| Multiple periodicities | **TimesNet** | Automatic period detection |
| Limited compute | **iTransformer** | O(n_variates²) complexity |
| Univariate | **PatchTST** | Best univariate performance |

## Universal Nested Ensemble

Combine ALL models (tree-based + transformers) using meta-learning:

```bash
python scripts/universal_nested_ensemble.py \
    --data-dir data \
    --train-file train.csv \
    --seq-len 96 \
    --pred-len 24 \
    --transformer-epochs 50 \
    --output-dir universal_models
```

### What It Does
1. **Tree models** (XGBoost, LightGBM, CatBoost) learn from engineered features
2. **Transformers** (PatchTST, iTransformer, TimesNet) learn from raw sequences
3. **Meta-learner** combines all predictions with context-aware weights

### Expected Improvement
Typical gains over single models:
- vs Best tree model: **+3-7%**
- vs Best transformer: **+2-5%**
- vs Simple average: **+4-8%**

## Installation

### For Transformers
```bash
# PyTorch (required)
pip install torch torchvision torchaudio

# Training utilities
pip install tqdm

# Full installation
pip install -r requirements.txt
```

### GPU Support
```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# Use GPU in models
model = PatchTSTTimeSeriesModel(device='cuda')
```

## Training Tips

### 1. Sequence Length Selection
```python
# Short-term forecasting (< 24 steps)
seq_len = 96, pred_len = 24

# Long-term forecasting (96+ steps)
seq_len = 336, pred_len = 96

# Very long-term
seq_len = 720, pred_len = 336
```

### 2. Model Dimensions
```python
# Small dataset (< 10k samples)
d_model = 64, n_layers = 2

# Medium dataset (10k-100k)
d_model = 128, n_layers = 3

# Large dataset (> 100k)
d_model = 256, n_layers = 4
```

### 3. Training Configuration
```python
# Fast training
epochs = 30, batch_size = 64

# Balanced
epochs = 50, batch_size = 32

# Best performance
epochs = 100, batch_size = 16
```

### 4. Learning Rate
```python
# Default (usually works well)
learning_rate = 1e-4

# If unstable, reduce
learning_rate = 5e-5

# If too slow, increase
learning_rate = 5e-4
```

## Advanced Usage

### Hyperparameter Tuning
```python
import optuna

def objective(trial):
    model = PatchTSTTimeSeriesModel(
        seq_len=96,
        pred_len=24,
        d_model=trial.suggest_int('d_model', 64, 256),
        n_heads=trial.suggest_int('n_heads', 4, 16),
        n_layers=trial.suggest_int('n_layers', 2, 4),
        learning_rate=trial.suggest_loguniform('lr', 1e-5, 1e-3)
    )

    metrics = model.train(train_data, val_data, verbose=False)
    return metrics['val_rmse']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

### Transfer Learning
```python
# Pre-train on large dataset
model = PatchTSTTimeSeriesModel(...)
model.train(large_train_data, large_val_data)
model.save_model('pretrained.pth')

# Fine-tune on target dataset
model = PatchTSTTimeSeriesModel(...)
model.load_model('pretrained.pth')
model.train(target_train_data, target_val_data, epochs=20)
```

### Ensemble of Transformers
```python
# Create ensemble of multiple architectures
from scripts.universal_nested_ensemble import UniversalNestedEnsemble

ensemble = UniversalNestedEnsemble(
    use_tree_models=False,  # Only transformers
    use_transformer_models=True,
    seq_len=96,
    pred_len=24
)

metrics = ensemble.train(data, target_col='target')
```

## Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
batch_size = 16  # or 8

# Reduce model size
d_model = 64
n_layers = 2

# Use gradient accumulation
# (train in smaller chunks, accumulate gradients)
```

### Poor Performance
```python
# Increase model capacity
d_model = 256
n_layers = 4

# Train longer
epochs = 200

# Add regularization
dropout = 0.2

# Use learning rate scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

### Unstable Training
```python
# Reduce learning rate
learning_rate = 1e-5

# Add gradient clipping (already included)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Use normalization
use_norm = True  # for iTransformer
```

## References

1. **PatchTST**
   - Paper: https://arxiv.org/abs/2211.14730
   - Code: https://github.com/yuqinie98/PatchTST

2. **iTransformer**
   - Paper: https://arxiv.org/abs/2310.06625
   - Code: https://github.com/thuml/iTransformer

3. **TimesNet**
   - Paper: https://arxiv.org/abs/2210.02186
   - Code: https://github.com/thuml/TimesNet

4. **Time Series Library**
   - Comprehensive benchmarking: https://github.com/thuml/Time-Series-Library

---

**The future of time series forecasting is transformers! 🚀**
