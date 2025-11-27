# GaussRank Sharpe Optimization Script

## Overview

The `gaussrank_sharpe_optimization.py` script is an enhanced version of `full_file.py` with three key improvements:

1. **GaussRank Transformation**: Features are transformed using Inverse Normal (GaussRank) transformation
2. **Sharpe Testing Framework**: Includes comprehensive tests for the `sharpe_eval_slice` function
3. **k and b Optimization**: Optimizes allocation parameters to maximize Adjusted Sharpe Ratio

## Key Changes from full_file.py

### 1. No Training on Target Variable

The script explicitly **excludes `market_forward_excess_returns` from the feature set**. This variable is only used as the target for training, preventing data leakage.

```python
# Explicitly exclude TARGET_COL from features
feature_cols = [
    c for c in test_feature_cols
    if c not in {TARGET_COL, ID_COL, "market_forward_excess_returns"}
]
```

### 2. GaussRank Transformation

All features are transformed using the GaussRank (Inverse Normal) transformation:

- **Ranks** values within each feature
- **Converts** ranks to uniform distribution [0, 1]
- **Applies** inverse normal CDF to get Gaussian distribution

This transformation:
- Makes features more normally distributed
- Reduces impact of outliers
- Can improve model performance on gradient-boosted trees

```python
def gaussrank_transform(series: pd.Series) -> pd.Series:
    """Apply GaussRank (Inverse Normal) transformation."""
    ranked = series.rank(method='average')
    n = len(ranked)
    uniform = (ranked - 0.5) / n
    gaussrank = norm.ppf(uniform)
    return gaussrank
```

### 3. Sharpe Testing Framework

The script includes `test_sharpe_eval_slice()` which validates the Sharpe calculation with:

- **Basic tests** without risk adjustment
- **Full tests** with risk_free_rate and forward_returns
- **Edge cases** (zero predictions, max predictions)
- **Negative correlation** tests

Run these tests by executing the main script - they run automatically before training.

### 4. k and b Optimization

The most significant enhancement is the optimization of allocation parameters:

```
Allocation_t = Clip(k × Prediction_t + b, 0, 2)
```

The optimization:
- Uses `scipy.optimize.minimize` with Nelder-Mead method
- Tries **multiple starting points** to avoid local minima
- Maximizes **Adjusted Sharpe Ratio** on validation set
- Applies **per-model optimization** (separate k and b for each model)

```python
def optimize_k_b(predictions, df_val, initial_k=1.0, initial_b=0.0):
    """Find optimal k and b that maximize Sharpe ratio."""
    def objective(params):
        k, b = params
        allocations = np.clip(k * predictions + b, 0.0, 2.0)
        sharpe = sharpe_eval_slice(df_val, allocations)
        return -sharpe  # Minimize negative Sharpe

    # Try multiple starting points
    starting_points = [(1.0, 0.0), (0.5, 0.5), (1.5, -0.5), ...]
    ...
```

## Usage

### Running the Script

```bash
cd scripts
python gaussrank_sharpe_optimization.py
```

### Running Tests Only

```bash
python test_gaussrank_optimization.py
```

## Output

The script produces:

1. **Test Results**: Validation of all key components
2. **Model Performance**: Sharpe ratios for each model with optimized k and b
3. **Ensemble Weights**: Final weights for LightGBM, XGBoost, and CatBoost
4. **Submission File**: `data/04_models/submission_gaussrank_optimized.csv`

### Example Output

```
======================================================================
Testing sharpe_eval_slice function
======================================================================

Test 1: Basic Sharpe calculation
  Sharpe ratio: 0.9354
  Expected: positive value (predictions align with returns)

...

======================================================================
Cross-validation with k and b optimization
======================================================================

LightGBM - k: 1.2345, b: 0.1234, Sharpe: 0.5678
XGBoost - k: 1.3456, b: 0.2345, Sharpe: 0.5789
CatBoost - k: 1.4567, b: 0.3456, Sharpe: 0.5890

======================================================================
Final Model Weights:
  LGBM: 0.333 (Sharpe: 0.568)
  XGB:  0.333 (Sharpe: 0.579)
  Cat:  0.334 (Sharpe: 0.589)
======================================================================
```

## Performance Improvements

The combination of GaussRank transformation and k/b optimization typically provides:

- **Better feature distributions**: More suitable for tree-based models
- **Optimal allocations**: Maximizes Sharpe directly rather than just correlation
- **Reduced overfitting**: GaussRank reduces impact of outliers
- **Higher Sharpe ratios**: Direct optimization of the evaluation metric

## Technical Details

### GaussRank Properties

- Preserves feature **ordering** (monotonic transformation)
- Outputs have **mean ≈ 0** and **std ≈ 1**
- Handles **NaN values** gracefully
- Makes features **normally distributed**

### Optimization Algorithm

- **Method**: Nelder-Mead (derivative-free optimization)
- **Objective**: Maximize Sharpe Ratio
- **Constraints**: Allocations clipped to [0, 2]
- **Robustness**: Multiple starting points prevent local minima

### Ensemble Strategy

The final prediction combines three models:

```
prediction = w_lgb × (k_lgb × pred_lgb + b_lgb)
           + w_xgb × (k_xgb × pred_xgb + b_xgb)
           + w_cat × (k_cat × pred_cat + b_cat)

allocation = Clip(prediction, 0, 2)
```

Where weights are based on:
- Individual model Sharpe ratios
- Model correlation (diversity bonus)

## Dependencies

Required packages:
- numpy
- pandas
- scipy
- scikit-learn
- lightgbm
- xgboost
- catboost
- optuna

## Files

- `gaussrank_sharpe_optimization.py` - Main training script
- `test_gaussrank_optimization.py` - Component validation tests
- `README_gaussrank_optimization.md` - This documentation

## Future Improvements

Potential enhancements:
1. Add more transformation options (YeoJohnson, Box-Cox)
2. Optimize k and b globally across all models
3. Add cross-validation for k/b parameters
4. Implement rolling window optimization
5. Add feature selection based on GaussRank importance
