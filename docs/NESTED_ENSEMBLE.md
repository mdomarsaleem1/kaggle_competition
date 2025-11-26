# Nested Ensemble (Stacking) with Meta-Learning

## Overview

The nested ensemble (also called **stacking**) is a powerful ensemble technique that goes beyond simple averaging. Instead of using fixed weights, it trains a **meta-learner** that dynamically adjusts the combination of base model predictions based on the input context.

## Key Concepts

### Traditional Ensemble (Simple Averaging)
```python
# Simple average
prediction = (model_A + model_B + model_C) / 3

# Weighted average
prediction = 0.4 * model_A + 0.3 * model_B + 0.3 * model_C
```

**Limitations:**
- Fixed weights for all samples
- Cannot adapt to different contexts
- Ignores when certain models are more reliable

### Nested Ensemble (Meta-Learning)
```python
# Meta-learner learns dynamic weights
meta_features = [original_features, pred_A, pred_B, pred_C]
prediction = meta_learner(meta_features)
```

**Advantages:**
- ✅ **Context-aware**: Different weights for different scenarios
- ✅ **Dynamic**: Adapts based on input features
- ✅ **Learns patterns**: Discovers when each model excels
- ✅ **Better performance**: Typically 1-5% improvement over simple averaging

## Architecture

```
Level 0 (Base Models)          Level 1 (Meta-Learner)
┌─────────────┐                ┌──────────────────┐
│  XGBoost    │───┐            │                  │
└─────────────┘   │            │                  │
                  ├───────────▶│  Meta-Learner    │──▶ Final
┌─────────────┐   │            │   (XGBoost/      │    Prediction
│  LightGBM   │───┤            │    Ridge/Lasso)  │
└─────────────┘   │            │                  │
                  │            │                  │
┌─────────────┐   │            └──────────────────┘
│  CatBoost   │───┘                    ▲
└─────────────┘                        │
       ▲                               │
       │                               │
  [Features]────────────────────────────┘
                (Context Injection)
```

## How It Works

### Step 1: Train Base Models
Train Level 0 models on training data:
- XGBoost
- LightGBM
- CatBoost
- (optionally) Prophet, Chronos

### Step 2: Generate Base Predictions
Get predictions from all base models on validation data.

### Step 3: Create Meta-Features
Combine original features with base predictions:
```python
meta_features = [
    # Original features (context)
    feature_1, feature_2, ..., feature_n,

    # Base model predictions
    xgboost_pred,
    lightgbm_pred,
    catboost_pred
]
```

### Step 4: Train Meta-Learner
Train meta-learner on meta-features to predict the target.

The meta-learner learns rules like:
- "When volatility is high, trust XGBoost more"
- "When trend is clear, trust LightGBM more"
- "When data is noisy, average all models"

## Usage

### Basic Training (Holdout Validation)

```bash
python scripts/nested_ensemble_predict.py \
    --data-dir data \
    --train-file train.csv \
    --target-col target \
    --date-col date \
    --output-dir nested_ensemble_models
```

### Training with Cross-Validation

```bash
python scripts/nested_ensemble_predict.py \
    --data-dir data \
    --train-file train.csv \
    --use-cv \
    --n-folds 5 \
    --output-dir nested_ensemble_models
```

### Different Meta-Learners

```bash
# XGBoost meta-learner (default, best for non-linear patterns)
python scripts/nested_ensemble_predict.py --meta-learner xgboost

# Ridge regression (fast, linear, prevents overfitting)
python scripts/nested_ensemble_predict.py --meta-learner ridge

# Lasso regression (sparse, feature selection)
python scripts/nested_ensemble_predict.py --meta-learner lasso
```

### Traditional Stacking (No Original Features)

```bash
# Only use base predictions, not original features
python scripts/nested_ensemble_predict.py --no-original-features
```

### Making Predictions

```bash
python scripts/predict_with_nested_ensemble.py \
    --model-dir nested_ensemble_models \
    --test-file data/test.csv \
    --output-file submission.csv
```

## Python API

### Training

```python
from nested_ensemble_predict import NestedEnsemble
from utils import TimeSeriesPreprocessor

# Prepare data
preprocessor = TimeSeriesPreprocessor()
df_features = preprocessor.create_all_features(train_df, 'target')

X = df_features[feature_cols].values
y = df_features['target'].values
X = preprocessor.fit_transform(X)

# Create and train ensemble
ensemble = NestedEnsemble(
    meta_learner_type='xgboost',      # 'xgboost', 'ridge', 'lasso'
    use_original_features=True         # Include context
)

# Train with holdout
metrics = ensemble.train_with_holdout(X, y, val_split=0.3)

# OR train with cross-validation
metrics = ensemble.train_with_cv(X, y, n_folds=5)

# Save
ensemble.save('my_ensemble')
```

### Prediction

```python
# Load
ensemble = NestedEnsemble()
ensemble.load('my_ensemble')

# Predict
predictions = ensemble.predict(X_test)
```

## Performance Comparison

Typical improvements over simple averaging:

| Dataset Type | Improvement |
|--------------|-------------|
| High volatility | 2-5% |
| Clear patterns | 1-3% |
| Noisy data | 1-2% |
| Complex non-linear | 3-7% |

Example output:
```
COMPARISON: Meta-Learner vs Simple Average
======================================================================
Simple Average RMSE:   0.125647
Meta-Learner RMSE:     0.119832
Improvement:           4.63%
```

## Meta-Learner Feature Importance

When using XGBoost as meta-learner, you can analyze which features and base predictions are most important:

```
Meta-Learner Feature Importance
----------------------------------------------------------------------
Base Model Predictions:
  xgboost     : 0.2847
  lightgbm    : 0.2635
  catboost    : 0.2518

Top 10 Original Features:
  Feature  42: 0.0234  # lag_7
  Feature  67: 0.0189  # rolling_mean_30
  Feature  15: 0.0156  # month
  ...
```

This tells you:
- **Which base models** the meta-learner relies on most
- **Which features** provide the most context for weighting decisions

## Advanced: Out-of-Fold Predictions

Cross-validation prevents overfitting by using out-of-fold predictions:

```
Fold 1: Train on [0-1000]  → Predict on [1000-1250]
Fold 2: Train on [0-1250]  → Predict on [1250-1500]
Fold 3: Train on [0-1500]  → Predict on [1500-1750]
Fold 4: Train on [0-1750]  → Predict on [1750-2000]

Meta-learner trains on concatenated predictions [1000-2000]
```

This ensures the meta-learner never sees predictions from models that were trained on the same data.

## When to Use Nested Ensemble vs Simple Ensemble

### Use Nested Ensemble When:
- ✅ You have enough data (>5000 samples)
- ✅ Different models excel in different scenarios
- ✅ You want to maximize performance
- ✅ You can afford longer training time

### Use Simple Ensemble When:
- ✅ Limited data (<1000 samples)
- ✅ Models perform similarly
- ✅ Need fast training
- ✅ Simplicity is preferred

## Tips for Best Results

1. **Data Split**
   - Use at least 30% of data for meta-learner training
   - Ensure temporal ordering is preserved

2. **Meta-Learner Selection**
   - XGBoost: Best overall, handles non-linear patterns
   - Ridge: Fast, prevents overfitting, good for linear patterns
   - Lasso: Sparse, good for feature selection

3. **Feature Engineering**
   - Include diverse features for context
   - More context = better dynamic weighting

4. **Base Model Diversity**
   - Use different model types (tree-based + neural + statistical)
   - Diverse models = better ensemble

5. **Regularization**
   - Keep meta-learner simple (max_depth=3 for XGBoost)
   - Prevent overfitting on base predictions

## Common Issues

### Meta-learner overfits
**Solution:**
- Use Ridge/Lasso instead of XGBoost
- Reduce meta-learner complexity (lower max_depth)
- Use more CV folds

### No improvement over simple average
**Solution:**
- Ensure base models are diverse
- Include original features (context injection)
- Check if validation split is representative

### Training takes too long
**Solution:**
- Use holdout instead of cross-validation
- Use Ridge/Lasso instead of XGBoost
- Reduce number of base models

## References

- Wolpert, D. H. (1992). "Stacked generalization"
- Breiman, L. (1996). "Stacked regressions"
- [Kaggle Ensembling Guide](https://mlwave.com/kaggle-ensembling-guide/)

---

**Remember:** The nested ensemble learns to be smart about combining models. It's not magic, but when used correctly, it consistently outperforms simple averaging! 🚀
