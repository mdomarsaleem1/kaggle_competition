# Time Series Forecasting Models for Hull Tactical Market Prediction

A comprehensive collection of state-of-the-art time series forecasting models for the Hull Tactical Market Prediction Kaggle competition.

## 📋 Overview

This repository implements five powerful time series forecasting approaches:

1. **XGBoost** - Gradient boosting framework with tree-based models
2. **LightGBM** - Fast gradient boosting framework by Microsoft
3. **CatBoost** - Gradient boosting library with native categorical feature support
4. **Prophet** - Facebook's time series forecasting tool
5. **Chronos-2** - Amazon's universal time series forecasting foundation model

## 🏗️ Project Structure

```
kaggle_competition/
├── data/                          # Data directory
├── models/                        # Model implementations
│   ├── xgboost_model.py          # XGBoost implementation
│   ├── lightgbm_model.py         # LightGBM implementation
│   ├── catboost_model.py         # CatBoost implementation
│   ├── prophet_model.py          # Prophet implementation
│   └── chronos_model.py          # Chronos-2 implementation
├── utils/                         # Utility functions
│   └── data_utils.py             # Data preprocessing utilities
├── scripts/                       # Training and prediction scripts
│   ├── train_all_models.py       # Train all models
│   ├── ensemble_predict.py       # Ensemble predictions
│   └── example_usage.py          # Usage examples
├── notebooks/                     # Jupyter notebooks
├── trained_models/                # Saved models
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd kaggle_competition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For Chronos-2 (optional, requires PyTorch):
```bash
pip install torch transformers accelerate
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

### Usage

#### 1. Prepare Your Data

Place your training data in the `data/` directory. Expected format:
- CSV file with a date column and target column
- Additional feature columns (optional)

Example:
```csv
date,target,feature1,feature2
2020-01-01,100.5,1.2,3.4
2020-01-02,101.2,1.3,3.5
...
```

#### 2. Train All Models

```bash
python scripts/train_all_models.py \
    --data-dir data \
    --train-file train.csv \
    --target-col target \
    --date-col date \
    --output-dir trained_models
```

To include Chronos-2:
```bash
python scripts/train_all_models.py --include-chronos
```

#### 3. Create Ensemble Predictions

```bash
python scripts/ensemble_predict.py \
    --model-dir trained_models \
    --test-file data/test.csv \
    --submission-file submission.csv \
    --method weighted_average
```

#### 4. Run Examples

```bash
python scripts/example_usage.py
```

## 📊 Model Details

### XGBoost

**Strengths:**
- Excellent performance on structured data
- Built-in regularization
- Handles missing values
- Fast training with GPU support

**Key Parameters:**
- `learning_rate`: 0.05
- `max_depth`: 6
- `n_estimators`: 1000
- `early_stopping_rounds`: 50

**Usage:**
```python
from models import XGBoostTimeSeriesModel

model = XGBoostTimeSeriesModel()
metrics = model.train(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
```

### LightGBM

**Strengths:**
- Very fast training speed
- Lower memory usage
- Better accuracy on large datasets
- Native categorical feature support

**Key Parameters:**
- `learning_rate`: 0.05
- `num_leaves`: 31
- `n_estimators`: 1000
- `early_stopping_rounds`: 50

**Usage:**
```python
from models import LightGBMTimeSeriesModel

model = LightGBMTimeSeriesModel()
metrics = model.train(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
```

### CatBoost

**Strengths:**
- Best-in-class handling of categorical features
- Robust to overfitting
- Built-in GPU support
- Ordered boosting algorithm

**Key Parameters:**
- `learning_rate`: 0.05
- `depth`: 6
- `iterations`: 1000
- `early_stopping_rounds`: 50

**Usage:**
```python
from models import CatBoostTimeSeriesModel

model = CatBoostTimeSeriesModel()
metrics = model.train(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
```

### Facebook Prophet

**Strengths:**
- Excellent for data with strong seasonal patterns
- Handles holidays and special events
- Interpretable components (trend, seasonality)
- Robust to missing data and outliers

**Key Parameters:**
- `changepoint_prior_scale`: 0.05
- `seasonality_prior_scale`: 10.0
- `seasonality_mode`: 'additive'

**Usage:**
```python
from models import ProphetTimeSeriesModel

model = ProphetTimeSeriesModel()
prophet_df = model.prepare_data(df, 'date', 'target')
metrics = model.train(prophet_df)
forecast = model.predict(future_df)
```

### Chronos-2 (Foundation Model)

**Strengths:**
- Pre-trained on diverse time series datasets
- Zero-shot forecasting capability
- State-of-the-art performance
- Probabilistic forecasts with uncertainty quantification

**Available Model Sizes:**
- `tiny`: Fastest, smallest model (~8M parameters)
- `mini`: Small model (~20M parameters)
- `small`: Balanced model (~46M parameters)
- `base`: Standard model (~200M parameters)
- `large`: Best performance (~710M parameters)

**Usage:**
```python
from models import ChronosTimeSeriesModel

model = ChronosTimeSeriesModel(model_size='small')
model.load_model()

# Generate forecasts
forecasts = model.predict(
    context=historical_data,
    prediction_length=30,
    num_samples=20
)

# Get quantile forecasts
quantile_forecasts = model.predict_quantiles(
    context=historical_data,
    prediction_length=30,
    quantiles=[0.1, 0.5, 0.9]
)
```

## 🎯 Feature Engineering

The `TimeSeriesPreprocessor` class provides comprehensive feature engineering:

### Lag Features
- Creates lagged versions of the target variable
- Default lags: [1, 2, 3, 5, 7, 14, 21, 30]

### Rolling Window Features
- Rolling mean, std, min, max
- Default windows: [7, 14, 30, 60]

### Time-based Features
- Year, month, day, day of week
- Quarter, day of year, week of year

### Example:
```python
from utils import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor(scaler_type='standard')
df_features = preprocessor.create_all_features(
    df,
    target_col='target',
    lags=[1, 2, 3, 5, 7, 14, 21, 30],
    windows=[7, 14, 30, 60]
)
```

## 🔧 Ensemble Methods

The framework supports multiple ensemble strategies:

1. **Mean Ensemble**: Average predictions from all models
2. **Median Ensemble**: Take median of predictions (robust to outliers)
3. **Weighted Average**: Optimize weights based on validation performance

### Optimize Ensemble Weights:
```python
from scripts.ensemble_predict import optimize_ensemble_weights

optimal_weights = optimize_ensemble_weights(
    model_dir='trained_models',
    val_features=X_val,
    val_targets=y_val
)
```

## 📈 Hyperparameter Optimization

All tree-based models support hyperparameter optimization using Optuna:

```python
model = XGBoostTimeSeriesModel()
best_params = model.optimize_hyperparameters(
    X_train, y_train,
    X_val, y_val,
    n_trials=100
)

# Train with optimized parameters
optimized_model = XGBoostTimeSeriesModel(params=best_params)
metrics = optimized_model.train(X_train, y_train, X_val, y_val)
```

## 💾 Model Persistence

### Save Models:
```python
# XGBoost, LightGBM, CatBoost
model.save_model('model_path')

# Prophet
model.save_model('prophet_model.pkl')
```

### Load Models:
```python
# XGBoost, LightGBM, CatBoost
model = XGBoostTimeSeriesModel()
model.load_model('model_path')

# Prophet
model = ProphetTimeSeriesModel()
model.load_model('prophet_model.pkl')
```

## 🎓 Best Practices

### 1. Data Preparation
- Ensure data is sorted by date
- Handle missing values appropriately
- Check for outliers
- Normalize/standardize features

### 2. Feature Engineering
- Create meaningful lag features based on domain knowledge
- Include seasonal indicators (day of week, month, etc.)
- Add external features (holidays, events, etc.)

### 3. Model Selection
- Use tree-based models (XGBoost, LightGBM, CatBoost) for structured data
- Use Prophet for data with strong seasonality
- Use Chronos-2 for zero-shot forecasting or when training data is limited

### 4. Ensemble
- Combine multiple models for better generalization
- Use different model types in ensemble (tree-based + Prophet + Chronos)
- Optimize ensemble weights on validation set

### 5. Validation
- Use time series cross-validation
- Respect temporal ordering (no data leakage)
- Evaluate on multiple metrics (RMSE, MAE, MAPE)

## 🔬 Advanced Features

### Multi-step Forecasting
```python
from models import XGBoostMultiStepForecaster

forecaster = XGBoostMultiStepForecaster(
    forecast_horizon=7,
    params=custom_params
)
metrics = forecaster.train(X_train, y_train, X_val, y_val)
predictions = forecaster.predict(X_test)  # Shape: [samples, 7]
```

### Cross-validation with Prophet
```python
from prophet.diagnostics import cross_validation, performance_metrics

df_cv = cross_validation(
    model.model,
    initial='730 days',
    period='180 days',
    horizon='365 days'
)
df_metrics = performance_metrics(df_cv)
```

### Chronos Ensemble
```python
from models import ChronosEnsemble

ensemble = ChronosEnsemble(model_sizes=['tiny', 'small', 'base'])
ensemble.load_models()
predictions = ensemble.predict(
    context=historical_data,
    prediction_length=30,
    method='mean'
)
```

## 📊 Evaluation Metrics

The framework tracks multiple evaluation metrics:

- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAE** (Mean Absolute Error): Robust to outliers
- **MAPE** (Mean Absolute Percentage Error): Scale-independent
- **R²** (Coefficient of Determination): Explained variance

## 🛠️ Troubleshooting

### Common Issues

1. **GPU Support**
   - XGBoost: Set `device='cuda'` in parameters
   - LightGBM: Set `device='gpu'` in parameters
   - CatBoost: Set `task_type='GPU'` in parameters
   - Chronos: Automatically uses GPU if available

2. **Memory Issues**
   - Reduce batch size for Chronos
   - Use `num_leaves` parameter in LightGBM
   - Use data sampling for hyperparameter optimization

3. **Installation Issues**
   - Prophet may require additional dependencies (pystan)
   - Chronos requires PyTorch installation first

## 📚 References

- XGBoost: [Documentation](https://xgboost.readthedocs.io/)
- LightGBM: [Documentation](https://lightgbm.readthedocs.io/)
- CatBoost: [Documentation](https://catboost.ai/docs/)
- Prophet: [Documentation](https://facebook.github.io/prophet/)
- Chronos: [Paper](https://arxiv.org/abs/2403.07815) | [GitHub](https://github.com/amazon-science/chronos-forecasting)

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Forecasting! 📈**
