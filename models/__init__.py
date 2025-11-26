"""Time series forecasting models"""
# Tree-based models
from .xgboost_model import XGBoostTimeSeriesModel, XGBoostMultiStepForecaster
from .lightgbm_model import LightGBMTimeSeriesModel, LightGBMMultiStepForecaster
from .catboost_model import CatBoostTimeSeriesModel, CatBoostMultiStepForecaster

# Statistical and foundation models
from .prophet_model import ProphetTimeSeriesModel, ProphetEnsembleModel
from .chronos_model import ChronosTimeSeriesModel, ChronosEnsemble, ChronosFineTuner

# Transformer models (SOTA for time series)
from .patchtst_model import PatchTSTTimeSeriesModel
from .itransformer_model import iTransformerTimeSeriesModel
from .timesnet_model import TimesNetTimeSeriesModel

# Hybrid models (Combining multiple approaches)
from .hybrid_chronos_patchtst import HybridChronosPatchTSTModel

__all__ = [
    # Tree-based
    'XGBoostTimeSeriesModel',
    'XGBoostMultiStepForecaster',
    'LightGBMTimeSeriesModel',
    'LightGBMMultiStepForecaster',
    'CatBoostTimeSeriesModel',
    'CatBoostMultiStepForecaster',
    # Statistical
    'ProphetTimeSeriesModel',
    'ProphetEnsembleModel',
    # Foundation
    'ChronosTimeSeriesModel',
    'ChronosEnsemble',
    'ChronosFineTuner',
    # Transformers
    'PatchTSTTimeSeriesModel',
    'iTransformerTimeSeriesModel',
    'TimesNetTimeSeriesModel',
    # Hybrid
    'HybridChronosPatchTSTModel'
]
