# Hull Tactical Kaggle pipeline (notebook-first)

This repository restructures the Hull Tactical Market Prediction project around a cookie-cutter/Kedro-inspired layout that keeps data in a dedicated folder and uses Jupyter notebooks as the primary execution surface.

## Business case
- Build time-series models that rank well on the Hull Kaggle leaderboard while keeping experimentation reproducible.
- Standardize data cleaning, feature generation, and cross-validation so that every model uses identical folds and evaluation (the competition Sharpe metric).
- Enable meta-learning blends so new model signals can be swapped in without rewriting the pipeline.

## Architecture (high level)
- `data/01_raw`, `02_interim`, `03_processed`, `04_models`: cookie-cutter data zones, ignored from Git except for `.gitkeep` placeholders.
- `conf/base/catalog.yml`: lightweight Kedro-style catalog describing where train/test and model artifacts live.
- `notebooks/pipelines/00_data_cleaning.ipynb`: single source of truth for cleaning raw CSVs, feature generation, and saving shared folds.
- `notebooks/models/*.ipynb`: one notebook per model family (e.g., XGBoost, LightGBM) that pulls the processed matrix and runs the centralized 10-fold chronological validation.
- `notebooks/meta/meta_predict.ipynb`: meta-learner that blends model predictions and writes the final submission file.
- `utils/`: small Python helpers for data catalog paths, preprocessing, chronological CV, the Hull Sharpe metric, and blending.

## Data and metrics
- Place `train.csv` and `test.csv` in `data/01_raw/`. Paths are resolved automatically by the catalog helpers.
- Cleaning notebook outputs:
  - `data/02_interim/clean_train.parquet` and saved fold CSVs.
  - `data/03_processed/model_matrix.parquet` (model-ready features) and `oof_predictions.parquet` (for the meta-learner).
- Validation uses a centralized 10-split chronological scheme with the competition Sharpe-style score (higher is better) implemented in `utils/metrics.py`.

## How to run
1. Install the dependencies in `requirements.txt` (or mirror them in your preferred environment).
2. Add raw Kaggle files to `data/01_raw/train.csv` and `data/01_raw/test.csv`.
3. Execute `notebooks/pipelines/00_data_cleaning.ipynb` to create cleaned data and cross-validation folds.
4. Run one or more model notebooks in `notebooks/models/` to generate scores and out-of-fold predictions.
5. Open `notebooks/meta/meta_predict.ipynb` to blend model outputs and produce `data/04_models/submission.csv`.

## Meta-learning predict function
The `blend_predictions` helper in `utils/meta.py` builds a simple weighted ensemble. Adjust the `weights` argument inside the meta notebook to experiment with different blends or add new model columns to the OOF file to extend the meta learner without changing the surrounding code.

## Improvements and next steps
- Expand the model notebook set (e.g., deep models) that all consume the same feature matrix and folds.
- Add automated tests for the catalog paths and scoring helpers.
- Integrate experiment tracking for each CV run.
- Parameterize feature generation and model hyperparameters through `conf/` to mirror a full Kedro pipeline.
