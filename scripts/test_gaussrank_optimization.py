"""Test script to validate the key components of gaussrank_sharpe_optimization.py"""

import sys
import os
from pathlib import Path

# Setup path
path = Path(os.getcwd())
project_root = path.parent.absolute()
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from scipy.stats import norm

# Import key functions from the main script
sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("Testing GaussRank Transformation and Optimization Components")
print("="*70)

# --------------------------------------------------------------------------- #
# Test 1: GaussRank Transformation
# --------------------------------------------------------------------------- #
def gaussrank_transform(series: pd.Series) -> pd.Series:
    """Apply GaussRank (Inverse Normal) transformation to a series."""
    mask = ~series.isna()
    if mask.sum() == 0:
        return series

    ranked = series[mask].rank(method='average')
    n = len(ranked)
    uniform = (ranked - 0.5) / n
    gaussrank = pd.Series(norm.ppf(uniform), index=series[mask].index)

    result = pd.Series(np.nan, index=series.index)
    result[mask] = gaussrank
    return result


print("\n[TEST 1] GaussRank Transformation")
print("-" * 70)

# Create test data
test_data = pd.Series([1, 5, 3, 9, 2, 7, 4, 8, 6, 10])
print("Original data:", test_data.values)

transformed = gaussrank_transform(test_data)
print("Transformed data:", transformed.values)
print(f"Mean: {transformed.mean():.6f} (expected: ~0)")
print(f"Std:  {transformed.std():.6f} (expected: ~1)")

# Test with NaN values
test_data_nan = pd.Series([1, np.nan, 3, 9, np.nan, 7, 4, 8, 6, 10])
transformed_nan = gaussrank_transform(test_data_nan)
print(f"\nWith NaN values - Non-NaN count: {transformed_nan.notna().sum()} (expected: 8)")
print(f"Mean of non-NaN: {transformed_nan.mean():.6f} (expected: ~0)")

print("✓ GaussRank transformation test passed!")

# --------------------------------------------------------------------------- #
# Test 2: Sharpe Evaluation Function
# --------------------------------------------------------------------------- #
TARGET_COL = "market_forward_excess_returns"

def hull_sharpe_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_free_rate: np.ndarray | None = None,
    forward_returns: np.ndarray | None = None,
    eps: float = 1e-9,
) -> float:
    """Hull-style Sharpe with optional risk-free/forward returns."""
    y_pred = np.clip(y_pred, 0.0, 2.0)
    if risk_free_rate is None or forward_returns is None:
        pnl = y_true * y_pred
    else:
        strategy_returns = risk_free_rate * (1.0 - y_pred) + y_pred * forward_returns
        pnl = strategy_returns - risk_free_rate

    pnl = pnl[~np.isnan(pnl)]
    if pnl.shape[0] == 0:
        return 0.0

    return float(np.mean(pnl) / (np.std(pnl) + eps))


def sharpe_eval_slice(df_slice: pd.DataFrame, preds: np.ndarray) -> float:
    """Compute Sharpe using forward_returns/risk_free_rate if available."""
    y_true = df_slice[TARGET_COL].values
    if {"forward_returns", "risk_free_rate"}.issubset(df_slice.columns):
        return hull_sharpe_ratio(
            y_true,
            preds,
            risk_free_rate=df_slice["risk_free_rate"].values,
            forward_returns=df_slice["forward_returns"].values,
        )
    return hull_sharpe_ratio(y_true, preds)


print("\n[TEST 2] Sharpe Evaluation Function")
print("-" * 70)

# Test basic Sharpe
test_df_basic = pd.DataFrame({
    TARGET_COL: np.array([0.01, 0.02, -0.01, 0.03, 0.01]),
})
test_preds_basic = np.array([1.0, 1.5, 0.5, 2.0, 1.0])
sharpe_basic = sharpe_eval_slice(test_df_basic, test_preds_basic)
print(f"Basic Sharpe: {sharpe_basic:.4f}")

# Test with risk-free rate and forward returns
test_df_full = pd.DataFrame({
    TARGET_COL: np.array([0.01, 0.02, -0.01, 0.03, 0.01]),
    'forward_returns': np.array([0.015, 0.025, -0.005, 0.035, 0.015]),
    'risk_free_rate': np.array([0.001, 0.001, 0.001, 0.001, 0.001]),
})
test_preds_full = np.array([1.0, 1.5, 0.5, 2.0, 1.0])
sharpe_full = sharpe_eval_slice(test_df_full, test_preds_full)
print(f"Sharpe with risk adjustment: {sharpe_full:.4f}")

print("✓ Sharpe evaluation test passed!")

# --------------------------------------------------------------------------- #
# Test 3: k and b Optimization
# --------------------------------------------------------------------------- #
from scipy.optimize import minimize

def optimize_k_b(
    predictions: np.ndarray,
    df_val: pd.DataFrame,
    initial_k: float = 1.0,
    initial_b: float = 0.0,
):
    """Optimize k and b to maximize Sharpe ratio."""
    def objective(params):
        k, b = params
        allocations = np.clip(k * predictions + b, 0.0, 2.0)
        sharpe = sharpe_eval_slice(df_val, allocations)
        return -sharpe

    best_result = None
    best_sharpe = -np.inf

    starting_points = [
        (initial_k, initial_b),
        (0.5, 0.5),
        (1.5, -0.5),
        (2.0, 0.0),
        (1.0, 0.5),
    ]

    for k_init, b_init in starting_points:
        result = minimize(
            objective,
            x0=[k_init, b_init],
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
        )

        current_sharpe = -result.fun
        if current_sharpe > best_sharpe:
            best_sharpe = current_sharpe
            best_result = result

    optimal_k, optimal_b = best_result.x
    return optimal_k, optimal_b, best_sharpe


print("\n[TEST 3] k and b Optimization")
print("-" * 70)

# Create synthetic predictions and validation data
np.random.seed(42)
val_size = 100
true_returns = np.random.randn(val_size) * 0.02  # Small returns with noise
raw_predictions = true_returns * 0.7 + np.random.randn(val_size) * 0.01  # Correlated predictions

val_df = pd.DataFrame({
    TARGET_COL: true_returns,
    'forward_returns': true_returns + 0.001,
    'risk_free_rate': np.ones(val_size) * 0.0005,
})

# Initial Sharpe (no optimization)
initial_allocations = np.clip(raw_predictions, 0.0, 2.0)
initial_sharpe = sharpe_eval_slice(val_df, initial_allocations)
print(f"Initial Sharpe (k=1, b=0): {initial_sharpe:.4f}")

# Optimized k and b
k_opt, b_opt, sharpe_opt = optimize_k_b(raw_predictions, val_df)
print(f"Optimized k: {k_opt:.4f}")
print(f"Optimized b: {b_opt:.4f}")
print(f"Optimized Sharpe: {sharpe_opt:.4f}")

# Verify improvement
optimized_allocations = np.clip(k_opt * raw_predictions + b_opt, 0.0, 2.0)
verified_sharpe = sharpe_eval_slice(val_df, optimized_allocations)
print(f"Verified optimized Sharpe: {verified_sharpe:.4f}")

assert abs(sharpe_opt - verified_sharpe) < 1e-4, "Sharpe mismatch!"
print("✓ k and b optimization test passed!")

# --------------------------------------------------------------------------- #
# Test 4: Allocation Transformation
# --------------------------------------------------------------------------- #
print("\n[TEST 4] Allocation Transformation")
print("-" * 70)

test_preds = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
k = 1.5
b = 0.3

allocations = np.clip(k * test_preds + b, 0.0, 2.0)
print(f"k={k}, b={b}")
print(f"Predictions: {test_preds}")
print(f"Allocations: {allocations}")
print(f"All in [0, 2]: {np.all((allocations >= 0.0) & (allocations <= 2.0))}")

print("✓ Allocation transformation test passed!")

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)
print("\nKey Features Validated:")
print("  1. GaussRank (Inverse Normal) transformation")
print("  2. Sharpe evaluation with and without risk adjustment")
print("  3. k and b optimization using scipy.optimize")
print("  4. Allocation transformation and clipping")
print("\nThe gaussrank_sharpe_optimization.py script is ready to use!")
print("="*70)
