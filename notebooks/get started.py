# %% [markdown]
# # Time Series Forecasting - Getting Started
# 
# This notebook demonstrates how to use the time series forecasting models for the Hull Tactical Market Prediction competition.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)


from utils import TimeSeriesPreprocessor, split_time_series
from models import (
    XGBoostTimeSeriesModel,
    LightGBMTimeSeriesModel,
    #CatBoostTimeSeriesModel,
    ProphetTimeSeriesModel,
    ChronosTimeSeriesModel
)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 6)

print("Setup complete!")

# %% [markdown]
# ## 2. Load and Explore Data

# %%
# Load data
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')


print(f"\n Data Shapes:")
print(f"  Train: {train_df.shape}")
print(f"  Test:  {test_df.shape}")

print(f"\n Target Variable: market_forward_excess_returns = forward_returns - risk_free_rate")
print(f"  This is the S&P 500 excess return over risk-free rate")

print(f"\n Time Period:")
print(f"  Training: date_id {train_df['date_id'].min()} to {train_df['date_id'].max()} ({len(train_df)} days)")
print(f"  Test:     date_id {test_df['date_id'].min()} to {test_df['date_id'].max()} ({len(test_df)} days)")
print(f"  Assuming ~252 trading days/year: {len(train_df)/252:.1f} years of data")

# Parse dates
train_df['date'] = train_df['date_id'].astype(int)
train_df['target'] = train_df['market_forward_excess_returns']


train_df = train_df.sort_values('date').reset_index(drop=True)
train_df.drop(['date_id'],axis=1)

print(f"Data shape: {train_df.shape}")
print(f"\nDate range: {train_df['date'].min()} to {train_df['date'].max()}")
print(f"\nFirst few rows:")
train_df.head()

# %%
print("\n" + "="*80)
print(" TARGET VARIABLE ANALYSIS (Critical for Sharpe Optimization)")
print("="*80)

target = train_df['market_forward_excess_returns'].values
forward_returns = train_df['forward_returns'].values
risk_free = train_df['risk_free_rate'].values

print(f"\n Target Statistics:")
print(f"  Mean:     {target.mean():.6f} ({target.mean()*252*100:.2f}% annualized)")
print(f"  Median:   {np.median(target):.6f}")
print(f"  Std:      {target.std():.6f} ({target.std()*np.sqrt(252)*100:.2f}% annualized)")
print(f"  Min:      {target.min():.6f}")
print(f"  Max:      {target.max():.6f}")
print(f"  Skewness: {stats.skew(target):.3f}")
print(f"  Kurtosis: {stats.kurtosis(target):.3f} (excess)")

# Sharpe ratio calculation
sharpe = (target.mean() / target.std()) * np.sqrt(252)
print(f"\n Baseline Sharpe Ratio: {sharpe:.3f}")
print(f"   (This is what we're trying to beat!)")

# Winning vs losing days
print(f"\n Return Distribution:")
print(f"  Positive days: {(target > 0).sum()} ({(target > 0).sum()/len(target)*100:.1f}%)")
print(f"  Negative days: {(target < 0).sum()} ({(target < 0).sum()/len(target)*100:.1f}%)")
print(f"  Near zero (<0.1%): {(np.abs(target) < 0.001).sum()}")

# Extreme events
print(f"\n Extreme Events (>2 std):")
extreme_positive = (target > target.mean() + 2*target.std()).sum()
extreme_negative = (target < target.mean() - 2*target.std()).sum()
print(f"  Extreme positive: {extreme_positive} days")
print(f"  Extreme negative: {extreme_negative} days")
print(f"  Total extreme: {extreme_positive + extreme_negative} ({(extreme_positive + extreme_negative)/len(target)*100:.1f}%)")

# %%
fig = plt.figure(figsize=(18, 10))

# Time series plot
ax1 = plt.subplot(3, 3, 1)
plt.plot(train_df['date_id'], target, alpha=0.6, linewidth=0.5)
plt.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
plt.fill_between(train_df['date_id'], 0, target, where=(target>0), alpha=0.3, color='green', label='Positive')
plt.fill_between(train_df['date_id'], 0, target, where=(target<0), alpha=0.3, color='red', label='Negative')
plt.title('Target Over Time', fontsize=12, fontweight='bold')
plt.xlabel('Date ID')
plt.ylabel('Excess Returns')
plt.legend()
plt.grid(True, alpha=0.3)

# Distribution histogram
ax2 = plt.subplot(3, 3, 2)
plt.hist(target, bins=100, edgecolor='black', alpha=0.7)
plt.axvline(target.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {target.mean():.5f}')
plt.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Zero')
plt.title('Distribution of Excess Returns', fontsize=12, fontweight='bold')
plt.xlabel('Excess Returns')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

# Q-Q plot (test for normality)
ax3 = plt.subplot(3, 3, 3)
stats.probplot(target, dist="norm", plot=plt)
plt.title('Q-Q Plot (Normality Test)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# Rolling mean and std
ax4 = plt.subplot(3, 3, 4)
rolling_mean = pd.Series(target).rolling(window=63).mean()  # ~3 months
rolling_std = pd.Series(target).rolling(window=63).std()
plt.plot(train_df['date_id'], rolling_mean, label='Rolling Mean (63d)', linewidth=2)
plt.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.fill_between(train_df['date_id'], rolling_mean - rolling_std, rolling_mean + rolling_std,
                 alpha=0.3, label='±1 Std')
plt.title('Rolling Mean & Volatility (63-day window)', fontsize=12, fontweight='bold')
plt.xlabel('Date ID')
plt.ylabel('Excess Returns')
plt.legend()
plt.grid(True, alpha=0.3)

# Autocorrelation
ax5 = plt.subplot(3, 3, 5)
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(pd.Series(target))
plt.title('Autocorrelation of Target', fontsize=12, fontweight='bold')
plt.xlabel('Lag (days)')
plt.ylabel('Autocorrelation')
plt.grid(True, alpha=0.3)

# Cumulative returns
ax6 = plt.subplot(3, 3, 6)
cumulative_returns = (1 + pd.Series(target)).cumprod()
plt.plot(train_df['date_id'], cumulative_returns, linewidth=2, color='navy')
plt.title('Cumulative Returns (Compound)', fontsize=12, fontweight='bold')
plt.xlabel('Date ID')
plt.ylabel('Cumulative Return Factor')
plt.grid(True, alpha=0.3)

# Monthly returns boxplot
ax7 = plt.subplot(3, 3, 7)
train_df['month'] = train_df['date_id'] // 21  # Approximate month
monthly_data = train_df.groupby('month')['market_forward_excess_returns'].apply(list)
plt.boxplot([m for m in monthly_data if len(m) > 0], showfliers=True)
plt.axhline(0, color='red', linestyle='--', alpha=0.7)
plt.title('Returns Distribution by Month', fontsize=12, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Excess Returns')
plt.grid(True, alpha=0.3)

# Volatility over time
ax8 = plt.subplot(3, 3, 8)
rolling_vol = pd.Series(target).rolling(window=21).std()  # ~1 month
plt.plot(train_df['date_id'], rolling_vol * np.sqrt(252), linewidth=1.5, color='darkred')
plt.title('Rolling Volatility (21-day, annualized)', fontsize=12, fontweight='bold')
plt.xlabel('Date ID')
plt.ylabel('Annualized Volatility')
plt.grid(True, alpha=0.3)

# Return vs volatility scatter
ax9 = plt.subplot(3, 3, 9)
window = 63
rolling_ret = pd.Series(target).rolling(window=window).mean()
rolling_vol = pd.Series(target).rolling(window=window).std()
plt.scatter(rolling_vol, rolling_ret, alpha=0.5, s=10)
plt.xlabel('Volatility (63-day)')
plt.ylabel('Mean Return (63-day)')
plt.title('Risk-Return Tradeoff', fontsize=12, fontweight='bold')
plt.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.axvline(rolling_vol.mean(), color='green', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)

plt.tight_layout()

# %%
feature_groups = {}
for prefix in ['D', 'E', 'I', 'M', 'P', 'S', 'V']:
    cols = [c for c in train_df.columns if c.startswith(prefix)]
    feature_groups[prefix] = cols

group_descriptions = {
    'D': 'Categorical/Binary Regime Indicators',
    'E': 'Economic Indicators',
    'I': 'Interest Rate Features',
    'M': 'Market Features',
    'P': 'Price/Performance Features',
    'S': 'Sentiment Features',
    'V': 'Volatility Features'
}

print(f"\n Feature Group Summary:")
print(f"{'Group':<8} {'Count':<8} {'Missing %':<12} {'Description'}")
print("-" * 80)

for prefix, cols in feature_groups.items():
    if len(cols) > 0:
        missing_pct = train_df[cols].isnull().sum().sum() / (len(train_df) * len(cols)) * 100
        print(f"{prefix:<8} {len(cols):<8} {missing_pct:<12.1f} {group_descriptions[prefix]}")

# %%
# Calculate missing percentages
missing_analysis = []
for col in train_df.columns:
    if col not in ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']:
        missing_count = train_df[col].isnull().sum()
        if missing_count > 0:
            missing_pct = missing_count / len(train_df) * 100
            # When does missing data appear?
            first_non_null = train_df[col].first_valid_index()
            missing_analysis.append({
                'feature': col,
                'missing_count': missing_count,
                'missing_pct': missing_pct,
                'first_valid_idx': first_non_null,
                'group': col[0]
            })

missing_df = pd.DataFrame(missing_analysis).sort_values('missing_pct', ascending=False)

print(f"\n Missing Data Summary:")
print(f"  Total features: {len([c for c in train_df.columns if c not in ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']])}")
print(f"  Features with missing data: {len(missing_df)}")
print(f"  Features >50% missing: {(missing_df['missing_pct'] > 50).sum()}")
print(f"  Features >80% missing: {(missing_df['missing_pct'] > 80).sum()}")

print(f"\n Top 10 Most Sparse Features:")
print(missing_df.head(10).to_string(index=False))

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Missing data heatmap by group
ax = axes[0, 0]
group_missing = missing_df.groupby('group')['missing_pct'].agg(['mean', 'min', 'max', 'count'])
group_missing.plot(kind='bar', ax=ax)
ax.set_title('Missing Data Statistics by Feature Group', fontsize=12, fontweight='bold')
ax.set_xlabel('Feature Group')
ax.set_ylabel('Missing Percentage')
ax.legend(['Mean', 'Min', 'Max', 'Count'])
ax.grid(True, alpha=0.3)

# Distribution of missing percentages
ax = axes[0, 1]
ax.hist(missing_df['missing_pct'], bins=50, edgecolor='black', alpha=0.7)
ax.axvline(50, color='red', linestyle='--', linewidth=2, label='50% threshold')
ax.axvline(80, color='darkred', linestyle='--', linewidth=2, label='80% threshold')
ax.set_title('Distribution of Missing Data Percentages', fontsize=12, fontweight='bold')
ax.set_xlabel('Missing Percentage')
ax.set_ylabel('Number of Features')
ax.legend()
ax.grid(True, alpha=0.3)

# When does missing data start? (important for understanding data collection)
ax = axes[1, 0]
first_valid_counts = missing_df['first_valid_idx'].value_counts().sort_index()
ax.plot(first_valid_counts.index, first_valid_counts.values, marker='o', linewidth=2)
ax.set_title('When Do Features Become Available?', fontsize=12, fontweight='bold')
ax.set_xlabel('Date ID (First Valid Index)')
ax.set_ylabel('Number of Features Starting')
ax.grid(True, alpha=0.3)

# Missing data by time period (check if missing is time-dependent)
ax = axes[1, 1]
# Sample a few high-missing features
high_missing_features = missing_df.head(5)['feature'].tolist()
for feat in high_missing_features:
    missing_by_time = train_df[feat].isnull().rolling(window=100).mean()
    ax.plot(train['date_id'], missing_by_time, label=feat, alpha=0.7)
ax.set_title('Missing Data Rate Over Time (Top 5 Sparse Features)', fontsize=12, fontweight='bold')
ax.set_xlabel('Date ID')
ax.set_ylabel('Missing Rate (100-day window)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()

# %%
# Calculate correlations
correlations = []
for col in train_df.columns:
    if col not in ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']:
        valid_mask = train_df[col].notna()
        if valid_mask.sum() > 100:  # At least 100 valid points
            corr_pearson, p_value = pearsonr(train_df.loc[valid_mask, col],
                                             train_df.loc[valid_mask, 'market_forward_excess_returns'])
            correlations.append({
                'feature': col,
                'correlation': corr_pearson,
                'abs_correlation': abs(corr_pearson),
                'p_value': p_value,
                'significant': p_value < 0.05,
                'group': col[0],
                'valid_samples': valid_mask.sum()
            })

corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)

print(f"\n Correlation Summary:")
print(f"  Features analyzed: {len(corr_df)}")
print(f"  Significantly correlated (p<0.05): {corr_df['significant'].sum()}")
print(f"  Correlation >0.05: {(corr_df['abs_correlation'] > 0.05).sum()}")
print(f"  Correlation >0.10: {(corr_df['abs_correlation'] > 0.10).sum()}")

print(f"\n Top 15 Most Correlated Features:")
print(corr_df.head(15)[['feature', 'correlation', 'p_value', 'group']].to_string(index=False))

print(f"\n Top 15 Least Correlated Features:")
print(corr_df.tail(15)[['feature', 'correlation', 'p_value', 'group']].to_string(index=False))

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top correlations by group
ax = axes[0, 0]
top_by_group = corr_df.groupby('group').apply(lambda x: x.nlargest(3, 'abs_correlation')).reset_index(drop=True)
colors = plt.cm.RdYlGn(0.5 + top_by_group['correlation'] / 2)
bars = ax.barh(range(len(top_by_group)), top_by_group['abs_correlation'], color=colors)
ax.set_yticks(range(len(top_by_group)))
ax.set_yticklabels(top_by_group['feature'], fontsize=8)
ax.set_xlabel('Absolute Correlation')
ax.set_title('Top 3 Features by Group (Absolute Correlation)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Correlation distribution by group
ax = axes[0, 1]
for group in corr_df['group'].unique():
    group_corrs = corr_df[corr_df['group'] == group]['correlation']
    ax.hist(group_corrs, bins=20, alpha=0.5, label=f'Group {group}')
ax.axvline(0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('Correlation with Target')
ax.set_ylabel('Frequency')
ax.set_title('Correlation Distribution by Feature Group', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Correlation vs valid samples
ax = axes[1, 0]
scatter = ax.scatter(corr_df['valid_samples'], corr_df['abs_correlation'],
                     c=corr_df['significant'], cmap='RdYlGn', alpha=0.6, s=50)
ax.set_xlabel('Number of Valid Samples')
ax.set_ylabel('Absolute Correlation')
ax.set_title('Correlation Strength vs Data Availability', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Significant (p<0.05)')
ax.grid(True, alpha=0.3)

# P-value distribution
ax = axes[1, 1]
ax.hist(corr_df['p_value'], bins=50, edgecolor='black', alpha=0.7)
ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')
ax.set_xlabel('P-value')
ax.set_ylabel('Frequency')
ax.set_title('Statistical Significance Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

# %%
# Check 1: Test data has lagged features
if 'lagged_forward_returns' in test_df.columns:
    print(f"   Test set has lagged features (lagged_forward_returns, etc.)")
    print(f"    This means we CAN use lagged values from test set")
else:
    print(f"   Test set does NOT have lagged features")

# Check 2: Check if train/test overlap
train_max_date = train_df['date_id'].max()
test_min_date = test_df['date_id'].min()
print(f"\n  Train max date_id: {train_max_date}")
print(f"  Test min date_id:  {test_min_date}")
print(f"  Gap: {test_min_date - train_max_date} days")

if test_min_date > train_max_date:
    print(f"   No temporal overlap (test is after train)")
else:
    print(f"   WARNING: Potential temporal overlap!")

# Check 3: Feature value ranges
print(f"\n Checking if test features are within train ranges...")
feature_range_issues = []
for col in test_df.columns:
    if col in train_df.columns and col not in ['date_id', 'is_scored']:
        train_min, train_max = train_df[col].min(), train_df[col].max()
        test_min, test_max = test_df[col].min(), test_df[col].max()

        if not pd.isna(test_min) and not pd.isna(train_min):
            if test_min < train_min or test_max > train_max:
                feature_range_issues.append({
                    'feature': col,
                    'train_range': f"[{train_min:.4f}, {train_max:.4f}]",
                    'test_range': f"[{test_min:.4f}, {test_max:.4f}]"
                })

if len(feature_range_issues) > 0:
    print(f"   {len(feature_range_issues)} features have out-of-range values in test set")
    print(f"    (This could indicate distribution shift)")
else:
    print(f"   All test features within train ranges")

print(f"\n Leakage Prevention Recommendations:")
print(f"  1. Use only past data for feature engineering")
print(f"  2. Create lag features AFTER train/test split")
print(f"  3. Use purged time series cross-validation")
print(f"  4. No group-based target encoding on full data")
print(f"  5. Be careful with rolling statistics - use shift(1)")

# %%
# Augmented Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(target[~np.isnan(target)])
print(f"  ADF Statistic: {adf_result[0]:.4f}")
print(f"  P-value: {adf_result[1]:.4f}")
print(f"  Result: {'Stationary ✓' if adf_result[1] < 0.05 else 'Non-stationary ⚠'}")

# Autocorrelation analysis
print(f"\n Autocorrelation at key lags:")
for lag in [1, 5, 21, 63]:
    if len(target) > lag:
        autocorr = pd.Series(target).autocorr(lag=lag)
        print(f"  Lag {lag:3d}: {autocorr:7.4f}")

# %%
# Select top features for detailed analysis
top_features = corr_df.head(40)['feature'].tolist()
print(f"Analyzing top {len(top_features)} correlated features...")

# Rolling window parameters
rolling_windows = [63, 126, 252]  # 3 months, 6 months, 1 year
window_names = ['3-month', '6-month', '1-year']

# Calculate rolling correlations for each feature
rolling_corr_results = {}

for feature in top_features:
    feature_data = train_df[feature].values
    target_data = train_df['market_forward_excess_returns'].values

    # Create a DataFrame for rolling calculations
    temp_df = pd.DataFrame({
        'feature': feature_data,
        'target': target_data
    })

    rolling_corr_results[feature] = {}

    for window, name in zip(rolling_windows, window_names):
        # Calculate rolling Pearson correlation
        rolling_corr = temp_df['feature'].rolling(window=window, min_periods=window//2).corr(temp_df['target'])
        rolling_corr_results[feature][name] = rolling_corr.values

        # Calculate rolling Spearman correlation (rank-based, more robust)
        def rolling_spearman(x, y, window):
            result = np.full(len(x), np.nan)
            for i in range(window, len(x)):
                x_window = x[i-window:i]
                y_window = y[i-window:i]
                valid_mask = ~(np.isnan(x_window) | np.isnan(y_window))
                if valid_mask.sum() > window//2:
                    result[i] = spearmanr(x_window[valid_mask], y_window[valid_mask])[0]
            return result

        if name == '3-month':  # Only calculate Spearman for one window to save time
            rolling_corr_results[feature]['spearman_3m'] = rolling_spearman(
                feature_data, target_data, window
            )

print(f"  ✓ Calculated rolling correlations for {len(rolling_windows)} window sizes")

# %%
# ============================================================================
#  ROLLING MUTUAL INFORMATION (Nonlinear Relationships)
# ============================================================================

print("Mutual Information captures both linear AND nonlinear dependencies...")

def calculate_rolling_mi(feature_data, target_data, window=252, step=21):
    """
    Calculate rolling mutual information between feature and target.

    Parameters:
    - window: rolling window size (default 252 = 1 year)
    - step: step size to reduce computation (default 21 = monthly)

    Returns:
    - mi_values: array of MI values at each step
    - mi_indices: indices corresponding to each MI value
    """
    mi_values = []
    mi_indices = []

    for i in range(window, len(feature_data), step):
        x_window = feature_data[i-window:i].reshape(-1, 1)
        y_window = target_data[i-window:i]

        # Remove NaN values
        valid_mask = ~(np.isnan(x_window.flatten()) | np.isnan(y_window))

        if valid_mask.sum() > window//2:
            try:
                mi = mutual_info_regression(
                    x_window[valid_mask],
                    y_window[valid_mask],
                    n_neighbors=5,
                    random_state=42
                )[0]
                mi_values.append(mi)
                mi_indices.append(i)
            except:
                mi_values.append(np.nan)
                mi_indices.append(i)
        else:
            mi_values.append(np.nan)
            mi_indices.append(i)

    return np.array(mi_values), np.array(mi_indices)

# Calculate rolling MI for top features (limited to top 10 for computational efficiency)
rolling_mi_results = {}
mi_features = top_features[:50]

print(f"Calculating rolling MI for top {len(mi_features)} features (this may take a moment)...")

for i, feature in enumerate(mi_features):
    feature_data = train_df[feature].values
    target_data = train_df['market_forward_excess_returns'].values

    mi_values, mi_indices = calculate_rolling_mi(feature_data, target_data)
    rolling_mi_results[feature] = {
        'mi_values': mi_values,
        'mi_indices': mi_indices
    }

    if (i + 1) % 5 == 0:
        print(f"  Processed {i+1}/{len(mi_features)} features...")

# %%
# ============================================================================
#  STRUCTURAL BREAK DETECTION IN CORRELATIONS
# ============================================================================

print("Detecting regime changes in feature-target relationships...")

structural_breaks = {}
RUPTURES_AVAILABLE = False

if RUPTURES_AVAILABLE:
    for feature in mi_features[:40]:  # Analyze top 5 for structural breaks
        # Use the rolling correlation as the signal for break detection
        rolling_corr = rolling_corr_results[feature]['3-month']

        # Remove NaN values for break detection
        valid_mask = ~np.isnan(rolling_corr)
        valid_indices = np.where(valid_mask)[0]
        valid_signal = rolling_corr[valid_mask]

        if len(valid_signal) > 100:
            try:
                # Use PELT algorithm for change point detection
                model = rpt.Pelt(model="rbf", min_size=63).fit(valid_signal)
                breaks = model.predict(pen=10)

                # Convert back to original indices
                original_breaks = [valid_indices[min(b, len(valid_indices)-1)] for b in breaks[:-1]]
                structural_breaks[feature] = original_breaks

                print(f"  {feature}: {len(original_breaks)} structural breaks detected")
            except Exception as e:
                structural_breaks[feature] = []
                print(f"  {feature}: Could not detect breaks ({str(e)[:30]})")
        else:
            structural_breaks[feature] = []
else:
    print("  ⚠ Skipping structural break detection (ruptures not installed)")
    # Simple alternative: detect large changes in correlation
    for feature in mi_features[:40]:
        rolling_corr = rolling_corr_results[feature]['3-month']
        valid_mask = ~np.isnan(rolling_corr)

        if valid_mask.sum() > 100:
            # Calculate rolling std of correlation changes
            corr_diff = np.abs(np.diff(rolling_corr[valid_mask]))
            threshold = np.nanmean(corr_diff) + 2 * np.nanstd(corr_diff)
            break_points = np.where(corr_diff > threshold)[0]
            structural_breaks[feature] = break_points.tolist()[:10]  # Limit to 10 breaks
            print(f"  {feature}: {len(structural_breaks[feature])} potential regime changes (simple detection)")

# %%
print("Analyzing high/low correlation regimes and their characteristics...")

regime_analysis = []

for feature in mi_features:
    rolling_corr = rolling_corr_results[feature]['3-month']
    valid_corr = rolling_corr[~np.isnan(rolling_corr)]

    if len(valid_corr) > 100:
        # Define correlation regimes
        corr_mean = np.mean(valid_corr)
        corr_std = np.std(valid_corr)

        high_corr_threshold = corr_mean + 0.5 * corr_std
        low_corr_threshold = corr_mean - 0.5 * corr_std

        # Classify each period
        high_corr_periods = valid_corr > high_corr_threshold
        low_corr_periods = valid_corr < low_corr_threshold
        neutral_periods = ~(high_corr_periods | low_corr_periods)

        # Calculate regime statistics
        regime_analysis.append({
            'feature': feature,
            'mean_corr': corr_mean,
            'std_corr': corr_std,
            'high_corr_pct': high_corr_periods.sum() / len(valid_corr) * 100,
            'low_corr_pct': low_corr_periods.sum() / len(valid_corr) * 100,
            'neutral_pct': neutral_periods.sum() / len(valid_corr) * 100,
            'max_corr': np.max(valid_corr),
            'min_corr': np.min(valid_corr),
            'corr_range': np.max(valid_corr) - np.min(valid_corr),
            # Stability score: lower std and range = more stable
            'stability_score': 1 / (1 + corr_std + (np.max(valid_corr) - np.min(valid_corr))/2)
        })

regime_df = pd.DataFrame(regime_analysis)
regime_df = regime_df.sort_values('stability_score', ascending=False)

print(f"\n Feature Correlation Regime Summary:")
print(regime_df[['feature', 'mean_corr', 'std_corr', 'corr_range', 'stability_score']].to_string(index=False))

# %%
# ============================================================================
#  COMPREHENSIVE FEATURE STABILITY SCORING
# ============================================================================

# Select top features for detailed analysis
#top_features = corr_df.head(50)['feature'].tolist()

# Split data into periods for period-based analysis
n_periods = 5
period_size = len(train_df) // n_periods

stability_analysis = []

for feature in top_features:
    period_corrs = []
    period_mi = []

    for period in range(n_periods):
        start_idx = period * period_size
        end_idx = (period + 1) * period_size if period < n_periods - 1 else len(train_df)

        period_data = train_df.iloc[start_idx:end_idx]
        valid_mask = period_data[feature].notna()

        if valid_mask.sum() > 50:
            # Pearson correlation
            corr, p_val = pearsonr(
                period_data.loc[valid_mask, feature],
                period_data.loc[valid_mask, 'market_forward_excess_returns']
            )
            period_corrs.append(corr)

            # Mutual information for this period
            try:
                mi = mutual_info_regression(
                    period_data.loc[valid_mask, feature].values.reshape(-1, 1),
                    period_data.loc[valid_mask, 'market_forward_excess_returns'].values,
                    n_neighbors=5,
                    random_state=42
                )[0]
                period_mi.append(mi)
            except:
                period_mi.append(np.nan)
        else:
            period_corrs.append(np.nan)
            period_mi.append(np.nan)

    # Calculate comprehensive stability metrics
    corr_mean = np.nanmean(period_corrs)
    corr_std = np.nanstd(period_corrs)
    mi_mean = np.nanmean(period_mi)
    mi_std = np.nanstd(period_mi)

    # Check if correlation sign is consistent
    valid_corrs = [c for c in period_corrs if not np.isnan(c)]
    sign_consistency = 1.0 if len(valid_corrs) > 0 and (all(c > 0 for c in valid_corrs) or all(c < 0 for c in valid_corrs)) else 0.0

    # Composite stability score
    # Higher = more stable and predictive
    stability_score = (
        abs(corr_mean) * 0.3 +  # Strength of correlation
        (1 - min(corr_std, 0.2) / 0.2) * 0.3 +  # Low variance is good
        sign_consistency * 0.2 +  # Consistent sign is good
        min(mi_mean, 0.1) / 0.1 * 0.2  # MI indicates predictive power
    ) if not np.isnan(corr_mean) else 0

    stability_analysis.append({
        'feature': feature,
        'mean_corr': corr_mean,
        'std_corr': corr_std,
        'min_corr': np.nanmin(period_corrs),
        'max_corr': np.nanmax(period_corrs),
        'mean_mi': mi_mean,
        'std_mi': mi_std,
        'sign_consistent': sign_consistency,
        'stability_score': stability_score,
        'periods': period_corrs
    })

stability_df = pd.DataFrame(stability_analysis)
stability_df = stability_df.sort_values('stability_score', ascending=False)
print(stability_df.shape)
print(f"\n Feature Stability Ranking (Top 20 Features):")
print(stability_df[['feature', 'mean_corr', 'std_corr', 'mean_mi', 'sign_consistent', 'stability_score']].to_string(index=False))

# Identify most stable and unstable features
most_stable = stability_df.head(5)['feature'].tolist()
least_stable = stability_df.tail(5)['feature'].tolist()

print(f"\n Stability Insights:")
print(f"  Most Stable Features: {most_stable}")
print(f"  Least Stable Features: {least_stable}")

# %%
# ============================================================================
#  VISUALIZATION: TIME-VARYING RELATIONSHIPS
# ============================================================================

# Figure 1: Rolling Correlation Heatmap
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Subplot 1: Rolling correlations over time for top 5 features
ax = axes[0, 0]
for feature in mi_features[:5]:
    rolling_corr = rolling_corr_results[feature]['3-month']
    ax.plot(train_df['date_id'].values, rolling_corr, label=feature, alpha=0.7, linewidth=1.5)

ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Date ID')
ax.set_ylabel('Rolling Correlation (3-month)')
ax.set_title('Rolling Correlation Over Time (Top 5 Features)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# Subplot 2: Rolling MI over time
ax = axes[0, 1]
for feature in list(rolling_mi_results.keys())[:5]:
    mi_data = rolling_mi_results[feature]
    ax.plot(mi_data['mi_indices'], mi_data['mi_values'], label=feature, alpha=0.7, linewidth=1.5)

ax.set_xlabel('Date ID')
ax.set_ylabel('Mutual Information')
ax.set_title('Rolling Mutual Information Over Time (Top 5 Features)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# Subplot 3: Correlation distribution by period
ax = axes[1, 0]
period_labels = [f'Period {i+1}' for i in range(n_periods)]
for i, feature in enumerate(mi_features[:5]):
    periods = stability_df[stability_df['feature'] == feature]['periods'].values[0]
    x_positions = np.arange(n_periods) + i * 0.15
    ax.bar(x_positions, periods, width=0.12, label=feature, alpha=0.8)

ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.set_xlabel('Time Period')
ax.set_ylabel('Correlation')
ax.set_title('Feature-Target Correlation by Period', fontsize=12, fontweight='bold')
ax.set_xticks(np.arange(n_periods) + 0.3)
ax.set_xticklabels(period_labels)
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Subplot 4: Stability Score Ranking
ax = axes[1, 1]
top_stable = stability_df.head(15)
colors = plt.cm.RdYlGn(top_stable['stability_score'] / top_stable['stability_score'].max())
bars = ax.barh(range(len(top_stable)), top_stable['stability_score'], color=colors)
ax.set_yticks(range(len(top_stable)))
ax.set_yticklabels(top_stable['feature'], fontsize=9)
ax.set_xlabel('Stability Score')
ax.set_title('Feature Stability Ranking', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# %%
# Create a matrix of correlations over time periods
n_time_bins = 20
time_bin_size = len(train_df) // n_time_bins
corr_evolution_matrix = np.zeros((len(mi_features), n_time_bins))

for i, feature in enumerate(mi_features):
    for j in range(n_time_bins):
        start_idx = j * time_bin_size
        end_idx = (j + 1) * time_bin_size if j < n_time_bins - 1 else len(train_df)

        period_data = train_df.iloc[start_idx:end_idx]
        valid_mask = period_data[feature].notna()

        if valid_mask.sum() > 30:
            corr, _ = pearsonr(
                period_data.loc[valid_mask, feature],
                period_data.loc[valid_mask, 'market_forward_excess_returns']
            )
            corr_evolution_matrix[i, j] = corr
        else:
            corr_evolution_matrix[i, j] = np.nan

fig, ax = plt.subplots(figsize=(16, 10))
im = ax.imshow(corr_evolution_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=0.2)

ax.set_yticks(range(len(mi_features)))
ax.set_yticklabels(mi_features, fontsize=9)
ax.set_xlabel('Time Period', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title('Feature-Target Correlation Evolution Over Time', fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Correlation', fontsize=11)

# Add time period labels
period_labels = [f'P{i+1}' for i in range(n_time_bins)]
ax.set_xticks(range(n_time_bins))
ax.set_xticklabels(period_labels, fontsize=8)

# %% [markdown]
# ## 3. Feature Engineering

# %%
exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns','month']
feature_cols = [c for c in train_df.columns if c not in exclude_cols]
target_col = 'market_forward_excess_returns'


# Create preprocessor
preprocessor = TimeSeriesPreprocessor(scaler_type='standard')


print(f"Total features available: {len(feature_cols)}")
print(f"Target variable: {target_col}")

# Handle missing data using forward fill + median
print("\nHandling missing data...")
train_filled = train_df.copy()
for col in feature_cols:
    if train_df[col].isnull().sum() > 0:
        # Forward fill (data starts at different times)
        train_filled[col] = train_filled[col].fillna(method='ffill')
        # Fill remaining with median
        train_filled[col] = train_filled[col].fillna(train_filled[col].median())

missing_after = train_filled[feature_cols].isnull().sum().sum()
print(f"Missing values after handling: {missing_after}")

# Extract features and target
df_features = train_filled[feature_cols]
y = train_filled[target_col]
# # Create all features
# df_features = preprocessor.create_all_features(
#     train_df[feature_cols],
#     target_col='target',
#     lags=[1, 2, 3, 5, 7, 14, 21, 30],
#     windows=[7, 14, 30, 60]
# )

print(f"Features shape: {df_features.shape}")
print(f"\nFeature columns:")
print(df_features.columns.tolist())

# %% [markdown]
# ## 4. Train-Test Split

# %%
# Split data
train_data, val_data = split_time_series(df_features, test_size=0.2)

print(f"Train set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")

# Prepare features and targets
feature_cols = [col for col in df_features.columns if col not in ['date', 'target']]

X_train = train_data[feature_cols].values
y_train = train_data['target'].values
X_val = val_data[feature_cols].values
y_val = val_data['target'].values

# Scale features
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")

# %% [markdown]
# ## 5. Train Models
# 
# ### 5.1 XGBoost

# %%
# Train XGBoost
xgb_model = XGBoostTimeSeriesModel()
xgb_metrics = xgb_model.train(X_train, y_train, X_val, y_val)

print("\nXGBoost Results:")
print(f"  Train RMSE: {xgb_metrics['train_rmse']:.6f}")
print(f"  Val RMSE: {xgb_metrics['val_rmse']:.6f}")
print(f"  Val MAE: {xgb_metrics['val_mae']:.6f}")

# %% [markdown]
# ### 5.2 LightGBM

# %%
# Train LightGBM
lgb_model = LightGBMTimeSeriesModel()
lgb_metrics = lgb_model.train(X_train, y_train, X_val, y_val)

print("\nLightGBM Results:")
print(f"  Train RMSE: {lgb_metrics['train_rmse']:.6f}")
print(f"  Val RMSE: {lgb_metrics['val_rmse']:.6f}")
print(f"  Val MAE: {lgb_metrics['val_mae']:.6f}")

# %% [markdown]
# ### 5.4 Prophet

# %%
# Prepare data for Prophet
prophet_model = ProphetTimeSeriesModel()
prophet_train = prophet_model.prepare_data(train_data, 'date', 'target')
prophet_val = prophet_model.prepare_data(val_data, 'date', 'target')

# Train Prophet
prophet_metrics = prophet_model.train(prophet_train, verbose=False)

# Validate
val_forecast = prophet_model.predict(prophet_val[['ds']])
val_rmse = np.sqrt(np.mean((val_forecast['yhat'].values - prophet_val['y'].values)**2))
val_mae = np.mean(np.abs(val_forecast['yhat'].values - prophet_val['y'].values))

print("\nProphet Results:")
print(f"  Train RMSE: {prophet_metrics['train_rmse']:.6f}")
print(f"  Val RMSE: {val_rmse:.6f}")
print(f"  Val MAE: {val_mae:.6f}")

# %% [markdown]
# ## 6. Model Comparison

# %%
# Compare models
comparison_df = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'Prophet'],
    'Val RMSE': [
        xgb_metrics['val_rmse'],
        lgb_metrics['val_rmse'],
        cat_metrics['val_rmse'],
        val_rmse
    ],
    'Val MAE': [
        xgb_metrics['val_mae'],
        lgb_metrics['val_mae'],
        cat_metrics['val_mae'],
        val_mae
    ]
})

comparison_df = comparison_df.sort_values('Val RMSE')
print("\nModel Comparison:")
print(comparison_df.to_string(index=False))

# %%
# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

comparison_df.plot(x='Model', y='Val RMSE', kind='bar', ax=axes[0], legend=False)
axes[0].set_title('Validation RMSE Comparison', fontsize=14)
axes[0].set_ylabel('RMSE')
axes[0].grid(True, alpha=0.3)

comparison_df.plot(x='Model', y='Val MAE', kind='bar', ax=axes[1], legend=False, color='orange')
axes[1].set_title('Validation MAE Comparison', fontsize=14)
axes[1].set_ylabel('MAE')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Feature Importance (XGBoost)

# %%
# Get feature importance
importance_df = xgb_model.get_feature_importance(top_n=15)
importance_df['feature_name'] = [feature_cols[i] for i in importance_df['feature']]

# Plot
plt.figure(figsize=(12, 6))
plt.barh(range(len(importance_df)), importance_df['importance'])
plt.yticks(range(len(importance_df)), importance_df['feature_name'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance (XGBoost)', fontsize=14)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Predictions Visualization

# %%
# Get predictions from all models
xgb_pred = xgb_model.predict(X_val)
lgb_pred = lgb_model.predict(X_val)
cat_pred = cat_model.predict(X_val)

# Ensemble prediction (simple average)
ensemble_pred = (xgb_pred + lgb_pred + cat_pred) / 3

# Plot
plt.figure(figsize=(15, 6))
plt.plot(val_data['date'].values, y_val, label='Actual', linewidth=2, alpha=0.8)
plt.plot(val_data['date'].values, xgb_pred, label='XGBoost', alpha=0.6)
plt.plot(val_data['date'].values, lgb_pred, label='LightGBM', alpha=0.6)
plt.plot(val_data['date'].values, cat_pred, label='CatBoost', alpha=0.6)
plt.plot(val_data['date'].values, ensemble_pred, label='Ensemble', linewidth=2, linestyle='--')
plt.title('Model Predictions vs Actual', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Target Value')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate ensemble RMSE
ensemble_rmse = np.sqrt(np.mean((ensemble_pred - y_val)**2))
print(f"\nEnsemble RMSE: {ensemble_rmse:.6f}")

# %% [markdown]
# ## 9. Save Models

# %%
import os
import joblib

# Create output directory
os.makedirs('../trained_models', exist_ok=True)

# Save models
xgb_model.save_model('../trained_models/xgboost_model.json')
lgb_model.save_model('../trained_models/lightgbm_model.txt')
cat_model.save_model('../trained_models/catboost_model.cbm')
prophet_model.save_model('../trained_models/prophet_model.pkl')

# Save preprocessor
joblib.dump(preprocessor, '../trained_models/preprocessor.pkl')

print("All models saved successfully!")

# %% [markdown]
# ## 10. Next Steps
# 
# 1. **Hyperparameter Tuning**: Use the `optimize_hyperparameters()` method to find better parameters
# 2. **Chronos Model**: Try the Chronos-2 foundation model for zero-shot forecasting
# 3. **Ensemble Optimization**: Optimize ensemble weights using validation data
# 4. **Feature Engineering**: Add domain-specific features (holidays, events, etc.)
# 5. **Cross-validation**: Implement time series cross-validation for robust evaluation
# 
# Good luck with the competition! 🚀


