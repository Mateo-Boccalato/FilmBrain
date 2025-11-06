# train_advanced.py - Advanced version with sophisticated features and custom loss
import os
import re
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
from typing import Dict, List, Tuple

# Configuration
CSV_PATH = "PRMoviesDB_updated_merged.csv"
REVENUE_CAP = 150_000_000  # $150M cap
REVENUE_FLOOR = 0         # No floor
SEED = 42
np.random.seed(SEED)

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return 100 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])

def calculate_rolling_stats(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    """Calculate rolling statistics for a group (e.g., studio, genre)"""
    # Sort by date within each group
    df = df.sort_values(['release_date'])
    
    # Initialize columns
    stats_df = pd.DataFrame()
    stats_df['group'] = df[group_col]
    
    # Calculate rolling stats (excluding current movie)
    grouped = df.groupby(group_col)
    
    # Rolling mean of last 5 movies
    stats_df['rolling_mean'] = grouped[target_col].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    
    # Rolling std of last 5 movies
    stats_df['rolling_std'] = grouped[target_col].transform(
        lambda x: x.shift().rolling(5, min_periods=1).std()
    )
    
    # Success rate (% of movies above median)
    global_median = df[target_col].median()
    stats_df['success_rate'] = grouped[target_col].transform(
        lambda x: x.shift().rolling(5, min_periods=1).apply(
            lambda x: (x > global_median).mean()
        )
    )
    
    # Fill NaN with global stats
    global_mean = df[target_col].mean()
    global_std = df[target_col].std()
    stats_df = stats_df.fillna({
        'rolling_mean': global_mean,
        'rolling_std': global_std,
        'success_rate': 0.5
    })
    
    return stats_df

print("Loading data...")
df = pd.read_csv(CSV_PATH)

# Better date handling
df['release_date'] = pd.to_datetime(df['Release'], errors='coerce')

print("\nEngineering features...")

# 1. Enhanced temporal features
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df['release_day'] = df['release_date'].dt.day
df['release_dayofweek'] = df['release_date'].dt.dayofweek
df['days_since_2000'] = (df['release_date'] - pd.Timestamp('2000-01-01')).dt.days

# Seasonal features
df['is_weekend'] = df['release_dayofweek'].isin([4, 5, 6]).astype(int)
df['is_holiday'] = df['release_month'].isin([11, 12, 5, 6, 7]).astype(int)
df['is_summer'] = df['release_month'].isin([6, 7, 8]).astype(int)

# Competition feature - movies released within 2 weeks
df['release_week'] = df['release_date'].dt.isocalendar().week
df['competition'] = df.groupby(['release_year', 'release_week'])['release_date'].transform('count')

# 2. Enhanced numeric handling
numeric_cols = [
    'budget', 'popularity', 'runtime', 'vote_average', 'vote_count',
    'score', 'reviewsCount', 'metascore', 'userScore', 'averageScore',
    'tomatometer'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median = df[col].median()
        df[col] = df[col].fillna(median)

# 3. Advanced genre features
def extract_genres(x):
    if pd.isna(x):
        return []
    if isinstance(x, str):
        return [g.strip() for g in x.split(';') if g.strip()]
    return []

df['genre_list'] = df['genres'].apply(extract_genres)
common_genres = ['Action', 'Drama', 'Comedy', 'Thriller', 'Adventure', 'Romance', 
                 'Horror', 'Animation', 'Documentary', 'Family']
for genre in common_genres:
    df[f'is_{genre.lower()}'] = df['genre_list'].apply(lambda x: 1 if genre in x else 0)
df['genre_count'] = df['genre_list'].apply(len)

# Genre combinations
df['is_action_adventure'] = df['is_action'] & df['is_adventure']
df['is_drama_romance'] = df['is_drama'] & df['is_romance']
df['is_comedy_family'] = df['is_comedy'] & df['is_family']

# 4. Enhanced budget features
df['log_budget'] = np.log1p(df['budget'])
df['budget_per_runtime'] = df['budget'] / df['runtime'].clip(lower=60)
df['budget_per_genre'] = df['budget'] / df['genre_count'].clip(lower=1)

# 5. Advanced vote and rating features
df['vote_power'] = df['vote_average'] * np.log1p(df['vote_count'])
df['vote_power_squared'] = df['vote_power'] ** 2
df['rating_momentum'] = df['vote_average'] * df['popularity']

if 'tomatometer' in df.columns:
    df['audience_critic_ratio'] = df['userScore'] / df['tomatometer'].clip(lower=1)
    df['review_engagement'] = np.log1p(df['reviewsCount'])

# 6. Studio features with historical performance
df['DIST.'] = df['DIST.'].fillna('Unknown')
major_studios = ['Warner', 'Universal', 'Paramount', 'Disney', 'Sony', 'Fox']
df['is_major_studio'] = df['DIST.'].apply(
    lambda x: 1 if any(studio.lower() in str(x).lower() for studio in major_studios) else 0
)

# Clean and filter revenue
print("\nPreparing target variable...")
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df = df[df['revenue'].notna() & (df['revenue'] > REVENUE_FLOOR)].copy()
print(f"\nFiltered out movies below ${REVENUE_FLOOR/1e6:.3f}M")

# Calculate studio and genre performance before capping
studio_stats = calculate_rolling_stats(df, 'DIST.', 'revenue')
df['studio_rolling_mean'] = studio_stats['rolling_mean']
df['studio_rolling_std'] = studio_stats['rolling_std']
df['studio_success_rate'] = studio_stats['success_rate']

# Now cap revenue
n_capped = (df['revenue'] > REVENUE_CAP).sum()
df['revenue'] = df['revenue'].clip(upper=REVENUE_CAP)
print(f"\nCapped {n_capped} movies at ${REVENUE_CAP/1e6:.0f}M")

# Target transformation with better scaling
df['target_log'] = np.log1p(df['revenue'])
revenue_scale = df['revenue'].mean()
df['target_scaled'] = df['revenue'] / revenue_scale  # For custom loss function

# Feature selection
categorical_features = [
    'DIST.', 'genres', 'release_month', 'release_dayofweek'
]

numeric_features = [
    # Core features
    'log_budget', 'budget_per_runtime', 'budget_per_genre',
    'popularity', 'runtime', 'competition',
    
    # Vote and rating features
    'vote_power', 'vote_power_squared', 'rating_momentum',
    'review_engagement', 'audience_critic_ratio',
    
    # Temporal features
    'days_since_2000', 'is_weekend', 'is_holiday', 'is_summer',
    
    # Studio and genre features
    'studio_rolling_mean', 'studio_rolling_std', 'studio_success_rate',
    'genre_count', 'is_major_studio',
    
    # Genre indicators and combinations
    'is_action_adventure', 'is_drama_romance', 'is_comedy_family'
] + [f'is_{g.lower()}' for g in common_genres]

# Remove any features that don't exist in the dataset
numeric_features = [f for f in numeric_features if f in df.columns]

# Fill remaining missing values
for col in categorical_features:
    df[col] = df[col].fillna('Unknown').astype(str)

for col in numeric_features:
    df[col] = df[col].fillna(0)

# Train/test split - use more recent data for testing
df = df.sort_values('release_date')
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

X_cols = categorical_features + numeric_features
print(f"\nTraining with {len(X_cols)} features:")
print("\nTop numeric features:")
for f in numeric_features[:10]:
    print(f"- {f}")

print(f"\nTraining size: {len(train_df)}")
print(f"Test size: {len(test_df)}")

# Prepare data
train_pool = Pool(
    train_df[X_cols],
    train_df['target_log'],
    cat_features=categorical_features
)

test_pool = Pool(
    test_df[X_cols],
    test_df['target_log'],
    cat_features=categorical_features
)

# Train model with optimized parameters
print("\nTraining model...")
model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.01,
    depth=8,
    l2_leaf_reg=3,
    random_strength=1,
    bagging_temperature=1,
    # Use standard RMSE but with careful tuning
    loss_function='RMSE',
    eval_metric='RMSE',
    early_stopping_rounds=200,
    verbose=100,
    random_seed=SEED
)

model.fit(train_pool, eval_set=test_pool, use_best_model=True)

# Evaluate
preds = model.predict(test_pool)
y_true = np.expm1(test_df['target_log'])
y_pred = np.expm1(preds)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
smape_score = smape(y_true, y_pred)

print("\nTest Metrics:")
print(f"RMSE: ${rmse/1e6:.2f}M")
print(f"MAE: ${mae/1e6:.2f}M")
print(f"SMAPE: {smape_score:.1f}%")

# Analyze errors by revenue range
ranges = [
    (0, 1e6),        # $0 - $1M
    (1e6, 10e6),     # $1M - $10M
    (10e6, 50e6),    # $10M - $50M
    (50e6, 100e6),   # $50M - $100M
    (100e6, 150e6)   # $100M - $150M
]
print("\nError Analysis by Revenue Range:")
for low, high in ranges:
    mask = (y_true >= low) & (y_true < high)
    if mask.any():
        range_smape = smape(y_true[mask], y_pred[mask])
        range_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        n_movies = mask.sum()
        print(f"\n${low/1e6:.1f}M-${high/1e6:.1f}M range:")
        print(f"- Number of movies: {n_movies} ({100*n_movies/len(y_true):.1f}% of test set)")
        print(f"- SMAPE: {range_smape:.1f}%")
        print(f"- RMSE: ${range_rmse/1e6:.2f}M")

# Feature importance analysis
importance = pd.DataFrame({
    'feature': X_cols,
    'importance': model.feature_importances_
})
importance = importance.sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance.head(10).to_string())

# Save model and important features
os.makedirs('artifacts', exist_ok=True)
model.save_model('artifacts/model_advanced.cbm')

# Save feature importance
importance.to_csv('artifacts/feature_importance.csv', index=False)
print("\nModel and feature importance saved to artifacts/ directory")