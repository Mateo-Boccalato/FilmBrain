# train_improved.py - Enhanced version with better features and tuning
import os
import re
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

# Configuration
CSV_PATH = "PRMoviesDB_updated_merged.csv"
REVENUE_CAP = 120_000_000  # $120M cap
SEED = 42
np.random.seed(SEED)

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return 100 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])

print("Loading data...")
df = pd.read_csv(CSV_PATH)

# Better date handling
df['release_date'] = pd.to_datetime(df['Release'], errors='coerce')

# Advanced feature engineering
print("\nEngineering features...")

# 1. Temporal features
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df['release_day'] = df['release_date'].dt.day
df['release_dayofweek'] = df['release_date'].dt.dayofweek
df['is_weekend'] = df['release_dayofweek'].isin([4, 5, 6]).astype(int)  # Fri, Sat, Sun
df['release_quarter'] = df['release_month'].apply(lambda x: (x-1)//3 + 1)
df['is_holiday'] = df['release_month'].isin([11, 12, 5, 6, 7]).astype(int)  # Holiday and summer months
df['is_summer'] = df['release_month'].isin([6, 7, 8]).astype(int)

# 2. Better numeric handling
numeric_cols = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count',
                'score', 'reviewsCount', 'metascore', 'userScore', 'averageScore']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill missing values with column median
        median = df[col].median()
        df[col] = df[col].fillna(median)

# 3. Advanced genre handling
def extract_genres(x):
    if pd.isna(x):
        return []
    if isinstance(x, str):
        return [g.strip() for g in x.split(';') if g.strip()]
    return []

df['genre_list'] = df['genres'].apply(extract_genres)
common_genres = ['Action', 'Drama', 'Comedy', 'Thriller', 'Adventure', 'Romance']
for genre in common_genres:
    df[f'is_{genre.lower()}'] = df['genre_list'].apply(lambda x: 1 if genre in x else 0)
df['genre_count'] = df['genre_list'].apply(len)

# 4. Advanced budget features
df['log_budget'] = np.log1p(df['budget'])
df['budget_per_runtime'] = df['budget'] / df['runtime'].clip(lower=60)

# 5. Vote and rating features
df['vote_power'] = df['vote_average'] * np.log1p(df['vote_count'])
df['rating_momentum'] = df['vote_average'] * df['popularity']

# 6. Distributor features
df['DIST.'] = df['DIST.'].fillna('Unknown')
major_studios = ['Warner', 'Universal', 'Paramount', 'Disney', 'Sony', 'Fox']
df['is_major_studio'] = df['DIST.'].apply(
    lambda x: 1 if any(studio.lower() in str(x).lower() for studio in major_studios) else 0
)

# Clean and cap revenue
print("\nPreparing target variable...")
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df = df[df['revenue'].notna() & (df['revenue'] > 0)].copy()
print(f"Revenue stats before cap:\n{df['revenue'].describe()}")

# Apply revenue cap
n_capped = (df['revenue'] > REVENUE_CAP).sum()
df['revenue'] = df['revenue'].clip(upper=REVENUE_CAP)
print(f"\nCapped {n_capped} movies at ${REVENUE_CAP/1e6:.0f}M")
print(f"Revenue stats after cap:\n{df['revenue'].describe()}")

# Target transformation
df['target_log'] = np.log1p(df['revenue'])

# Feature selection
categorical_features = [
    'DIST.', 'genres', 'release_quarter',
    'release_month', 'release_dayofweek'
]

numeric_features = [
    'log_budget', 'budget_per_runtime',
    'popularity', 'runtime',
    'vote_power', 'rating_momentum',
    'genre_count', 'is_major_studio',
    'is_weekend', 'is_holiday', 'is_summer'
] + [f'is_{g.lower()}' for g in common_genres]

# Fill and convert categorical features to strings
for col in categorical_features:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown').astype(str)

for col in numeric_features:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Train/test split - use more recent data for testing
df = df.sort_values('release_date')
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

X_cols = categorical_features + numeric_features
print(f"\nTraining with {len(X_cols)} features:")
print("Categorical:", categorical_features)
print("Numeric:", numeric_features)

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

# Train model with better parameters
print("\nTraining model...")
model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.01,  # Smaller learning rate
    depth=8,             # Slightly deeper trees
    l2_leaf_reg=3,      # Prevent overfitting
    random_strength=1,   # Randomization for better generalization
    bagging_temperature=1, # More aggressive bagging
    verbose=100,
    early_stopping_rounds=200,
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

# Feature importance analysis
importance = pd.DataFrame({
    'feature': X_cols,
    'importance': model.feature_importances_
})
importance = importance.sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance.head(10).to_string())

# Save model
os.makedirs('artifacts', exist_ok=True)
model.save_model('artifacts/model_improved.cbm')