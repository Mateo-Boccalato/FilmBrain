"""
Interactive movie revenue prediction script.
Loads the trained CatBoost model and predicts revenue for user-input movie title.
Automatically fetches movie data from TMDB and IMDb APIs.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from catboost import CatBoostRegressor
from mDGetter import gather_movie_data

def display_movie_info(movie_data):
    """Display fetched movie information"""
    print("\n=== Movie Details ===")
    print(f"Title: {movie_data['title']}")
    print(f"Release Date: {movie_data['release_date']}")
    print(f"Genres: {', '.join(movie_data['genres'])}")
    if movie_data['budget']:
        print(f"Budget: {format_currency(movie_data['budget'])}")
    print(f"Runtime: {movie_data['runtime']} minutes")
    if movie_data['vote_average']:
        print(f"TMDB Rating: {movie_data['vote_average']}/10 ({movie_data['vote_count']} votes)")
    if movie_data['tomatometer']:
        print(f"Tomatometer: {movie_data['tomatometer']}%")
    print("=" * 30)

def prepare_features(movie_data):
    """Transform API data into model features"""
    # Initial data preparation
    budget = float(movie_data['budget'] or 0)
    runtime = float(movie_data['runtime'] or 0)
    genre_list = movie_data['genres']
    
    # Create base features
    movie = {
        'budget': budget,
        'genres': ';'.join(str(g) for g in genre_list),
        'runtime': runtime,
        'DIST.': str(movie_data['origin_country'] or "Unknown"),
        'release_date': str(movie_data['release_date']),
        'vote_average': float(movie_data['vote_average'] or 0),
        'vote_count': float(movie_data['vote_count'] or 0),
        'popularity': float(movie_data['popularity'] or 0),
        'score': float(movie_data['score'] or 0),
        'reviewsCount': float(movie_data['reviewsCount'] or 0),
        'metascore': float(movie_data['metascore'] or 0),
        'userScore': float(movie_data['userScore'] or 0),
        'averageScore': float(movie_data['averageScore'] or 0),
        'tomatometer': float(movie_data['tomatometer'] or 0)
    }
    
    df = pd.DataFrame([movie])
    
    # Basic date features
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month.astype(str)  # Convert to string for categorical
    df['release_day'] = df['release_date'].dt.day
    df['release_dayofweek'] = df['release_date'].dt.dayofweek.astype(str)  # Convert to string for categorical
    df['days_since_2000'] = (df['release_date'] - pd.Timestamp('2000-01-01')).dt.days
    
    # Competition feature (approximate since we don't have full dataset)
    df['release_week'] = df['release_date'].dt.isocalendar().week
    df['competition'] = 1  # Default to 1 since we can't calculate true competition
    
    # Seasonal features
    df['is_weekend'] = df['release_dayofweek'].isin([4, 5, 6]).astype(int)
    df['is_holiday'] = df['release_month'].isin([11, 12, 5, 6, 7]).astype(int)
    df['is_summer'] = df['release_month'].isin([6, 7, 8]).astype(int)
    
    # Budget features
    df['log_budget'] = np.log1p(df['budget'])
    df['budget_per_runtime'] = df['budget'] / df['runtime'].clip(lower=60)
    
    # Genre features
    df['genre_list'] = df['genres'].str.split(';')
    df['genre_count'] = df['genre_list'].str.len()
    df['budget_per_genre'] = df['budget'] / df['genre_count'].clip(lower=1)
    
    # Studio features
    major_studios = ['Warner', 'Universal', 'Paramount', 'Disney', 'Sony', 'Fox']
    df['is_major_studio'] = df['DIST.'].apply(
        lambda x: 1 if any(studio.lower() in str(x).lower() for studio in major_studios) else 0
    )
    
    # Genre indicators
    common_genres = ['Action', 'Drama', 'Comedy', 'Thriller', 'Adventure', 'Romance', 
                    'Horror', 'Animation', 'Documentary', 'Family']
    for genre in common_genres:
        df[f'is_{genre.lower()}'] = df['genre_list'].apply(lambda x: 1 if genre in x else 0)
    
    # Genre combinations
    df['is_action_adventure'] = df['is_action'] & df['is_adventure']
    df['is_drama_romance'] = df['is_drama'] & df['is_romance']
    df['is_comedy_family'] = df['is_comedy'] & df['is_family']
    
    # Studio performance (using defaults since we don't have historical data)
    df['studio_rolling_mean'] = df['budget'] * 1.5  # Rough estimate
    df['studio_rolling_std'] = df['budget'] * 0.5   # Rough estimate
    df['studio_success_rate'] = 0.5                 # Default 50%
    
    # Vote and rating features
    df['vote_power'] = df['vote_average'] * np.log1p(df['vote_count'])
    df['vote_power_squared'] = df['vote_power'] ** 2
    df['rating_momentum'] = df['vote_average'] * df['popularity']
    
    # Additional features
    df['review_engagement'] = np.log1p(df['reviewsCount'])
    df['audience_critic_ratio'] = df['userScore'] / df['tomatometer'].clip(lower=1) if 'tomatometer' in df else 1.0
    
    # Common genres to check for
    common_genres = ['Action', 'Drama', 'Comedy', 'Thriller', 'Adventure', 'Romance', 
                    'Horror', 'Animation', 'Documentary', 'Family']
    
    for genre in common_genres:
        df[f'is_{genre.lower()}'] = df['genre_list'].apply(lambda x: 1 if genre in x else 0)
    
    # Genre combinations
    df['is_action_adventure'] = df['is_action'] & df['is_adventure']
    df['is_drama_romance'] = df['is_drama'] & df['is_romance']
    df['is_comedy_family'] = df['is_comedy'] & df['is_family']
    
    # Vote and rating features
    df['vote_power'] = df['vote_average'] * np.log1p(df['vote_count'])
    df['vote_power_squared'] = df['vote_power'] ** 2
    df['rating_momentum'] = df['vote_average'] * df['popularity']
    
    # Studio features
    major_studios = ['Warner', 'Universal', 'Paramount', 'Disney', 'Sony', 'Fox']
    df['is_major_studio'] = df['DIST.'].apply(
        lambda x: 1 if any(studio.lower() in str(x).lower() for studio in major_studios) else 0
    )
    
    return df

def format_currency(amount):
    """Format amount in millions or billions"""
    if amount >= 1_000_000_000:
        return f"${amount/1_000_000_000:.2f}B"
    else:
        return f"${amount/1_000_000:.2f}M"

def get_feature_lists():
    """Get lists of categorical and numeric features in the exact order expected by the model"""
    categorical_features = [
        'DIST.',
        'genres',
        'release_month',
        'release_dayofweek'
    ]
    
    common_genres = ['Action', 'Drama', 'Comedy', 'Thriller', 'Adventure', 'Romance', 
                    'Horror', 'Animation', 'Documentary', 'Family']
    
    numeric_features = [
        # Core features
        'log_budget',
        'budget_per_runtime',
        'budget_per_genre',
        'popularity',
        'runtime',
        'competition',
        
        # Vote and rating features
        'vote_power',
        'vote_power_squared',
        'rating_momentum',
        'review_engagement',
        'audience_critic_ratio',
        
        # Temporal features
        'days_since_2000',
        'is_weekend',
        'is_holiday',
        'is_summer',
        
        # Studio and genre features
        'studio_rolling_mean',
        'studio_rolling_std',
        'studio_success_rate',
        'genre_count',
        'is_major_studio',
        
        # Genre indicators and combinations
        'is_action_adventure',
        'is_drama_romance',
        'is_comedy_family'
    ] + [f'is_{g.lower()}' for g in common_genres]
    
    return categorical_features, numeric_features

def main():
    # Load the model
    print("Loading model...")
    model_path = 'artifacts/model_advanced.cbm'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = CatBoostRegressor()
    model.load_model(model_path)
    
    # Get feature lists
    cat_features, num_features = get_feature_lists()
    
    while True:
        try:
            # Get movie title from user
            title = input("\nEnter movie title: ").strip()
            if not title:
                break
                
            print(f"\nFetching data for '{title}'...")
            movie_data = gather_movie_data(title)
            
            # Display fetched movie information
            display_movie_info(movie_data)
            
            # Prepare features
            df = prepare_features(movie_data)
            
            # Ensure categorical features are strings
            for cat_feature in cat_features:
                if cat_feature in df.columns:
                    df[cat_feature] = df[cat_feature].astype(str)
            
            # Make prediction
            print("\nAnalyzing...")
            from catboost import Pool
            
            # Select only the features we need for prediction
            features_to_use = cat_features + num_features
            prediction_df = df[features_to_use]
            
            prediction_pool = Pool(prediction_df, cat_features=cat_features)
            prediction = model.predict(prediction_pool)
            
            # Convert log prediction back to dollars
            predicted_revenue = np.expm1(prediction[0])
            
            print("\n=== Revenue Prediction ===")
            print(f"Predicted Revenue: {format_currency(predicted_revenue)}")
            
            # If we have budget, show ROI
            if movie_data['budget'] and movie_data['budget'] > 0:
                budget_ratio = predicted_revenue / movie_data['budget']
                print(f"Predicted Return on Investment: {budget_ratio:.2f}x")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different movie title.")
        
        # Ask if user wants to try another prediction
        again = input("\nPredict another movie? (y/n): ").strip().lower()
        if again != 'y':
            break
    
    print("\nThank you for using the Movie Revenue Predictor!")

if __name__ == "__main__":
    main()