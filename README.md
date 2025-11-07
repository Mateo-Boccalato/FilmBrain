# FilmBrain

A machine learning project for predicting movie revenue using CatBoost regression.

## Project Structure


- `train_advanced.py`: Training Script using DB
- `update_ratings.py`: Script for updating movie ratings
- `mDGetter.py`: Additional data gathering utility: Calls both tmdb and rapidapi for differnt movie data points.

## Features

- Revenue prediction using CatBoost regression
- Advanced feature engineering including:
  - Temporal features (release timing, seasonality)
  - Budget-based features
  - Vote and rating features
  - Genre analysis
  - Studio classification
- Data preprocessing and cleaning
- Model evaluation with multiple metrics (RMSE, MAE, SMAPE)

## Model Performance

Current metrics on test set:
- RMSE: ~$29M
- SMAPE: ~60%
- MAE: ~$20.5M

## Requirements

See `requirements.txt` for dependencies.

## Usage

1. Prepare your data in CSV format
2. Run the training script:
```bash
python train_improved.py
```

## Future Improvements

- Enhanced feature engineering
- Ensemble modeling
- Better handling of outliers
- More sophisticated vote/rating features
