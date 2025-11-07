# FilmBrain

A comprehensive movie exploration platform combining a modern React frontend with machine learning-powered revenue predictions using CatBoost regression.

## Features

### Web Application
- **Movie Database Integration**: Browse and search movies using TMDB API
- **Detailed Movie Information**: View comprehensive details about movies, including:
  - Cast and crew information
  - Release dates
  - Ratings and reviews
  - Box office performance
- **Revenue Predictions**: AI-powered box office predictions using our FilmBrain model
  - View both actual and predicted revenues when available
  - Get revenue estimates for upcoming movies
- **User-Friendly Interface**: Modern, responsive design with intuitive navigation
- **Real-Time Predictions**: Instant revenue predictions through integrated API

### Machine Learning Model
- **Revenue Prediction**: Using CatBoost regression
- **Advanced Feature Engineering**:
  - Temporal features (release timing, seasonality)
  - Budget-based features
  - Vote and rating features
  - Genre analysis
  - Studio classification
- **High Accuracy**: Trained on extensive movie dataset
- **Real-Time Integration**: Seamlessly integrated with web frontend

## Project Structure

```
FilmBrain/
├── webApp/                    # React frontend application
│   └── react-movie-db-master/
│       ├── src/              # React source code
│       ├── FilmBrain/        # Python backend
│       └── public/           # Static assets
├── artifacts/                # Model artifacts and data
├── train_advanced.py         # Advanced model training
├── mDGetter.py              # Data collection utility
├── predict_revenue.py        # Prediction module
└── requirements.txt         # Python dependencies
```

## Technical Stack

- **Frontend**: 
  - React
  - MobX for state management
  - SCSS for styling
  - TMDB API integration
- **Backend**:
  - Flask API
  - CatBoost ML model
  - Python data processing
- **ML Model**:
  - CatBoost Regressor
  - Feature engineering pipeline
  - Model performance tracking

## Model Performance

Current metrics on test set:
- RMSE: ~$29M
- SMAPE: ~60%
- MAE: ~$20.5M

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   # Install Python dependencies
   pip install -r requirements.txt
   
   # Install React dependencies
   cd webApp/react-movie-db-master
   npm install
   ```
3. Start the backend server:
   ```bash
   cd FilmBrain
   python api.py
   ```
4. Start the React application:
   ```bash
   cd webApp/react-movie-db-master
   npm start
   ```
5. Access the application at `http://localhost:3000`

## Requirements

- Python 3.8+
- Node.js 14+
- See `requirements.txt` for Python dependencies
- See `webApp/react-movie-db-master/package.json` for Node.js dependencies

## Future Improvements

- Enhanced feature engineering
- User accounts and personalization
- Recommendation system integration
- Mobile application
- Enhanced visualization of prediction metrics
- Export and sharing capabilities
- Ensemble modeling
- Better handling of outliers
- More sophisticated vote/rating features
