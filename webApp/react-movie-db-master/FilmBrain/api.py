from flask import Flask, request, jsonify
from flask_cors import CORS
from mDGetter import gather_movie_data
from predict_revenue import prepare_features, get_feature_lists
from catboost import CatBoostRegressor, Pool
import numpy as np
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})

@app.route('/')
def home():
    return jsonify({
        'status': 'ok',
        'message': 'FilmBrain API is running',
        'endpoints': {
            'predict_revenue': {
                'url': '/predict_revenue',
                'method': 'POST',
                'body': {'title': 'movie title'}
            }
        }
    })

# Initialize CatBoost model
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model_advanced.cbm')
    model = CatBoostRegressor()
    model.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

@app.route('/predict_revenue', methods=['POST'])
def predict_revenue():
    try:
        title = request.json.get('title')
        if not title:
            return jsonify({'error': 'Movie title is required'}), 400
            
        movie_data = gather_movie_data(title)
        df = prepare_features(movie_data)
        cat_features, num_features = get_feature_lists()
        
        features = cat_features + num_features
        prediction_df = df[features]
        
        for feature in cat_features:
            if feature in df.columns:
                prediction_df[feature] = prediction_df[feature].astype(str)
        
        prediction_pool = Pool(prediction_df, cat_features=cat_features)
        predicted_revenue = float(np.expm1(model.predict(prediction_pool)[0]))
        
        return jsonify({
            'revenue': predicted_revenue,
            'isEstimate': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)