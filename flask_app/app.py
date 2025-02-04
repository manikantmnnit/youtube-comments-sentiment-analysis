import os
import nltk
import re
import joblib
import mlflow
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify
from flask_cors import CORS
from wordcloud import WordCloud
from mlflow.tracking import MlflowClient
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required nltk datasets are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set MLflow Tracking URI (DagsHub)
mlflow.set_tracking_uri("https://dagshub.com/manikantmnnit/youtube-comments-sentiment-analysis.mlflow")
# os.environ['MLFLOW_TRACKING_USERNAME'] = "your_username"
# os.environ['MLFLOW_TRACKING_PASSWORD'] = "your_password"

# Define preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english'))
        stop_words -= {'not', 'but', 'however', 'no', 'yet'}

        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return comment

# Load model and vectorizer
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    """Load MLflow model and TF-IDF vectorizer."""
    try:
        client = MlflowClient()
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        vectorizer = joblib.load(vectorizer_path)  # Load the vectorizer
        return model, vectorizer
    except Exception as e:
        print(f"Error loading model/vectorizer: {e}")
        return None, None

# Initialize model
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model_updated", "1", "./tfidf_vectorizer.pkl")

@app.route('/')
def home():
    return "Welcome to our Flask API"

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    """Predict sentiment from comments with timestamps."""
    if model is None or vectorizer is None:
        return jsonify({"error": "Model/vectorizer not loaded properly"}), 500

    data = request.json
    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        predictions = model.predict(transformed_comments).tolist()
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} 
                for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
