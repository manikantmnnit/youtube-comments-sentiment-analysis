#%%
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
from pathlib import Path

# Ensure required nltk datasets are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (consider restricting in production)

# Set MLflow Tracking URI (DagsHub)
mlflow.set_tracking_uri("https://dagshub.com/manikantmnnit/youtube-comments-sentiment-analysis.mlflow")

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
root_path = Path(__file__).resolve().parent.parent
vectorizer_path=root_path/"tfidf_vectorizer.pkl"
# Initialize model
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model_updated", "1", vectorizer_path)
#%%
comments=["I love India"," I dont like vegetables"]
preprocessed_comments = [preprocess_comment(comment) for comment in comments]
transformed_comments = vectorizer.transform(preprocessed_comments).toarray()

# Convert to DataFrame with correct column names
feature_names = vectorizer.get_feature_names_out()
df_transformed = pd.DataFrame(transformed_comments, columns=feature_names)

# Predict using the model
predictions = model.predict(df_transformed).tolist()# predictions = [str(pred) for pred in predictions]

#%%
@app.route('/')
def home():
    return "Welcome to our Flask API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400  

    try:
        

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    
    return jsonify(response)

if __name__ == '__main__':
    # For development:
    app.run(host='0.0.0.0', port=8000)