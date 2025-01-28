#%%
import mlflow

import pandas as pd
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

import optuna
import numpy as np

import dagshub

dagshub.init(repo_owner='manikantmnnit', repo_name='youtube-comments-sentiment-analysis', mlflow=True)

# log the model

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/manikantmnnit/youtube-comments-sentiment-analysis.mlflow")

# set experiment name
mlflow.set_experiment("EXp -5:ML algos with hyperparameter tuning")

# Load the data
root_path=Path(__file__).parent.parent
df = pd.read_csv(root_path/'data'/'processed'/'processed_data.csv')

# Step 1: Remap the class labels from [-1, 0, 1] to [2, 0, 1]
df['category'] = df['category'].map({-1: 2, 0: 0, 1: 1})


#%%
# Step 1: TF-IDF vectorizer setup

# Step 3: TF-IDF vectorizer setup
ngram_range = (1, 3)  # Trigram
max_features = 500  # Set max_features to 500
vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
X = vectorizer.fit_transform(df['clean_comment'])
y = df['category']

# Step 4: Apply SMOTE to handle class imbalance (if needed)
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Function to log results in MLflow
def log_mlflow(model_name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        # Log model type
        mlflow.set_tag("mlflow.runName", f"{model_name}_KNN_TFIDF_Trigrams")
        mlflow.set_tag("experiment_type", "algorithm_comparison")

        # Log algorithm name as a parameter
        mlflow.log_param("algo_name", model_name)

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Log accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Log classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in classification_rep.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        # Log the model
        mlflow.sklearn.log_model(model, f"{model_name}_model")


# Step 6: Optuna objective function for KNN
def objective_knn(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 20)  # KNN Hyperparameter (number of neighbors)
    p = trial.suggest_int('p', 1, 2)  # Distance metric (1 for Manhattan, 2 for Euclidean)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])  # Weight function for prediction

    # KNN model setup
    model = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, weights=weights)
    return accuracy_score(y_test, model.fit(X_train, y_train).predict(X_test))


# Step 7: Run Optuna for KNN, log the best model only
def run_optuna_experiment():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_knn, n_trials=30)

    # Get the best parameters and log only the best model
    best_params = study.best_params
    best_model = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'],
                                      p=best_params['p'],
                                      weights=best_params['weights'])

    # Log the best model with MLflow, passing the algo_name as "KNN"
    log_mlflow("KNN", best_model, X_train, X_test, y_train, y_test)

# Run the experiment for KNN
run_optuna_experiment()
# %%
