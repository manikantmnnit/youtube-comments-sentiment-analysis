#%%
import mlflow

import pandas as pd
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.naive_bayes import MultinomialNB
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
ngram_range = (1, 3)  # Trigram
max_features = 500  # Limit vocabulary size
vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
X = vectorizer.fit_transform(df['clean_comment'])
y = df['category']

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Function to log results in MLflow
def log_mlflow(model_name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        # Log model type
        mlflow.set_tag("mlflow.runName", f"{model_name}_ClassWeight_TFIDF_Trigrams")
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


# Step 4: Optuna objective function for Multinomial Naive Bayes
def objective_mnb(trial):
    alpha = trial.suggest_float('alpha', 1e-4, 1.0, log=True)  # Tuning the smoothing parameter

    # MultinomialNB model setup with class weights
    model = MultinomialNB(alpha=alpha)
    sample_weights = np.array([class_weights_dict[label] for label in y_train])  # Apply weights
    model.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)


# Step 5: Run Optuna for Multinomial Naive Bayes, log the best model only
def run_optuna_experiment():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_mnb, n_trials=30)

    # Get the best parameters and log only the best model
    best_params = study.best_params
    best_model = MultinomialNB(alpha=best_params['alpha'])

    # Log the best model with MLflow
    log_mlflow("MultinomialNB", best_model, X_train, X_test, y_train, y_test)

# Run the experiment for Multinomial Naive Bayes
run_optuna_experiment()
# %%
