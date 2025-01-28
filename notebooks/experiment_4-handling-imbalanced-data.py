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


import dagshub

dagshub.init(repo_owner='manikantmnnit', repo_name='youtube-comments-sentiment-analysis', mlflow=True)

# log the model

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/manikantmnnit/youtube-comments-sentiment-analysis.mlflow")

# set experiment name
mlflow.set_experiment("EXp -4:RF with TFIDF (1,3) under handling imbalanced data")

# Load the data
root_path=Path(__file__).parent.parent
df = pd.read_csv(root_path/'data'/'processed'/'processed_data.csv')
# %%

def run_imbalanced_experiment_alternative(imbalance_method):
    ngram_range = (1, 3)  # Trigram setting
    max_features = 500  # Set max_features to 500 for TF-IDF

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_comment'], df['category'], test_size=0.2, random_state=42, stratify=df['category'])

    # Vectorization using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Handle class imbalance
    if imbalance_method == 'class_weights':
        class_weight = 'balanced'
    else:
        class_weight = None
        if imbalance_method == 'oversampling':
            # Oversample minority classes
            X_train_vec, y_train = resample_minority(X_train_vec, y_train)
        elif imbalance_method == 'undersampling':
            # Undersample majority classes
            X_train_vec, y_train = resample_majority(X_train_vec, y_train)

    # Define and train the model
    with mlflow.start_run() as run:
        mlflow.set_tag("mlflow.runName", f"Imbalance_{imbalance_method}_RandomForest_TFIDF_Trigrams")
        mlflow.set_tag("experiment_type", "imbalance_handling")
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("description", f"RandomForest with TF-IDF Trigrams, imbalance handling method={imbalance_method}")
        
        # Log vectorizer parameters
        mlflow.log_param("vectorizer_type", "TF-IDF")
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("vectorizer_max_features", max_features)

        # Random Forest parameters
        n_estimators = 200
        max_depth = 15
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("imbalance_method", imbalance_method)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, class_weight=class_weight)
        model.fit(X_train_vec, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Log classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in classification_rep.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        # Log confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix: TF-IDF Trigrams, Imbalance={imbalance_method}")
        confusion_matrix_filename = f"confusion_matrix_{imbalance_method}.png"
        plt.savefig(confusion_matrix_filename)
        mlflow.log_artifact(confusion_matrix_filename)
        plt.close()

        # Log the model
        mlflow.sklearn.log_model(model, f"random_forest_model_tfidf_trigrams_imbalance_{imbalance_method}")

def resample_minority(X, y):
    """Basic oversampling by duplicating minority class samples."""
    df = pd.DataFrame(X.toarray())
    df['target'] = y
    max_size = df['target'].value_counts().max()

    oversampled = df.groupby('target', as_index=False).apply(lambda x: x.sample(max_size, replace=True)).reset_index(drop=True)
    return oversampled.drop(columns='target'), oversampled['target']

def resample_majority(X, y):
    """Basic undersampling by reducing majority class samples."""
    df = pd.DataFrame(X.toarray())
    df['target'] = y
    min_size = df['target'].value_counts().min()

    undersampled = df.groupby('target', as_index=False).apply(lambda x: x.sample(min_size, replace=False)).reset_index(drop=True)
    return undersampled.drop(columns='target'), undersampled['target']

# Run experiments
imbalance_methods = ['class_weights', 'oversampling', 'undersampling']
for method in imbalance_methods:
    run_imbalanced_experiment_alternative(method)