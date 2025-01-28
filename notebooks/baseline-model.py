
import numpy as np


import pandas as pd
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
import mlflow

# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
df.head()

df.dropna(inplace=True)

df.drop_duplicates(inplace=True)

df=df[~(df['clean_comment'].str.strip()=='')]

nltk.download('wordnet')
nltk.download('stopwords')

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_comments(df, column_name):
    # Convert the column to lowercase
    df[column_name] = df[column_name].str.lower()
    
    # Count comments with leading or trailing spaces (optional for debugging/logging)
    leading_trailing_space_count = df[column_name].apply(lambda x: str(x).endswith(' ') or str(x).startswith(' ')).sum()
    print(f"Number of comments with leading/trailing spaces: {leading_trailing_space_count}")
    
    # Strip leading and trailing spaces
    df[column_name] = df[column_name].apply(lambda x: str(x).strip())
    
    # Dropping comments containing URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    df = df[~df[column_name].str.contains(url_pattern, regex=True, na=False)]  # Keep rows without URLs
    
    # Deleting comments containing newline characters
    df = df[~df[column_name].str.contains('\n', regex=False)]  # Keep rows without newline characters

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Define stopwords and retain important ones for sentiment analysis
    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    
    # Remove stopwords and lemmatize
    def clean_text(comment):
        # Remove stopwords
        words = [word for word in comment.split() if word not in stop_words]
        # Lemmatize words
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    # Apply cleaning function to the column
    df[column_name] = df[column_name].apply(clean_text)
    
    # Returning the modified DataFrame
    return df

# Preprocess the comments
vectorizer=CountVectorizer(max_features=10000)
X=vectorizer.fit_transform(df['clean_comment']).toarray()
y=df['category']

# initialize dagshub
import dagshub

dagshub.init(repo_owner='manikantmnnit', repo_name='youtube-comments-sentiment-analysis', mlflow=True)

# log the model

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/manikantmnnit/youtube-comments-sentiment-analysis.mlflow")

# set experiment name
mlflow.set_experiment("EXp -1:RF baseline")



# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


import mlflow
with mlflow.start_run() as run:
    # log a description of the run
    mlflow.set_tag('description', 'Randomforest base line model')
    mlflow.set_tag ('model', 'Randomforest classifier')
    mlflow.set_tag('experiment type',"baseline")

    # Add a description
    mlflow.set_tag("description", "Baseline RandomForest model for sentiment analysis using Bag of Words (BoW) with a simple train-test split")

    # log parameters for the vectorizer
    mlflow.log_param('max_features', 10000)
    mlflow.log_param('vectorizer_type', "countervectorizer")
    mlflow.log_param('model', 'RandomForestClassifier')

    n_estimators = 1000
    max_depth = 15

    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)

    # # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)

    accuracy=accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    mlflow.log_metric('accuracy', accuracy)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    plt.savefig('confusion_matrix.png')

    mlflow.log_image('confusion_matrix', 'confusion_matrix.png')    

# Evaluation
print(classification_report(y_test, y_pred))


