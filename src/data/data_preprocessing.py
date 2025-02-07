# src/data/data_preprocessing.py

import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
from pathlib import Path

# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download required NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment

def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets using pathlib."""
    try:
        # Convert data_path to a Path object and define the processed directory
        processed_data_path = Path(data_path) / "processed"
        
        # Create the 'processed' directory if it doesn't exist
        logger.debug(f"Creating directory {processed_data_path}")
        processed_data_path.mkdir(parents=True, exist_ok=True)  # This creates the directory
        
        logger.debug(f"Directory {processed_data_path} created or already exists")

        # Save the processed data
        train_data.to_csv(processed_data_path / "train_processed.csv", index=False)
        test_data.to_csv(processed_data_path / "test_processed.csv", index=False)
        
        logger.debug(f"Processed data saved to {processed_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

data_path = Path(__file__).resolve().parent.parent / "data"

def main():
    try:
        logger.debug("Starting data preprocessing...")

        # Define raw data path using pathlib
        raw_data_path = Path("./data/raw")

        # Load raw data
        train_data = pd.read_csv(raw_data_path / "train.csv")
        test_data = pd.read_csv(raw_data_path / "test.csv")
        logger.debug("Data loaded successfully")

        # Preprocess the data (Ensure `normalize_text` function exists)
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save the processed data using pathlib
        save_data(train_processed_data, test_processed_data, data_path="./data/")

    except Exception as e:
        logger.error("Failed to complete the data preprocessing process: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()