import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from pathlib  import Path

# Logging configuration
logger = logging.getLogger('data_ingestion') # logger object
logger.setLevel(logging.DEBUG) # set the logging level to DEBUG

console_handler = logging.StreamHandler() # console handler
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)    

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter) # set the formatter for the console handler
file_handler.setFormatter(formatter) # set the formatter for the file handler

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str) -> dict:
    """Load parameters from the params.yaml file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            logger.debug('Parameters loaded from %s', params_path)  # %s acts as a placeholder for params_path.
            return params 
    except FileNotFoundError as e:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Error reading the YAML file: %s', e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, and empty strings."""
    try:
        # Removing missing values
        df.dropna(inplace=True)
        # Removing duplicates
        df.drop_duplicates(inplace=True)
        # Removing rows with empty strings
        df = df[df['clean_comment'].str.strip() != '']
        
        logger.debug('Data preprocessing completed: Missing values, duplicates, and empty strings removed.')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: Path) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        raw_data_path = data_path / "raw"  # Use Path instead of os.path.join
        
        # Create the data/raw directory if it does not exist
        raw_data_path.mkdir(parents=True, exist_ok=True)
        
        # Save the train and test data
        train_data.to_csv(raw_data_path / "train.csv", index=False)
        test_data.to_csv(raw_data_path / "test.csv", index=False)
        
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

# Define data_path inside main or wherever it's called
data_path = Path(__file__).parent.parent.parent / "src" / "data"


def main():
    try:
        # Define the base directory (project root)
        base_dir = Path(__file__).resolve().parent.parent.parent  # Moves up 3 levels
        params_path = base_dir / "params.yaml"
        data_path = base_dir / "data"  # Define data path correctly
        
        # Load parameters
        params = load_params(params_path=params_path)
        test_size = params['data_ingestion']['test_size']
        
        # Load data
        df = load_data(data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
        
        # Preprocess the data
        final_df = preprocess_data(df)
        
        # Split data
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        
        # Save train and test datasets
        save_data(train_data, test_data, data_path)

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()