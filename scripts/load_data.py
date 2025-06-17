import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file_path, delimeter):
    """
    Load data from a file and return a DataFrame.

    Parameters:
    - file_path (str): Path to the data file.
    - sample_size (int): Number of lines to read from the file.

    Returns:
    - pd.DataFrame: DataFrame containing the loaded data.
    """
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} does not exist.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path, delimiter=delimeter)
        logging.info(
            f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None


def load_csv(filepath):
    """
    Load a CSV file into a DataFrame.

    Parameters:
    - filepath (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"CSV file loaded successfully from {filepath}.")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV file from {filepath}: {e}")
        return None
