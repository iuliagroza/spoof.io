import pandas as pd
import logging
import os
from ..config import Config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def save_data(dataframe, filename, directory=Config.PROCESSED_DATA_PATH):
    """
    Saves the given DataFrame to a CSV file in the specified directory,
    ensuring the directory exists and the DataFrame is not empty.
    
    Parameters:
        dataframe (pd.DataFrame): The data frame to save.
        filename (str): The name of the file (without path) where the data frame should be saved.
        directory (str): The directory to save the file. Default is Config.PROCESSED_DATA_PATH.
        
    Returns:
        None
    """
    if dataframe.empty:
        logging.warning(f"No data to save. The DataFrame is empty.")
        return

    full_path = os.path.join(directory, filename)

    # Ensure directory exists
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    try:
        dataframe.to_csv(full_path, index=False)
        logging.info(f"Data successfully saved to {full_path}")
    except pd.errors.EmptyDataError:
        logging.error(f"No data to write to file: {full_path}")
    except IOError as e:
        logging.error(f"IOError when trying to write file {full_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while saving data to {full_path}: {e}")

# Test
if __name__ == "__main__":
    df = pd.DataFrame({'column1': [1, 2, 3], 'column2': ['A', 'B', 'C']})
    save_data(df, 'test_output_data.csv', 'data/misc/')