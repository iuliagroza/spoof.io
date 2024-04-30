import pandas as pd
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_data(dataframe, filename):
    """
    Saves the given DataFrame to a CSV file.
    
    Parameters:
        dataframe (pd.DataFrame): The data frame to save.
        filename (str): The path to the file where the data frame should be saved.
        
    Returns:
        None
    """
    try:
        dataframe.to_csv(filename, index=False)
        logging.info(f"Data successfully saved to {filename}")
    except Exception as e:
        logging.error(f"Failed to save data to {filename}: {e}")
