import pandas as pd
import logging
import os

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_data(dataframe, filename):
    """
    Saves the given DataFrame to a CSV file, ensuring the directory exists and the DataFrame is not empty.
    
    Parameters:
        dataframe (pd.DataFrame): The data frame to save.
        filename (str): The path to the file where the data frame should be saved.
        
    Returns:
        None
    """
    if dataframe.empty:
        logging.warning(f"No data to save. The DataFrame is empty.")
        return

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    try:
        dataframe.to_csv(filename, index=False)
        logging.info(f"Data successfully saved to {filename}")
    except pd.errors.EmptyDataError:
        logging.error(f"No data to write to file: {filename}")
    except IOError as e:
        logging.error(f"IOError when trying to write file {filename}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while saving data to {filename}: {e}")

# Test
if __name__ == "__main__":
    # Create a DataFrame for testing
    df = pd.DataFrame({"column1": [1, 2, 3], "column2": ["A", "B", "C"]})
    save_data(df, "../data/misc/test_output_data.csv")
