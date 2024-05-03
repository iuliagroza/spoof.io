import pandas as pd
import os
from src.utils.log_config import setup_logger
from src.config import Config


logger = setup_logger()


def save_data(full_channel_df, ticker_df, full_channel_path, ticker_path):
    """
    Saves the given DataFrames for full_channel and ticker to CSV files in the specified paths.

    Args:
        full_channel_df (pd.DataFrame): DataFrame containing full channel data.
        ticker_df (pd.DataFrame): DataFrame containing ticker data.
        full_channel_path (str): Path to the full channel CSV file.
        ticker_path (str): Path to the ticker CSV file.
    """
    # Save full channel DataFrame
    if full_channel_df.empty:
        logger.warning(f"No data to save. The full channel DataFrame is empty.")
        return

    os.makedirs(os.path.dirname(full_channel_path), exist_ok=True)

    try:
        full_channel_df.to_csv(full_channel_path, index=False)
        logger.info(f"Full channel data successfully saved to {full_channel_path}")
    except pd.errors.EmptyDataError:
        logger.error(f"No data to write to file: {full_channel_path}")
    except IOError as e:
        logger.error(f"IOError when trying to write file {full_channel_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving data to {full_channel_path}: {e}")
    
    # Save ticker DataFrame
    if ticker_df.empty:
        logger.warning(f"No data to save. The ticker DataFrame is empty.")
        return

    os.makedirs(os.path.dirname(ticker_path), exist_ok=True)

    try:
        ticker_df.to_csv(ticker_path, index=False)
        logger.info(f"Ticker data successfully saved to {ticker_path}")
    except pd.errors.EmptyDataError:
        logger.error(f"No data to write to file: {ticker_path}")
    except IOError as e:
        logger.error(f"IOError when trying to write file {ticker_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving data to {ticker_path}: {e}")
 

# Test
if __name__ == "__main__":
    full_channel_df = pd.DataFrame({'time': ['2021-01-01', '2021-01-02'], 'price': [100, 101], 'volume': [200, 210]})
    ticker_df = pd.DataFrame({'time': ['2021-01-01', '2021-01-02'], 'last_price': [99, 98], 'last_volume': [190, 185]})

    save_data(full_channel_df, ticker_df, Config.MISC_DATA_PATH + 'full_channel_output.csv', Config.MISC_DATA_PATH + 'ticker_output.csv')
