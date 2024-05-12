import pandas as pd
import os
from trading_env.utils.log_config import setup_logger
from trading_env.config import Config


logger = setup_logger('load_json_data')


def load_json_file(file_path):
    """
    Attempts to load JSON data from a specified file path.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        DataFrame: Pandas DataFrame containing the loaded data, or an empty DataFrame if an error occurs.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return pd.DataFrame()
    
    try:
        return pd.read_json(file_path, lines=True)
    except ValueError as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
        return pd.DataFrame()


def load_json_data(full_channel_files=None, ticker_files=None):
    """
    Loads data into pandas DataFrames from JSON files specified in environment variables or defaults.
    
    Returns: 
        combined_full_channel (pd.DataFrame): Concatenated DataFrame for FullChannel data.
        combined_ticker (pd.DataFrame): Concatenated DataFrame for Ticker data.
    """
    if full_channel_files is None:
        full_channel_files = [
            Config.FULL_CHANNEL_RAW_17HR_PATH,
            Config.FULL_CHANNEL_RAW_19HR_PATH,
            Config.FULL_CHANNEL_RAW_20HR_PATH
        ]
    
    if ticker_files is None:
        ticker_files = [
            Config.TICKER_RAW_17HR_PATH,
            Config.TICKER_RAW_19HR_PATH,
            Config.TICKER_RAW_20HR_PATH
        ]
    full_channel_data = [load_json_file(file) for file in full_channel_files if os.path.exists(file)]
    ticker_data = [load_json_file(file) for file in ticker_files if os.path.exists(file)]

    # Concatenate data
    combined_full_channel = pd.concat(full_channel_data, ignore_index=True) if full_channel_data else pd.DataFrame()
    combined_ticker = pd.concat(ticker_data, ignore_index=True) if ticker_data else pd.DataFrame()

    if combined_full_channel.empty:
        logger.warning("No FullChannel data loaded.")
    else:
        logger.info(f"Loaded FullChannel data with {combined_full_channel.shape[0]} rows and {combined_full_channel.shape[1]} columns.")

    if combined_ticker.empty:
        logger.warning("No Ticker data loaded.")
    else:
        logger.info(f"Loaded Ticker data with {combined_ticker.shape[0]} rows and {combined_ticker.shape[1]} columns.")

    return combined_full_channel, combined_ticker


# Test
if __name__ == "__main__":
    # Default load
    full_channel, ticker = load_json_data()
    print(full_channel.head())
    print(ticker.head())

    # Load with specific files
    full_channel_specific_files = [
        Config.FULL_CHANNEL_RAW_19HR_PATH
    ]
    ticker_specific_files = [
        Config.TICKER_RAW_19HR_PATH
    ]
    full_channel_specific, ticker_specific = load_json_data(full_channel_specific_files, ticker_specific_files)
    print(full_channel_specific.head())
    print(ticker_specific.head())
