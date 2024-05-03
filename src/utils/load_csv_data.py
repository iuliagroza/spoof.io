import pandas as pd
from src.utils.log_config import setup_logger
from src.config import Config


logger = setup_logger('load_csv_data')


def load_csv_data(full_channel_path, ticker_path):
    """
    Loads CSV data for the full channel and ticker datasets.

    Args:
        full_channel_path (str): Path to the full channel CSV file.
        ticker_path (str): Path to the ticker CSV file.

    Returns:
        tuple: A tuple containing two pandas DataFrames (full_channel, ticker).
    """
    try:
        # Load full channel data from the CSV
        full_channel = pd.read_csv(full_channel_path)
        logger.info(f"Full channel data loaded successfully from {full_channel_path}")
    except Exception as e:
        logger.error(f"Error loading full channel data from {full_channel_path}: {e}")
        full_channel = pd.DataFrame()

    try:
        # Load ticker data from the CSV
        ticker = pd.read_csv(ticker_path)
        logger.info(f"Ticker data loaded successfully from {ticker_path}")
    except Exception as e:
        logger.error(f"Error loading ticker data from {ticker_path}: {e}")
        ticker = pd.DataFrame()

    return full_channel, ticker
    

# Test
if __name__ == "__main__":
    full_channel, ticker = load_csv_data(Config.FULL_CHANNEL_PROCESSED_PATH, Config.TICKER_PROCESSED_PATH)
    print(full_channel.head())
    print(ticker.head())
