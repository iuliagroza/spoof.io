import pandas as pd
import logging
import os

# Basic logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_json(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return pd.DataFrame()
    
    try:
        return pd.read_json(file_path, lines=True)
    except ValueError as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

def load_data():
    """
    Loads data into pandas DataFrames from JSON files specified in environment variables or defaults.
    
    Returns: 
        combined_full_channel (pd.DataFrame): Concatenated DataFrame for FullChannel data.
        combined_ticker (pd.DataFrame): Concatenated DataFrame for Ticker data.
    """
    # Environment variables for file paths or default paths
    full_channel_files = os.getenv("FULL_CHANNEL_FILES", "../data/raw/FullChannel_GDAX_20220511_17hr.json,../data/raw/FullChannel_GDAX_20220511_19hr.json,../data/raw/FullChannel_GDAX_20220511_20hr.json").split(",")
    ticker_files = os.getenv("TICKER_FILES", "../data/raw/Ticker_GDAX_20220511_17hr.json,../data/raw/Ticker_GDAX_20220511_19hr.json,../data/raw/Ticker_GDAX_20220511_20hr.json").split(",")

    full_channel_data = [load_json(file) for file in full_channel_files if os.path.exists(file)]
    ticker_data = [load_json(file) for file in ticker_files if os.path.exists(file)]

    # Concatenate data
    combined_full_channel = pd.concat(full_channel_data, ignore_index=True) if full_channel_data else pd.DataFrame()
    combined_ticker = pd.concat(ticker_data, ignore_index=True) if ticker_data else pd.DataFrame()

    if combined_full_channel.empty:
        logging.warning("No FullChannel data loaded.")
    else:
        logging.info(f"Loaded FullChannel data with {combined_full_channel.shape[0]} rows and {combined_full_channel.shape[1]} columns.")

    if combined_ticker.empty:
        logging.warning("No Ticker data loaded.")
    else:
        logging.info(f"Loaded Ticker data with {combined_ticker.shape[0]} rows and {combined_ticker.shape[1]} columns.")

    return combined_full_channel, combined_ticker

# Test
if __name__ == "__main__":
    full_channel, ticker = load_data()
    print(full_channel.head())
    print(ticker.head())
