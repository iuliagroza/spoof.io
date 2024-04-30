import pandas as pd
import logging

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json(file_path):
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
    Loads data into pandas DataFrames: each list contains three DataFrames (one for each file).
    
    Parameters:
        None

    Returns: 
        combined_full_channel (pd.DataFrame): The list of three DataFrames for FullChannel.
        combined_ticker (pd.DataFrame): The list of three DataFrames for Ticker.
    """
    # File paths for FullChannel and Ticker files
    full_channel_files = ["../data/raw/FullChannel_GDAX_20220511_17hr.json", "../data/raw/FullChannel_GDAX_20220511_19hr.json", "../data/raw/FullChannel_GDAX_20220511_20hr.json"]
    ticker_files = ["../data/raw/Ticker_GDAX_20220511_17hr.json", "../data/raw/Ticker_GDAX_20220511_19hr.json", "../data/raw/Ticker_GDAX_20220511_20hr.json"]

    full_channel_data = [load_json(file) for file in full_channel_files]
    ticker_data = [load_json(file) for file in ticker_files]

    if any(df.empty for df in full_channel_data + ticker_data):
        logging.warning("One or more files failed to load, check logs for details.")

    # Concatenate data
    combined_full_channel = pd.concat(full_channel_data, ignore_index=True) if full_channel_data else pd.DataFrame()
    combined_ticker = pd.concat(ticker_data, ignore_index=True) if ticker_data else pd.DataFrame()

    if not combined_full_channel.empty:
        logging.info(f"Loaded FullChannel data with {combined_full_channel.shape[0]} rows and {combined_full_channel.shape[1]} columns.")
    if not combined_ticker.empty:
        logging.info(f"Loaded Ticker data with {combined_ticker.shape[0]} rows and {combined_ticker.shape[1]} columns.")

    return combined_full_channel, combined_ticker

# Test
if __name__ == "__main__":
    full_channel, ticker = load_data()
    print(full_channel.head())
    print(ticker.head())
