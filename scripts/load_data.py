import pandas as pd

# Function to load a JSON lines file
def load_json(file_path):
    return pd.read_json(file_path, lines=True)

# File paths for FullChannel and Ticker files
full_channel_files = ["data/raw/FullChannel_GDAX_20220511_17hr.json", "data/raw/FullChannel_GDAX_20220511_19hr.json", "data/raw/FullChannel_GDAX_20220511_20hr.json"]
ticker_files = ["data/raw/Ticker_GDAX_20220511_17hr.json", "data/raw/Ticker_GDAX_20220511_19hr.json", "data/raw/Ticker_GDAX_20220511_20hr.json"]

# Loading data into pandas DataFrames: each list contains three DataFrames
# (one for each file).
full_channel_data = [load_json(file) for file in full_channel_files]
ticker_data = [load_json(file) for file in ticker_files]

# Concatenate data
combined_full_channel = pd.concat(full_channel_data, ignore_index=True)
combined_ticker = pd.concat(ticker_data, ignore_index=True)

