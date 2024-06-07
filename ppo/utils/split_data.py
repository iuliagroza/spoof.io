import pandas as pd
from ppo.config import Config

def split_data(file_path, save_path):
    data = pd.read_csv(file_path)
    
    split_point = int(len(data) * 0.7)
    # Extract the last 30% of the data
    sim_data = data[split_point:]

    # Save the last 30% to a new CSV file
    sim_data.to_csv(save_path, index=False)

def main():
    full_channel_path = Config.FULL_CHANNEL_RAW_PATH
    ticker_path = Config.TICKER_RAW_PATH

    # Define paths for the new test datasets
    full_channel_sim_path = Config.FULL_CHANNEL_SIM_PATH
    ticker_sim_path = Config.TICKER_SIM_PATH

    # Split the full_channel and ticker datasets
    split_data(full_channel_path, full_channel_sim_path)
    split_data(ticker_path, ticker_sim_path)

if __name__ == '__main__':
    main()
