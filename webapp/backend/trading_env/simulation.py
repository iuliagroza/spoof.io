import pandas as pd
import asyncio
from src.config import Config
from src.preprocess_data import preprocess_full_channel_data, preprocess_ticker_data
from src.extract_features import extract_full_channel_features, extract_ticker_features
from src.market_env import MarketEnvironment
from src.test import load_model, test_model
from src.utils.save_data import save_data


def print_spoofing_attempts(data):
    # Filter for spoofing attempts where action == 1
    spoofing_attempts = data[data['actions'] == 1]

    # Check if there are any spoofing attempts
    if not spoofing_attempts.empty:
        print("Detected Spoofing Attempts:")
        for index, row in spoofing_attempts.iterrows():
            print(f"Order ID: {index}")
            # print(f"Features (States): {row['states']}")
            # print(f"Anomaly Score: {row['anomaly_scores']}")
            # print(f"Threshold: {row['spoofing_thresholds']}\n")
    else:
        print("No spoofing attempts detected in this batch.")


async def simulate_market_data():
    # Load the data
    full_channel_data = pd.read_csv(Config.FULL_CHANNEL_SIM_PATH)
    ticker_data = pd.read_csv(Config.TICKER_SIM_PATH)

    # Convert 'time' to datetime and sort by it if necessary
    full_channel_data['time'] = pd.to_datetime(full_channel_data['time'])
    ticker_data['time'] = pd.to_datetime(ticker_data['time'])

    # Calculate the delays
    full_channel_data['delay'] = full_channel_data['time'].diff().dt.total_seconds().fillna(0)
    ticker_data['delay'] = ticker_data['time'].diff().dt.total_seconds().fillna(0)

    # Initialize batches
    full_channel_batch = []
    ticker_batch = []

    # Start processing each row as a 'real-time' feed
    full_channel_iter = full_channel_data.iterrows()
    ticker_iter = ticker_data.iterrows()

    try:
        while True:
            # Get next entry from each data source
            _, full_channel_row = next(full_channel_iter)
            _, ticker_row = next(ticker_iter)

            # Sleep to simulate real-time data feed
            await asyncio.sleep(full_channel_row['delay'])
            await asyncio.sleep(ticker_row['delay'])

            # Append to batch without the 'delay' column
            full_channel_batch.append(full_channel_row.drop('delay'))
            ticker_batch.append(ticker_row.drop('delay'))

            # If batch is ready, process it
            if len(full_channel_batch) == Config.BATCH_SIZE:
                full_channel_df = pd.DataFrame(full_channel_batch)
                ticker_df = pd.DataFrame(ticker_batch)
                save_data(full_channel_df, ticker_df, Config.MISC_DATA_PATH + 'full_channel_df.csv', Config.MISC_DATA_PATH + 'ticker_df.csv')

                # Preprocess and feature engineer
                processed_full_channel = preprocess_full_channel_data(full_channel_df)
                processed_ticker = preprocess_ticker_data(ticker_df)
                save_data(processed_full_channel, processed_ticker, Config.MISC_DATA_PATH + 'full_channel_prep.csv', Config.MISC_DATA_PATH + 'ticker_prep.csv')
                enhanced_full_channel = extract_full_channel_features(processed_full_channel)
                enhanced_ticker = extract_ticker_features(processed_ticker)

                save_data(enhanced_full_channel, enhanced_ticker, Config.MISC_DATA_PATH + 'full_channel_output.csv', Config.MISC_DATA_PATH + 'ticker_output.csv')

                env = MarketEnvironment(initial_index=0, full_channel_data=enhanced_full_channel, ticker_data=enhanced_ticker, train=False)
                model = load_model(Config.PPO_POLICY_NETWORK_MODEL_PATH, len(env.reset()), 2)
                data = test_model(env, model)
                print_spoofing_attempts(data)

                # Reset batches
                full_channel_batch = []
                ticker_batch = []

    except StopIteration:
        pass  # End of data

    except Exception as e:
        print(f"An error occurred: {e}")

def start_simulation():
    asyncio.run(simulate_market_data())

if __name__ == "__main__":
    start_simulation()
