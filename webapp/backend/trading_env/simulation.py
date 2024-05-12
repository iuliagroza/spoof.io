import asyncio
from datetime import datetime
import json
from channels.layers import get_channel_layer
import pandas as pd
from trading_env.config import Config
from trading_env.preprocess_data import preprocess_full_channel_data, preprocess_ticker_data
from trading_env.extract_features import extract_full_channel_features, extract_ticker_features
from trading_env.market_env import MarketEnvironment
from trading_env.test import load_model, test_model
from trading_env.utils.save_data import save_data


async def send_order(order):
    channel_layer = get_channel_layer()
    # Convert DataFrame to a dictionary, replacing NaN with None
    order = {k: (None if pd.isna(v) else v) for k, v in order.to_dict().items()}
    
    # Convert Timestamp or datetime objects to string
    for key, value in order.items():
        if isinstance(value, (pd.Timestamp, datetime)):
            # ISO format is a good choice as it's standard and includes the timezone
            order[key] = value.isoformat()

    await channel_layer.group_send(
        'order_group',
        {
            'type': 'order.message',
            'message': json.dumps(order)  # Serialize the dictionary to a JSON formatted string
        }
    )


def print_spoofing_attempts(data):
    # Filter for spoofing attempts where action == 1
    spoofing_attempts = data[data['actions'] == 1]

    # Check if there are any spoofing attempts
    if not spoofing_attempts.empty:
        print("Detected Spoofing Attempts:")
        # for index, row in spoofing_attempts.iterrows():
            # print(f"Order ID: {index}")
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
            full_channel_row_without_delay = full_channel_row.drop('delay')
            ticker_row_without_delay = ticker_row.drop('delay')
            full_channel_batch.append(full_channel_row_without_delay)
            ticker_batch.append(ticker_row_without_delay)

            # Send simulated order to frontend order box
            await send_order(full_channel_row.drop('delay'))

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
        print("End of data.")

    except Exception as e:
        print(f"An error occurred: {e}")


def start_simulation():
    simulate_market_data()

if __name__ == "__main__":
    start_simulation()
