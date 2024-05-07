import pandas as pd
import asyncio
from src.config import Config
from src.preprocess_data import preprocess_full_channel_data, preprocess_ticker_data
from src.extract_features import extract_full_channel_features, extract_ticker_features


async def simulate_market_data():
    # Load the data
    full_channel_data = pd.read_csv(Config.FULL_CHANNEL_SIM_PATH)
    ticker_data = pd.read_csv(Config.TICKER_SIM_PATH)

    # Convert 'time' to datetime and sort by it
    full_channel_data['time'] = pd.to_datetime(full_channel_data['time'])
    ticker_data['time'] = pd.to_datetime(ticker_data['time'])

    # Initialize batches
    full_channel_batch = []
    ticker_batch = []

    # Start processing each row as a 'real-time' feed
    full_channel_iter = full_channel_data.iterrows()
    ticker_iter = ticker_data.iterrows()
    full_channel_prev_time = None
    ticker_prev_time = None

    try:
        while True:
            try:
                # Get next entry from each data source
                _, full_channel_row = next(full_channel_iter)
                _, ticker_row = next(ticker_iter)

                # Calculate the delay based on the time difference
                if full_channel_prev_time is not None:
                    full_channel_delay = (full_channel_row['time'] - full_channel_prev_time).total_seconds()
                    await asyncio.sleep(full_channel_delay)

                if ticker_prev_time is not None:
                    ticker_delay = (ticker_row['time'] - ticker_prev_time).total_seconds()
                    await asyncio.sleep(ticker_delay)

                # Append to batch
                full_channel_batch.append(full_channel_row)
                ticker_batch.append(ticker_row)

                # If batch is ready, process it
                if len(full_channel_batch) == Config.BATCH_SIZE:
                    full_channel_df = pd.DataFrame(full_channel_batch)
                    ticker_df = pd.DataFrame(ticker_batch)

                    # Preprocess and feature engineer
                    processed_full_channel = preprocess_full_channel_data(full_channel_df)
                    processed_ticker = preprocess_ticker_data(ticker_df)
                    enhanced_full_channel = extract_full_channel_features(processed_full_channel)
                    enhanced_ticker = extract_ticker_features(processed_ticker)

                    # Here you would feed it into the market environment or do further processing
                    # For now, let's just print the result
                    print(enhanced_full_channel.head())
                    print(enhanced_ticker.head())

                    # Reset batches
                    full_channel_batch = []
                    ticker_batch = []

                # Update previous time
                full_channel_prev_time = full_channel_row['time']
                ticker_prev_time = ticker_row['time']

            except StopIteration:
                break  # When no more data, exit loop

    except Exception as e:
        print(f"An error occurred: {e}")

def start_simulation():
    asyncio.run(simulate_market_data())

if __name__ == "__main__":
    start_simulation()
