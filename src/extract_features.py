import pandas as pd
from utils.save_data import save_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def calculate_rolling_stats(data, column, windows, operations):
    """Calculate rolling statistics for specified windows and operations on a given column."""
    for window in windows:
        for op in operations:
            target_column_name = f"{column}_{window}_{op}"
            try:
                if op == "mean":
                    data[target_column_name] = data[column].rolling(window=window, min_periods=1).mean()
                elif op == "std":
                    data[target_column_name] = data[column].rolling(window=window, min_periods=1).std().bfill()
                elif op == "var":
                    data[target_column_name] = data[column].rolling(window=window, min_periods=1).var().bfill()
            except Exception as e:
                logging.error(f"Error calculating {op} for {target_column_name}: {e}")
    return data

def calculate_order_flow_imbalance(data):
    """Calculate the order flow imbalance over a specified rolling window."""
    try:
        data["signed_size"] = data["size"] * data["side_buy"].replace({True: 1, False: -1})
        data["order_flow_imbalance"] = data["signed_size"].rolling(window=10, min_periods=1).sum()
    except Exception as e:
        logging.error(f"Error calculating order flow imbalance: {e}")
    return data

def add_cancellation_ratio(data):
    """Add the ratio of cancelled orders to received orders as a feature."""
    # Replace 0 with 1 in 'type_received' to avoid division by zero
    data['type_received_adjusted'] = data['type_received'].replace(0, 1)
    
    # Perform the division ensuring float output
    data["cancel_to_received_ratio"] = data["reason_canceled"].astype(float) / data['type_received_adjusted']

    # Clean up by dropping the temporary column
    data.drop(columns='type_received_adjusted', inplace=True)
    
    return data

def market_spread(data):
    """Calculate the market spread from best bid and best ask."""
    try:
        data["spread"] = data["best_ask"] - data["best_bid"]
    except Exception as e:
        logging.error(f"Error calculating market spread: {e}")
    return data

def encode_hour_of_day(data):
    """One-hot encode the 'hour_of_day' column."""
    try:
        encoded_hours = pd.get_dummies(data["hour_of_day"], prefix="hour")
        data = pd.concat([data, encoded_hours], axis=1)
    except Exception as e:
        logging.error(f"Error encoding hour of day: {e}")
    return data

def extract_full_channel_features(data):
    logging.info("Extracting full channel features...")
    data = calculate_rolling_stats(data, "price", [5, 10, 15], ["mean", "std"])
    data = calculate_rolling_stats(data, "size", [5, 10, 15], ["mean", "std"])
    data = calculate_order_flow_imbalance(data)
    data = add_cancellation_ratio(data)
    data = encode_hour_of_day(data)
    return data

def extract_ticker_features(data):
    logging.info("Extracting ticker features...")
    data = market_spread(data)
    data = calculate_rolling_stats(data, "last_size", [5, 10, 15], ["mean", "var"])
    data = encode_hour_of_day(data)
    return data

def extract_features():
    logging.info("Loading data...")
    full_channel = pd.read_csv("../data/processed/full_channel_processed.csv")
    ticker = pd.read_csv("../data/processed/ticker_processed.csv")

    enhanced_full_channel = extract_full_channel_features(full_channel)
    enhanced_ticker = extract_ticker_features(ticker)

    save_data(enhanced_full_channel, "../data/processed/full_channel_enhanced.csv")
    save_data(enhanced_ticker, "../data/processed/ticker_enhanced.csv")
    logging.info("Feature extraction complete and files saved.")

# Test
if __name__ == "__main__":
    extract_features()
