import pandas as pd
from src.utils.load_data import load_data
from src.utils.save_data import save_data
from src.config import Config
import logging


logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)


def calculate_rolling_stats(data, column):
    """
    Calculate rolling statistics for specified windows and operations on a given column.
    
    Args:
        data (pd.DataFrame): The dataset on which calculations will be performed.
        column (str): The column to calculate statistics on.
        windows (list): List of window sizes for rolling calculations.
        operations (list): List of operations (mean, std, var) to perform.
    
    Returns:
        data (pd.DataFrame): The dataset with new columns for each calculated statistic.
    """
    for window in Config.ROLLING_WINDOWS:
        for op in Config.OPERATIONS:
            target_column_name = f"{column}_{window}_{op}"
            try:
                if op == 'mean':
                    data[target_column_name] = data[column].rolling(window=window, min_periods=1).mean()
                elif op == 'std':
                    data[target_column_name] = data[column].rolling(window=window, min_periods=1).std().bfill()
                elif op == 'var':
                    data[target_column_name] = data[column].rolling(window=window, min_periods=1).var().bfill()
            except Exception as e:
                logging.error(f"Error calculating {op} for {target_column_name}: {e}")
    return data


def calculate_order_flow_imbalance(data):
    """
    Calculate the order flow imbalance over a specified rolling window.
    
    Args:
        data (pd.DataFrame): The dataset for which the imbalance will be calculated.
    
    Returns:
        data (pd.DataFrame): The dataset with the order flow imbalance added.
    """
    try:
        data['signed_size'] = data['size'] * data['side_buy'].replace({True: 1, False: -1})
        data['order_flow_imbalance'] = data['signed_size'].rolling(window=10, min_periods=1).sum()
    except Exception as e:
        logging.error(f"Error calculating order flow imbalance: {e}")
    return data


def add_cancellation_ratio(data):
    """
    Add the ratio of cancelled orders to received orders as a feature.
    
    Args:
        data (pd.DataFrame): The dataset where cancellation ratio needs to be added.
    
    Returns:
        data (pd.DataFrame): The dataset with cancellation ratio added.
    """
    try:
        data['type_received_adjusted'] = data['type_received'].replace(0, 1)
        data['cancel_to_received_ratio'] = data['reason_canceled'].astype(float) / data['type_received_adjusted']
        data.drop(columns='type_received_adjusted', inplace=True)
    except Exception as e:
        logging.error(f"Error adding cancellation ratio: {e}")
    return data


def market_spread(data):
    """
    Calculate the market spread from best bid and best ask.
    
    Args:
        data (pd.DataFrame): The dataset for which the market spread will be calculated.
    
    Returns:
        data (pd.DataFrame): The dataset with market spread added.
    """
    try:
        data['spread'] = data['best_ask'] - data['best_bid']
    except Exception as e:
        logging.error(f"Error calculating market spread: {e}")
    return data


def encode_hour_of_day(data):
    """
    One-hot encode the 'hour_of_day' column.
    
    Args:
        data (pd.DataFrame): The dataset for which the hour of day will be encoded.
    
    Returns:
        data (pd.DataFrame): The dataset with encoded hour of day.
    """
    try:
        encoded_hours = pd.get_dummies(data['hour_of_day'], prefix='hour')
        data = pd.concat([data, encoded_hours], axis=1)
    except Exception as e:
        logging.error(f"Error encoding hour of day: {e}")
    return data


def extract_full_channel_features(data):
    """
    Extract features from the full channel dataset.
    
    Args:
        data (pd.DataFrame): The full channel data.
    
    Returns:
        data (pd.DataFrame): The full channel data with new features added.
    """
    logging.info("Extracting full channel features...")
    data = calculate_rolling_stats(data, 'price')
    data = calculate_rolling_stats(data, 'size')
    data = calculate_order_flow_imbalance(data)
    data = add_cancellation_ratio(data)
    data = encode_hour_of_day(data)
    return data


def extract_ticker_features(data):
    """
    Extract features from the ticker dataset.
    
    Args:
        data (pd.DataFrame): The ticker data.
    
    Returns:
        data (pd.DataFrame): The ticker data with new features added.
    """
    logging.info("Extracting ticker features...")
    data = market_spread(data)
    data = calculate_rolling_stats(data, 'last_size')
    data = encode_hour_of_day(data)
    return data


def load_csv_data(file_path):
    """
    Load a CSV file into a DataFrame.

    Args:
        file_path (str): The complete path to the CSV file that needs to be loaded.

    Returns:
        pd.DataFrame: A DataFrame containing the data loaded from the CSV file.
                      Returns an empty DataFrame if the file cannot be loaded due to
                      any reason, such as the file not being found, being empty, or an error
                      during the loading process.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logging.error(f"No data in file: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return pd.DataFrame()


def extract_features():
    """
    Main function to execute feature extraction for both full channel and ticker data.
    Uses configurations from Config and leverages functions from load_data and save_data.
    """
    logging.info("Loading processed data...")
    full_channel = load_csv_data(Config.PROCESSED_DATA_PATH + 'full_channel_processed.csv')
    ticker = load_csv_data(Config.PROCESSED_DATA_PATH + 'ticker_processed.csv')

    logging.info("Extracting features for full channel data...")
    enhanced_full_channel = extract_full_channel_features(full_channel)
    logging.info("Extracting features for ticker data...")
    enhanced_ticker = extract_ticker_features(ticker)

    logging.info("Saving enhanced datasets...")
    save_data(enhanced_full_channel, 'full_channel_enhanced.csv')
    save_data(enhanced_ticker, 'ticker_enhanced.csv')
    logging.info("Feature extraction complete and files saved.")


# Test
if __name__ == "__main__":
    extract_features()
