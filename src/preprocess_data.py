import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils.load_data import load_data
from src.utils.save_data import save_data
from src.config import Config
import logging


logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)


def add_time_features(data):
    """ 
    Adds time-based features to a DataFrame by extracting hour of the day from the timestamp.
    
    Parameters:
        data (pd.DataFrame): The DataFrame with a 'time' column in datetime format.
    
    Returns:
        pd.DataFrame: The DataFrame with an additional 'hour_of_day' feature.
    """
    try:
        data["time"] = pd.to_datetime(data["time"])
        data["hour_of_day"] = data["time"].dt.hour
        return data
    except KeyError as e:
        logging.error(f"Error: The 'time' column is missing. {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while adding time features. {e}")
        return None


def create_numeric_transformer():
    """ 
    Creates a pipeline for processing numeric features. This includes imputation of missing values with the median and scaling features between 0 and 1.
    
    Returns:
        Pipeline: A configured pipeline for numeric feature transformations.
    """
    try:
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ])
    except Exception as e:
        logging.error(f"An error occurred while creating the numeric transformer. {e}")
        return None


def create_categorical_transformer():
    """ 
    Creates a pipeline for processing categorical features. This includes imputation of missing values with a placeholder and encoding of categorical features as one-hot vectors.
    
    Returns:
        Pipeline: A configured pipeline for categorical feature transformations.
    """
    try:
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
    except Exception as e:
        logging.error(f"An error occurred while creating the categorical transformer. {e}")
        return None


def preprocess_full_channel_data(data):
    """ 
    Processes the full channel data including adding time features, imputation, and encoding transformations.
    
    Parameters:
        data (pd.DataFrame): The DataFrame to process.
    
    Returns:
        pd.DataFrame: The preprocessed DataFrame ready for model consumption.
    """
    try:
        data = add_time_features(data)
        if data is None:
            return None
        data.fillna({"remaining_size": 0, "price": data["price"].mean()}, inplace=True)
        data["remaining_size_change"] = data.groupby("order_id")["remaining_size"].diff().fillna(0)

        numeric_features = Config.NUMERIC_COLUMNS
        categorical_features = Config.CATEGORICAL_COLUMNS

        preprocessor = ColumnTransformer(transformers=[
            ("num", create_numeric_transformer(), numeric_features),
            ("cat", create_categorical_transformer(), categorical_features)
        ])

        data_processed = pd.DataFrame(preprocessor.fit_transform(data))
        data_processed.columns = numeric_features + list(preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features))
        data_processed["hour_of_day"] = data["hour_of_day"].values

        return data_processed
    except KeyError as e:
        logging.error(f"Error: A required column is missing. {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while preprocessing full channel data. {e}")
        return None


def preprocess_ticker_data(data):
    """ 
    Processes the ticker data including adding time features, scaling, and one-hot encoding.
    
    Parameters:
        data (pd.DataFrame): The DataFrame to process.
    
    Returns:
        pd.DataFrame: The preprocessed DataFrame ready for model consumption.
    """
    try:
        data = add_time_features(data)
        if data is None:
            return None
        data.drop(columns=["type", "product_id", "low_24h"], inplace=True)

        numeric_cols = ["price", "open_24h", "volume_24h", "high_24h", "volume_30d", "best_bid", "best_ask", "last_size"]
        data[numeric_cols] = MinMaxScaler().fit_transform(data[numeric_cols])

        data = pd.get_dummies(data, columns=["side"], drop_first=False)
        return data
    except KeyError as e:
        logging.error(f"Error: A required column is missing. {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while preprocessing ticker data. {e}")
        return None
    

def preprocess_data():
    """ 
    Main function to load, process, and save the preprocessed full channel and ticker data.
    """
    try:
        full_channel, ticker = load_data()
    except Exception as e:
        logging.error(f"An error occurred while loading data. {e}")
        return

    full_channel_processed = preprocess_full_channel_data(full_channel)
    if full_channel_processed is None:
        logging.error("Full channel data preprocessing failed.")
        return

    ticker_processed = preprocess_ticker_data(ticker)
    if ticker_processed is None:
        logging.error("Ticker data preprocessing failed.")
        return

    try:
        save_data(full_channel_processed, Config.PROCESSED_DATA_PATH + 'full_channel_processed.csv')
        save_data(ticker_processed, Config.PROCESSED_DATA_PATH + 'ticker_processed.csv')
        logging.info("Data preprocessing complete and files saved.")
    except Exception as e:
        logging.error(f"An error occurred while saving data. {e}")


# Test
if __name__ == "__main__":
    preprocess_data()
    