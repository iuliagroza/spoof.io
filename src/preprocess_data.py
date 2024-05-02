import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from utils.load_data import load_data
from utils.save_data import save_data
from config import Config
import logging


logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)


def add_time_features(data):
    """ 
    Adds time-based features to a DataFrame.
    """
    data["time"] = pd.to_datetime(data["time"])
    data["hour_of_day"] = data["time"].dt.hour
    return data


def create_numeric_transformer():
    """ 
    Returns a pipeline for numeric transformations. 
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])


def create_categorical_transformer():
    """ 
    Returns a pipeline for categorical transformations.
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])


def preprocess_full_channel_data(data):
    data = add_time_features(data)
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


def preprocess_ticker_data(data):
    data = add_time_features(data)
    data.drop(columns=["type", "product_id", "low_24h"], inplace=True)

    numeric_cols = ["price", "open_24h", "volume_24h", "high_24h", "volume_30d", "best_bid", "best_ask", "last_size"]
    data[numeric_cols] = MinMaxScaler().fit_transform(data[numeric_cols])

    data = pd.get_dummies(data, columns=["side"], drop_first=False)
    return data


def preprocess_data():
    full_channel, ticker = load_data(Config.RAW_DATA_PATH)
    full_channel_processed = preprocess_full_channel_data(full_channel)
    ticker_processed = preprocess_ticker_data(ticker)

    save_data(full_channel_processed, Config.PROCESSED_DATA_PATH + 'full_channel_processed.csv')
    save_data(ticker_processed, Config.PROCESSED_DATA_PATH + 'ticker_processed.csv')
    logging.info("Data preprocessing complete and files saved.")


# Test
if __name__ == "__main__":
    preprocess_data()
    