import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from utils.load_data import load_data
from utils.save_data import save_data

def add_time_features(data):
    # Convert 'time' to datetime and extract the hour
    data["time"] = pd.to_datetime(data["time"])
    data["hour_of_day"] = data["time"].dt.hour
    
    # Normalize hour, range [16, 20]
    data["hour_of_day"] = (data["hour_of_day"] - 16) / 4
    return data

def preprocess_full_channel_data(data):
    data = add_time_features(data)
    
    # Handling missing values
    data.fillna({"remaining_size": 0, "price": data["price"].mean()}, inplace=True)
    
    # Feature Engineering
    data["remaining_size_change"] = data.groupby("order_id")["remaining_size"].diff().fillna(0)
    
    # Normalize numeric columns and encode categorical variables
    numeric_features = ["price", "size", "remaining_size", "remaining_size_change", "hour_of_day"]
    categorical_features = ["type", "side", "reason"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler())
            ]), numeric_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_features)
        ])

    # Fit and transform data
    data_processed = preprocessor.fit_transform(data)
    new_columns = numeric_features + list(preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features))
    data_processed = pd.DataFrame(data_processed, columns=new_columns)
    
    return data_processed

def preprocess_ticker_data(data):
    data = add_time_features(data)
    
    data = data.drop(columns=["type", "product_id", "low_24h"])
    
    # Normalize the numeric columns
    scaler = MinMaxScaler()
    numeric_cols = ["price", "open_24h", "volume_24h", "high_24h", "volume_30d", "best_bid", "best_ask", "last_size", "hour_of_day"]
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    data = pd.get_dummies(data, columns=["side"], drop_first=False)
    
    return data


def preprocess_data():
    full_channel, ticker = load_data()
    
    full_channel_processed = preprocess_full_channel_data(full_channel)
    ticker_processed = preprocess_ticker_data(ticker)
    
    save_data(full_channel_processed, "../data/processed/full_channel_processed.csv")
    save_data(ticker_processed, "../data/processed/ticker_processed.csv")
    print("Data preprocessing complete and files saved.")

if __name__ == "__main__":
    preprocess_data()
