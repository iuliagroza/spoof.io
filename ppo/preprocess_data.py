import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from ppo.utils.log_config import setup_logger
from ppo.utils.load_json_data import load_json_data
from ppo.utils.save_data import save_data
from ppo.config import Config


logger = setup_logger('preprocess_data')


def add_time_features(data):
    """ 
    Adds time-based features to a DataFrame by extracting hour of the day from the timestamp.
    
    Args:
        data (pd.DataFrame): The DataFrame with a 'time' column in datetime format.
    
    Returns:
        pd.DataFrame: The DataFrame with an additional 'hour_of_day' feature.
    """
    try:
        data['time'] = pd.to_datetime(data['time'])
        data['hour_of_day'] = data['time'].dt.hour
        return data
    except KeyError as e:
        logger.error(f"Error: The 'time' column is missing. {e}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while adding time features. {e}")
        return None


def create_numeric_transformer():
    """ 
    Creates a pipeline for processing numeric features. This includes imputation of missing values with the median and scaling features between 0 and 1.
    
    Returns:
        Pipeline: A configured pipeline for numeric feature transformations.
    """
    try:
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])
    except Exception as e:
        logger.error(f"An error occurred while creating the numeric transformer. {e}")
        return None


def create_categorical_transformer():
    """ 
    Creates a pipeline for processing categorical features. This includes imputation of missing values with a placeholder and encoding of categorical features as one-hot vectors.
    
    Returns:
        Pipeline: A configured pipeline for categorical feature transformations.
    """
    try:
        categories = Config.FULL_CHANNEL_CATEGORICAL_MAP
        transformers = []
        for key in categories.keys():
            transformer_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(categories=[categories[key]], handle_unknown='ignore'))
            ])
            transformers.append((key, transformer_pipeline, [key]))
        
        # Handle all categorical columns with a default transformer in case some are completely NaN
        default_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        return ColumnTransformer(transformers, remainder=default_transformer)
    except Exception as e:
        logger.error(f"An error occurred while creating the categorical transformer. {e}")
        return None


def get_feature_names(column_transformer):
    """
    Generate feature names from the column transformer. This function specifically handles extracting feature names
    from transformers within a ColumnTransformer, ensuring all transformations including those from OneHotEncoder
    result in appropriately named features.

    Args:
        column_transformer (ColumnTransformer): The ColumnTransformer instance from which to extract feature names.

    Returns:
        List[str]: A list containing all feature names derived from the transformers within the ColumnTransformer.
    """
    feature_names = []
    for transformer_name, transformer, original_features in column_transformer.transformers_[:-1]:  # Ignore the remainder transformer
        if transformer_name == 'drop':
            continue

        if hasattr(transformer, 'get_feature_names_out'):
            if isinstance(transformer, Pipeline):
                # If pipeline, the last step is usually the one with get_feature_names_out
                transformer = transformer.steps[-1][1]
            names = transformer.get_feature_names_out(original_features)
        else:
            # If no get_feature_names_out, use the original feature name
            names = original_features

        processed_names = [name.split('__')[-1] for name in names]
        feature_names.extend(processed_names)

    return feature_names


def preprocess_full_channel_data(data):
    """ 
    Processes the full channel data including adding time features, imputation, and encoding transformations.
    
    Args:
        data (pd.DataFrame): The DataFrame to process.
    
    Returns:
        pd.DataFrame: The preprocessed DataFrame ready for model consumption.
    """
    try:
        identifiers = data[['order_id', 'time']].copy()

        data = add_time_features(data)
        if data is None:
            return None
        
        # Convert categorical columns to string to ensure type safety
        for col in Config.FULL_CHANNEL_CATEGORICAL_COLUMNS:
             data[col] = data[col].astype(str)

        if 'size' not in data.columns or data['size'].isna().all():
            data['size'] = 0

        data.fillna({'remaining_size': 0, 'price': data['price'].mean()}, inplace=True)
        data['remaining_size_change'] = data.groupby('order_id')['remaining_size'].diff().fillna(0)

        preprocessor = ColumnTransformer([
            ('num', create_numeric_transformer(), Config.FULL_CHANNEL_NUMERIC_COLUMNS),
            ('cat', create_categorical_transformer(), Config.FULL_CHANNEL_CATEGORICAL_COLUMNS)
        ], remainder='drop')

        data_transformed = preprocessor.fit_transform(data)
        feature_names = get_feature_names(preprocessor)
        data_processed = pd.DataFrame(data_transformed, columns=feature_names, index=data.index)
        data_processed['hour_of_day'] = data['hour_of_day']
        data_processed = pd.concat([identifiers.reset_index(drop=True), data_processed.reset_index(drop=True)], axis=1)

        return data_processed
    except KeyError as e:
        logger.error(f"Error: A required column is missing. {e}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while preprocessing full channel data. {e}")
        return None



def preprocess_ticker_data(data):
    """ 
    Processes the ticker data including adding time features, scaling, and one-hot encoding.
    
    Args:
        data (pd.DataFrame): The DataFrame to process.
    
    Returns:
        pd.DataFrame: The preprocessed DataFrame ready for model consumption.
    """
    try:
        data = add_time_features(data)
        if data is None:
            return None
        data.drop(columns=['type', 'product_id', 'time', 'low_24h'], inplace=True)

        data[Config.TICKER_NUMERIC_COLUMNS] = MinMaxScaler().fit_transform(data[Config.TICKER_NUMERIC_COLUMNS])
        for category in Config.TICKER_SIDE_CATEGORIES:
            data[f'side_{category}'] = (data['side'] == category).astype(int)
        data.drop(columns=['side'], inplace=True)

        return data
    except KeyError as e:
        logger.error(f"Error: A required column is missing. {e}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while preprocessing ticker data. {e}")
        return None
    

def preprocess_data():
    """ 
    Main function to load, process, and save the preprocessed full channel and ticker data.
    """
    try:
        logger.info("Loading raw data...")
        full_channel, ticker = load_json_data()
    except Exception as e:
        logger.error(f"An error occurred while loading data. {e}")
        return

    full_channel_processed = preprocess_full_channel_data(full_channel)
    if full_channel_processed is None:
        logger.error("Full channel data preprocessing failed.")
        return

    ticker_processed = preprocess_ticker_data(ticker)
    if ticker_processed is None:
        logger.error("Ticker data preprocessing failed.")
        return

    try:
        logger.info("Saving preprocessed datasets...")
        save_data(full_channel_processed, ticker_processed, Config.FULL_CHANNEL_PROCESSED_PATH, Config.TICKER_PROCESSED_PATH)

        logger.info("Data preprocessing complete and files saved.")
    except Exception as e:
        logger.error(f"An error occurred while saving data. {e}")


# Test
if __name__ == "__main__":
    preprocess_data()
