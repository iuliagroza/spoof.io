import numpy as np
import logging
from src.utils.load_csv_data import load_csv_data
from src.config import Config


# Set up logging to save environment logs for debugging purposes
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format=Config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(Config.LOG_MARKET_ENV_PATH, mode='w'),  # Log file overwritten at each run
        logging.StreamHandler()
    ]
)


class MarketEnvironment:
    """
    Represents a market environment for reinforcement learning, handling time-series data for financial trading simulations.
    """

    def __init__(self, initial_index=0, history_t=10, train=True):
        """
        Initializes the MarketEnvironment with data loaded from specified paths in the configuration.

        Args:
            initial_index (int): The starting index for data to simulate the environment's time step.
            history_t (int): Number of past time steps to consider for generating a state.
            train (bool): Flag to determine if the environment is for training or testing, influencing data segmentation.
        """
        try:
            full_channel, ticker = load_csv_data(Config.FULL_CHANNEL_ENHANCED_PATH, Config.TICKER_ENHANCED_PATH)
        except Exception as e:
            logging.error(f"Failed to load data with error: {e}")
            raise

        # Calculate split index for training
        split_idx_full = int(len(full_channel) * Config.TRAIN_TEST_SPLIT_RATIO)
        split_idx_ticker = int(len(ticker) * Config.TRAIN_TEST_SPLIT_RATIO)

        self.full_channel_data = full_channel[:split_idx_full] if train else full_channel[split_idx_full:]
        self.ticker_data = ticker[:split_idx_ticker] if train else ticker[split_idx_ticker:]
        self.current_index = max(initial_index, history_t)
        self.history_t = history_t
        self.done = False
        self.spoofing_threshold = Config.DEFAULT_SPOOFING_THRESHOLD

    def reset(self):
        """
        Resets the environment to the initial state, ensuring the starting index is valid and not beyond data bounds.

        Returns:
            np.array: The initial state from concatenated features of both datasets.
        """
        self.current_index = self.history_t
        self.done = False
        return self.get_state()

    def step(self, action):
        """
        Processes the agent's action and returns the new state, reward, and a flag indicating if the episode is done.

        Args:
            action (int): The action taken by the agent (0 or 1).

        Returns:
            tuple: A tuple containing the next state (np.array or None), the reward (int), and the done flag (bool).
        """
        if self.current_index >= len(self.full_channel_data) or self.current_index >= len(self.ticker_data):
            self.done = True
            return None, 0, True

        try:
            anomaly_score = self.calculate_anomaly_score(self.current_index)
            is_spoofing = anomaly_score > self.spoofing_threshold
            reward = 1 if (action == 1 and is_spoofing) or (action == 0 and not is_spoofing) else -1

            self.current_index += 1
            if self.current_index >= len(self.full_channel_data) or self.current_index >= len(self.ticker_data):
                self.done = True

            next_state = self.get_state() if not self.done else None
            return next_state, reward, self.done
        except Exception as e:
            logging.error(f"An error occurred during the environment step: {e}")
            raise

    def get_state(self):
        """
        Generates the current state by concatenating historical features from both full channel and ticker datasets.

        Returns:
            np.array: Concatenated array of historical market features.
        """
        try:
            full_features = self.full_channel_data.iloc[self.current_index - self.history_t:self.current_index].values.flatten()
            ticker_features = self.ticker_data.iloc[self.current_index - self.history_t:self.current_index].values.flatten()
            return np.concatenate([full_features, ticker_features])
        except Exception as e:
            logging.error(f"Failed to retrieve state at index {self.current_index}: {e}")
            raise

    def calculate_anomaly_score(self, index):
        """
        Calculates an anomaly score based on predefined feature weights and market data features at a given index.

        Args:
            index (int): Index of the current state in the dataset.

        Returns:
            float: Computed anomaly score.
        """
        try:
            full_row = self.full_channel_data.iloc[index]
            ticker_row = self.ticker_data.iloc[index]
            weights = Config.FEATURE_WEIGHTS
            scores = {
                'order_flow_imbalance': np.log1p(abs(full_row['order_flow_imbalance'])),
                'spread': np.log1p(abs(ticker_row['spread'])) if ticker_row['spread'] != 0 else 0,
                'cancel_to_received_ratio': np.log1p(abs(full_row['cancel_to_received_ratio']))
            }
            return sum(weights[k] * scores[k] for k in weights)
        except Exception as e:
            logging.error(f"Error calculating anomaly score at index {index}: {e}")
            raise


# Test
if __name__ == "__main__":
    env = MarketEnvironment()
    state = env.reset()
    while not env.done:
        action = np.random.choice([0, 1])  # Randomly choose action
        try:
            state, reward, done = env.step(action)
            logging.info(f"Reward: {reward}")
            logging.debug(f"New State: {state if state is not None else 'End of Data'}")
        except Exception as e:
            logging.error(f"An error occurred during the simulation: {e}")
            break
