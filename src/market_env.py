import pandas as pd
import numpy as np
import logging
from src.utils.load_csv_data import load_csv_data
from src.config import Config


logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)


class MarketEnvironment:
    """
    A class to represent a market environment for reinforcement learning, handling time-series data for financial trading.
    """

    def __init__(self, initial_index=0, history_t=10, train=True):
        """
        Initializes the MarketEnvironment.

        Args:
            initial_index (int): The starting index for data to simulate the environment's time step.
            history_t (int): Number of past time steps to consider for generating a state.
            train (bool): Flag to determine if the environment is for training or testing.
        """
        try:
            full_channel, ticker = load_csv_data(Config.PROCESSED_DATA_PATH + 'full_channel_processed.csv', Config.PROCESSED_DATA_PATH + 'ticker_processed.csv')
        except Exception as e:
            logging.error(f"Failed to load data with error: {e}")
            raise

        # Calculate split index for training
        split_idx_full = int(len(full_channel) * Config.TRAIN_SPLIT)
        split_idx_ticker = int(len(ticker) * Config.TRAIN_SPLIT)

        self.full_channel_data = full_channel[:split_idx_full] if train else full_channel[split_idx_full:]
        self.ticker_data = ticker[:split_idx_ticker] if train else ticker[split_idx_ticker:]
        self.current_index = max(initial_index, history_t)
        self.history_t = history_t
        self.done = False
        self.spoofing_threshold = Config.SPOOFING_THRESHOLD

    def reset(self):
        """
        Resets the environment to the initial state.

        Returns:
            np.array: The initial state from concatenated features of both datasets.
        """
        self.current_index = self.history_t
        self.done = False
        return self.get_state()

    def step(self, action):
        """
        Executes a step in the environment based on the action provided.

        Args:
            action (int): The action taken by the agent (0 or 1).

        Returns:
            tuple: A tuple containing the next state, the reward received, and a done flag.
        """
        if self.current_index >= len(self.full_channel_data) or self.current_index >= len(self.ticker_data):
            self.done = True
            return None, 0, True

        anomaly_score = self.calculate_anomaly_score(self.current_index)
        is_spoofing = anomaly_score > self.spoofing_threshold
        reward = 1 if (action == 1 and is_spoofing) or (action == 0 and not is_spoofing) else -1

        self.current_index += 1
        if self.current_index >= len(self.full_channel_data) or self.current_index >= len(self.ticker_data):
            self.done = True

        next_state = self.get_state() if not self.done else None
        return next_state, reward, self.done

    def get_state(self):
        """
        Retrieves the current state of the market based on the history and current index.

        Returns:
            np.array: An array representing the current state.
        """
        full_features = self.full_channel_data.iloc[self.current_index - self.history_t:self.current_index].values.flatten()
        ticker_features = self.ticker_data.iloc[self.current_index - self.history_t:self.current_index].values.flatten()
        return np.concatenate([full_features, ticker_features])

    def calculate_anomaly_score(self, index):
        """
        Calculates the anomaly score based on predefined weights and market data features.

        Args:
            index (int): Index of the current state in the dataset.

        Returns:
            float: The computed anomaly score.
        """
        full_row = self.full_channel_data.iloc[index]
        ticker_row = self.ticker_data.iloc[index]
        weights = Config.ANOMALY_WEIGHTS
        scores = {
            'order_flow_imbalance': np.log1p(abs(full_row['order_flow_imbalance'])),
            'spread': np.log1p(abs(ticker_row['spread'])) if ticker_row['spread'] != 0 else 0,
            'cancel_to_received_ratio': np.log1p(abs(full_row['cancel_to_received_ratio']))
        }
        return sum(weights[k] * scores[k] for k in weights)


# Test
if __name__ == "__main__":
    env = MarketEnvironment()
    state = env.reset()
    while not env.done:
        action = np.random.choice([0, 1])  # Randomly choose action
        state, reward, done = env.step(action)
        print(f"Reward: {reward}, New State: {state if state is not None else 'End of Data'}")
