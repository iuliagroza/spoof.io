import numpy as np
from src.utils.log_config import setup_logger
from src.utils.load_csv_data import load_csv_data
from src.config import Config


# Set up logging to save environment logs for debugging purposes
logger = setup_logger(__name__, Config.LOG_MARKET_ENV_PATH)


class MarketEnvironment:
    """
    Represents a market environment for reinforcement learning, handling time-series data for financial trading simulations.
    """

    def __init__(self, initial_index=0, train=True):
        """
        Initializes the MarketEnvironment with data loaded from specified paths in the configuration.

        Args:
            initial_index (int): The starting index for data to simulate the environment's time step.
            train (bool): Flag to determine if the environment is for training or testing, influencing data segmentation.
        """
        try:
            self.full_channel_data, self.ticker_data = load_csv_data(Config.FULL_CHANNEL_ENHANCED_PATH, Config.TICKER_ENHANCED_PATH)
            self.split_data(train)
        except Exception as e:
            logger.error(f"Failed to load data with error: {e}")
            raise Exception(f"Data loading failed: {e}")

        self.current_index = max(initial_index, Config.HISTORY_WINDOW_SIZE)
        self.done = False
        self.spoofing_threshold = Config.DEFAULT_SPOOFING_THRESHOLD  # Initialize dynamic thresholding
        self.update_threshold()

    def split_data(self, train):
        """ 
        Splits the data into training or testing segments based on the configuration ratio.

        Args:
            train (bool): Determines whether the split is for training or testing purposes.
        """
        try:
            split_idx_full = int(len(self.full_channel_data) * Config.TRAIN_TEST_SPLIT_RATIO)
            split_idx_ticker = int(len(self.ticker_data) * Config.TRAIN_TEST_SPLIT_RATIO)
            if train:
                self.full_channel_data = self.full_channel_data[:split_idx_full]
                self.ticker_data = self.ticker_data[:split_idx_ticker]
            else:
                self.full_channel_data = self.full_channel_data[split_idx_full:]
                self.ticker_data = self.ticker_data[split_idx_ticker:]
        except Exception as e:
            logger.error(f"Failed to split data with error: {e}")
            raise Exception(f"Data splitting failed: {e}")

    def reset(self):
        """
        Resets the environment to the initial state, ensuring the starting index is valid and not beyond data bounds.

        Returns:
            np.array: The initial state from concatenated features of both datasets.
        """
        self.current_index = Config.HISTORY_WINDOW_SIZE
        self.done = False
        self.update_threshold()  # Update threshold each episode based on recent trends
        return self.get_state()

    def step(self, action):
        """
        Processes the agent's action and returns the new state, reward, and status if the episode is done.
        Additionally, it returns the anomaly score and spoofing threshold for detailed tracking.

        Args:
            action (int): The action taken by the agent (0 or 1, where 1 might represent spoofing detection).

        Returns:
            tuple: Contains the next state (np.array or None), reward (int), done status (bool), anomaly score (float), and spoofing threshold (float).
        """
        # Terminal state reached: neutral outcome (reward "zero")
        if self.current_index >= len(self.full_channel_data) or self.current_index >= len(self.ticker_data):
            self.done = True
            return None, 0, True, 0, self.spoofing_threshold

        try:
            anomaly_score = self.calculate_anomaly_score(self.current_index)
            is_spoofing = anomaly_score > self.spoofing_threshold
            reward = 1 if (action == 1 and is_spoofing) or (action == 0 and not is_spoofing) else -1

            self.current_index += 1
            next_state = self.get_state() if not self.done else None
            return next_state, reward, self.done, anomaly_score, self.spoofing_threshold
        except Exception as e:
            logger.error(f"An error occurred during the environment step: {e}")
            raise Exception(f"Step processing failed: {e}")

    def get_state(self):
        """
        Generates the current state by concatenating historical features from both full channel and ticker datasets.

        Returns:
            np.array: Concatenated array of historical market features, ensuring all data types are floats.
        """
        try:
            full_channel_features = self.full_channel_data.iloc[
                self.current_index - Config.HISTORY_WINDOW_SIZE:self.current_index
            ].astype(float).values.flatten()

            ticker_features = self.ticker_data.iloc[
                self.current_index - Config.HISTORY_WINDOW_SIZE:self.current_index
            ].astype(float).values.flatten()

            return np.concatenate([full_channel_features, ticker_features])
        except Exception as e:
            logger.error(f"Failed to retrieve state at index {self.current_index}: {e}")
            raise Exception(f"State retrieval failed: {e}")

    def calculate_anomaly_score(self, index):
        """
        Calculates an anomaly score based on predefined feature weights and market data features at a given index.

        Args:
            index (int): Index of the current state in the dataset.

        Returns:
            float: Computed anomaly score, a higher score suggests higher likelihood of spoofing.
        """
        try:
            full_row = self.full_channel_data.iloc[index]
            ticker_row = self.ticker_data.iloc[index]
            scores = {feature: np.log1p(abs(row[feature])) for row in (full_row, ticker_row) for feature in Config.FEATURE_WEIGHTS if feature in row}
            return sum(Config.FEATURE_WEIGHTS.get(k, 0) * v for k, v in scores.items())
        except Exception as e:
            logger.error(f"Error calculating anomaly score at index {index}: {e}")
            raise Exception(f"Anomaly score calculation failed: {e}")

    def update_threshold(self):
        """
        Dynamically updates the spoofing threshold based on recent anomaly scores to adapt to changing market conditions.
        """
        try:
            recent_scores = [self.calculate_anomaly_score(i) for i in range(max(0, self.current_index - 50), self.current_index)]
            self.spoofing_threshold = np.percentile(recent_scores, 75)
        except Exception as e:
            logger.error(f"Failed to update spoofing threshold: {e}")
            raise Exception(f"Updating spoofing threshold failed: {e}")


# Test
if __name__ == "__main__":
    env = MarketEnvironment()
    state = env.reset()
    while not env.done:
        action = np.random.choice([0, 1])  # Randomly choose action
        try:
            state, reward, done, anomaly_score, spoofing_threshold = env.step(action)
            logger.info(f"Action: {action}, Reward: {reward}, Anomaly Score: {anomaly_score}, Spoofing Threshold: {spoofing_threshold}")
            if done:
                logger.info("Simulation ended.")
        except Exception as e:
            logger.error(f"An error occurred during the simulation: {e}")
            break
