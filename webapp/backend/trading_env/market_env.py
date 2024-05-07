import numpy as np
from django.conf import settings
from src.config import Config

class MarketEnvironment:
    def __init__(self):
        self.spoofing_threshold = Config.DEFAULT_SPOOFING_THRESHOLD
        self.history_window_size = Config.HISTORY_WINDOW_SIZE
        self.feature_weights = Config.FEATURE_WEIGHTS
        self.current_index = 0
        self.done = False
        self.full_channel_data = None
        self.ticker_data = None

    def load_data(self, full_channel_batch, ticker_batch):
        """Loads data batches into the environment."""
        self.full_channel_data = full_channel_batch
        self.ticker_data = ticker_batch
        self.current_index = self.history_window_size

    def reset(self):
        """Resets the environment to start processing a new batch of data."""
        self.current_index = self.history_window_size
        self.done = False
        self.update_threshold()
        return self.get_state()

    def step(self, action):
        """Simulates one step in the environment based on the agent's action."""
        if self.current_index >= len(self.full_channel_data) or self.current_index >= len(self.ticker_data):
            self.done = True
            return None, 0, True, 0, self.spoofing_threshold

        anomaly_score = self.calculate_anomaly_score(self.current_index)
        is_spoofing = anomaly_score > self.spoofing_threshold
        reward = 1 if (action == 1 and is_spoofing) or (action == 0 and not is_spoofing) else -1

        self.current_index += 1
        next_state = self.get_state() if not self.done else None

        return next_state, reward, self.done, anomaly_score, self.spoofing_threshold

    def get_state(self):
        """Constructs the current state from the full channel and ticker data."""
        if self.current_index < self.history_window_size or self.current_index >= len(self.full_channel_data):
            return None
        full_channel_features = self.full_channel_data.iloc[self.current_index - self.history_window_size:self.current_index]
        ticker_features = self.ticker_data.iloc[self.current_index - self.history_window_size:self.current_index]
        return np.concatenate([full_channel_features.values.flatten(), ticker_features.values.flatten()])

    def calculate_anomaly_score(self, index):
        """Calculates the anomaly score for the current index in the dataset."""
        full_row = self.full_channel_data.iloc[index]
        ticker_row = self.ticker_data.iloc[index]
        scores = {feature: np.log1p(abs(full_row.get(feature, 0) + ticker_row.get(feature, 0))) for feature in self.feature_weights}
        return sum(self.feature_weights.get(k, 0) * v for k, v in scores.items())

    def update_threshold(self):
        """Dynamically updates the spoofing threshold based on recent anomaly scores."""
        if self.current_index < 50:
            return  # Not enough data to update
        recent_scores = [self.calculate_anomaly_score(i) for i in range(self.current_index - 50, self.current_index)]
        self.spoofing_threshold = np.percentile(recent_scores, 75)