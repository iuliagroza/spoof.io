import numpy as np
from django.conf import settings
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

class MarketEnvironment:
    def __init__(self):
        # Load initial configuration and thresholds
        self.spoofing_threshold = settings.DEFAULT_SPOOFING_THRESHOLD
        self.history_window_size = settings.HISTORY_WINDOW_SIZE
        self.feature_weights = settings.FEATURE_WEIGHTS
        self.current_index = 0
        self.done = False
        self.full_channel_data = None
        self.ticker_data = None

    def load_data(self, full_channel_batch, ticker_batch):
        """ Loads data batches into the environment. """
        self.full_channel_data = full_channel_batch
        self.ticker_data = ticker_batch
        self.current_index = self.history_window_size  # Start processing after initial history window

    def reset(self):
        """ Resets the environment to start processing a new batch of data. """
        self.current_index = self.history_window_size
        self.done = False
        self.update_threshold()
        return self.get_state()

    def step(self, action):
        """ Simulate one step in the environment based on the agent's action. """
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
        """ Constructs the current state from the full channel and ticker data. """
        if self.current_index < self.history_window_size or self.current_index >= len(self.full_channel_data):
            return None
        full_channel_features = self.full_channel_data.iloc[self.current_index - self.history_window_size:self.current_index]
        ticker_features = self.ticker_data.iloc[self.current_index - self.history_window_size:self.current_index]
        return np.concatenate([full_channel_features.values.flatten(), ticker_features.values.flatten()])

    def calculate_anomaly_score(self, index):
        """ Calculate the anomaly score for the current index in the dataset. """
        full_row = self.full_channel_data.iloc[index]
        ticker_row = self.ticker_data.iloc[index]
        scores = {feature: np.log1p(abs(full_row.get(feature, 0) + ticker_row.get(feature, 0))) for feature in self.feature_weights}
        return sum(self.feature_weights.get(k, 0) * v for k, v in scores.items())

    def update_threshold(self):
        """ Dynamically update the spoofing threshold based on recent anomaly scores. """
        if self.current_index < 50:
            return  # Not enough data to update
        recent_scores = [self.calculate_anomaly_score(i) for i in range(self.current_index - 50, self.current_index)]
        self.spoofing_threshold = np.percentile(recent_scores, 75)


def simulate_market_behavior():
    env = MarketEnvironment()
    # Assuming 'get_batch_data' fetches and preprocesses the next batch of data
    full_channel_batch, ticker_batch = get_batch_data()  # Define this function to fetch your data
    env.load_data(full_channel_batch, ticker_batch)
    state = env.reset()
    while not env.done:
        action = np.random.choice([0, 1])  # Example: Random action, replace with model prediction
        state, reward, done, anomaly_score, spoofing_threshold = env.step(action)
        if done:
            print("Batch simulation ended.")
