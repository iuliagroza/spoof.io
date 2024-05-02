import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketEnvironment:
    def __init__(self, full_channel_path=None, ticker_path=None, initial_index=0, history_t=10, train=True):
        full_channel, ticker = self.load_data(full_channel_path, ticker_path)
        
        # Calculate split index for training
        split_idx_full = int(len(full_channel) * 0.7)
        split_idx_ticker = int(len(ticker) * 0.7)

        # Use only the first 70% of data if training, else use the remaining 30%
        self.full_channel_data = full_channel[:split_idx_full] if train else full_channel[split_idx_full:]
        self.ticker_data = ticker[:split_idx_ticker] if train else ticker[split_idx_ticker:]

        self.current_index = initial_index
        self.history_t = history_t
        self.done = False

        # TODO: HYPERTUNE
        self.spoofing_threshold = 0.8 

    def load_data(self, full_channel_path, ticker_path):
        full_channel = pd.read_csv(full_channel_path) if full_channel_path else pd.DataFrame()
        ticker = pd.read_csv(ticker_path) if ticker_path else pd.DataFrame()
        return full_channel, ticker

    def reset(self):
        self.current_index = self.history_t
        self.done = False
        # Combining features from both datasets at reset
        full_features = self.full_channel_data.iloc[self.current_index - self.history_t:self.current_index].values.flatten()
        ticker_features = self.ticker_data.iloc[self.current_index - self.history_t:self.current_index].values.flatten()
        return np.concatenate([full_features, ticker_features])

    def step(self, action):
        # Calculate anomaly score based on current state
        anomaly_score = self.calculate_anomaly_score(self.current_index)
        is_spoofing = anomaly_score > self.spoofing_threshold
        reward = 1 if (action == 1 and is_spoofing) or (action == 0 and not is_spoofing) else -1

        self.current_index += 1
        if self.current_index >= len(self.full_channel_data) or self.current_index >= len(self.ticker_data):
            self.done = True

        if not self.done:
            full_features = self.full_channel_data.iloc[self.current_index - self.history_t:self.current_index].values.flatten()
            ticker_features = self.ticker_data.iloc[self.current_index - self.history_t:self.current_index].values.flatten()
            next_state = np.concatenate([full_features, ticker_features])  # Concatenated array
        else:
            next_state = None
        return next_state, reward, self.done

    def calculate_anomaly_score(self, index):
        full_row = self.full_channel_data.iloc[index]
        ticker_row = self.ticker_data.iloc[index]

        weights = {
            'order_flow_imbalance': 0.3,
            'spread': 0.4,  # Using spread directly from ticker
            'cancel_to_received_ratio': 0.3
        }
        scores = {
            'order_flow_imbalance': np.log1p(abs(full_row['order_flow_imbalance'])),
            'spread': np.log1p(abs(ticker_row['spread'])) if ticker_row['spread'] != 0 else 0,
            'cancel_to_received_ratio': np.log1p(full_row['cancel_to_received_ratio'])
        }
        anomaly_score = sum(weights[k] * scores[k] for k in weights)
        return anomaly_score

if __name__ == "__main__":
    env = MarketEnvironment("../data/processed/full_channel_enhanced.csv", "../data/processed/ticker_enhanced.csv")
    state = env.reset()
    while not env.done:
        action = np.random.choice([0, 1])  # Randomly choose action
        state, reward, done = env.step(action)
        print(f"Reward: {reward}, New State: {state if state is not None else 'End of Data'}")
