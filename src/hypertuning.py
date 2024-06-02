import torch
from torch import optim
import pandas as pd
from joblib import Parallel, delayed
from src.config import Config
from src.market_env import MarketEnvironment
from src.ppo_policy_network import PPOPolicyNetwork, ppo_update, get_discounted_rewards, compute_advantages
from src.utils.log_config import setup_logger
from src.train import train_model
from src.test import test_model, save_plots

logger = setup_logger('hypertuning', Config.LOG_TRAIN_PATH)

def evaluate_hyperparameters(learning_rate, batch_size, epochs, spoofing_threshold, feature_weights_key):
    logger.info(f"Testing combination: LR={learning_rate}, Batch Size={batch_size}, Epochs={epochs}, Spoofing Threshold={spoofing_threshold}, Feature Weights={feature_weights_key}")

    Config.PPO_CONFIG['learning_rate'] = learning_rate
    Config.PPO_CONFIG['batch_size'] = batch_size
    Config.PPO_CONFIG['n_epochs'] = epochs
    Config.DEFAULT_SPOOFING_THRESHOLD = spoofing_threshold
    Config.FEATURE_WEIGHTS = Config.FEATURE_WEIGHTS_CONFIGS[feature_weights_key]

    try:
        # Training phase
        env = MarketEnvironment(train=True)
        num_features = len(env.reset())
        network = PPOPolicyNetwork(num_features, 2)
        optimizer = optim.Adam(network.parameters(), lr=Config.PPO_CONFIG['learning_rate'])
        loss_data = train_model(env, network, optimizer)

        # Testing phase
        env = MarketEnvironment(train=False)
        model = PPOPolicyNetwork(num_features, 2)
        model.load_state_dict(torch.load(Config.PPO_POLICY_NETWORK_MODEL_PATH))
        model.eval()
        data = test_model(env, model)

        total_reward = data['rewards'].sum()
        logger.info(f"Total Reward for combination LR={learning_rate}, Batch Size={batch_size}, Epochs={epochs}, Spoofing Threshold={spoofing_threshold}, Feature Weights={feature_weights_key}: {total_reward}")

        # Save plots including loss data
        save_plots(data, loss_data)
        
        return total_reward, learning_rate, batch_size, epochs, spoofing_threshold, feature_weights_key

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None

def tune_hyperparameters():
    hyperparameters = Config.HYPERPARAMETERS

    results = Parallel(n_jobs=-1)(delayed(evaluate_hyperparameters)(
        lr, bs, epoch, st, fw_key
    ) for lr in hyperparameters['learning_rate']
      for bs in hyperparameters['batch_size']
      for epoch in hyperparameters['n_epochs']
      for st in hyperparameters['spoofing_threshold']
      for fw_key in hyperparameters['feature_weights'].keys())

    # Filter out failed runs
    results = [res for res in results if res is not None]

    # Convert results to DataFrame and save
    df_results = pd.DataFrame(results, columns=['total_reward', 'learning_rate', 'batch_size', 'epochs', 'spoofing_threshold', 'feature_weights'])
    df_results = df_results.sort_values(by='total_reward', ascending=False)  # Sort by total_reward
    df_results.to_csv(Config.OUTPUT_PATH + 'hyperparameter_tuning_results.csv', index=False)
    df_results.to_html(Config.OUTPUT_PATH + 'hyperparameter_tuning_results.html', index=False)
    logger.info("Hyperparameter tuning completed and results saved.")

if __name__ == "__main__":
    tune_hyperparameters()
