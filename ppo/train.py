import torch
from torch.distributions import Categorical
from torch import optim
from ppo.config import Config
from ppo.market_env import MarketEnvironment
from ppo.ppo_policy_network import PPOPolicyNetwork, ppo_update, get_discounted_rewards, compute_advantages
from ppo.utils.log_config import setup_logger


# Set up logging to save environment logs for debugging purposes
logger = setup_logger('train', Config.LOG_TRAIN_PATH)


def train_model(env: MarketEnvironment, network: PPOPolicyNetwork, optimizer: optim.Optimizer) -> None:
    """
    Trains a PPO policy network using the provided environment and optimizer.

    Args:
        env: The MarketEnvironment instance to use for training.
        network: The PPOPolicyNetwork instance to train.
        optimizer: The optimizer to use for training.

    Returns:
        None
    """
    try:
        state = env.reset()
        if state is not None:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        while not done and state is not None:
            log_probs, values, rewards, states, actions, anomaly_scores, spoofing_thresholds = [], [], [], [], [], [], []
            for _ in range(Config.PPO_CONFIG['n_steps']):
                logits = network(state)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = logits.mean()

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)

                state, reward, done, anomaly_score, spoofing_threshold = env.step(action.item())
                if state is not None:
                    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                rewards.append(reward)
                anomaly_scores.append(anomaly_score)
                spoofing_thresholds.append(spoofing_threshold)

                if done:
                    logger.info("End of episode.")
                    break

            next_value = 0 if done else logits.mean().item()
            discounted_rewards = get_discounted_rewards(rewards + [next_value], Config.PPO_CONFIG['gamma'])
            advantages = compute_advantages(rewards, values + [next_value], Config.PPO_CONFIG['gamma'], Config.PPO_CONFIG['gae_lambda'])

            # Convert lists to tensors
            states = torch.cat(states)
            actions = torch.tensor(actions)
            log_probs = torch.tensor(log_probs)
            returns = torch.tensor(discounted_rewards[:-1])  # exclude the last next_value
            advantages = torch.tensor(advantages)

            loss = ppo_update(network, optimizer, states, actions, log_probs, advantages, returns, Config.PPO_CONFIG['clip_range'])
            logger.info(f"Loss: {loss}, Last Reward: {rewards[-1]}, Total Steps: {_ + 1}")

        torch.save(network.state_dict(), Config.PPO_POLICY_NETWORK_MODEL_PATH)
        logger.info("Model training complete and saved.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")    


if __name__ == "__main__":
    try:
        env = MarketEnvironment(train=True)
        num_features = len(env.reset())
        network = PPOPolicyNetwork(num_features, 2)
        optimizer = torch.optim.Adam(network.parameters(), lr=Config.PPO_CONFIG['learning_rate'])
        train_model(env, network, optimizer)
    except Exception as e:
        logger.error(f"Failed to start the training process: {e}")
