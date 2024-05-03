import torch
from torch.distributions import Categorical
from torch import optim
from src.config import Config
from src.market_env import MarketEnvironment
from src.ppo_policy_network import PPOPolicyNetwork, ppo_update, get_discounted_rewards, compute_advantages
from src.utils.log_config import setup_logger


# Set up logging for training
logger = setup_logger(__name__, Config.LOG_TRAIN_PATH)


def main():
    # Initialize the market environment
    env = MarketEnvironment()
    state = env.reset()
    num_features = len(state)
    num_actions = 2  # 0 for no action, 1 for spoofing action

    # Initialize the PPO policy network and optimizer
    network = PPOPolicyNetwork(num_features, num_actions)
    optimizer = optim.Adam(network.parameters(), lr=Config.PPO_CONFIG['learning_rate'])

    # Main training loop
    done = False
    while not done:
        states, actions, rewards, log_probs, values = [], [], [], [], []
        for _ in range(Config.PPO_CONFIG['n_steps']):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = network(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = logits.mean()  # Approximate the value function

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)

            state, reward, done, anomaly_score, spoofing_threshold = env.step(action.item())
            rewards.append(reward)

            if done:
                break

        # Prepare for PPO update
        next_value = logits.mean().item() if not done else 0
        discounted_rewards = get_discounted_rewards(rewards + [next_value], Config.PPO_CONFIG['gamma'])
        advantages = compute_advantages(rewards, values + [next_value], Config.PPO_CONFIG['gamma'], Config.PPO_CONFIG['gae_lambda'])

        # Convert lists to tensors for PPO update
        states = torch.cat(states)
        actions = torch.tensor(actions)
        log_probs = torch.tensor(log_probs)
        returns = torch.tensor(discounted_rewards[:-1])
        advantages = torch.tensor(advantages)

        # Perform PPO update
        loss = ppo_update(network, optimizer, states, actions, log_probs, advantages, returns, Config.PPO_CONFIG['clip_range'])
        logger.info(f"Loss: {loss}, Last Reward: {rewards[-1]}, Total Steps: {_ + 1}")

    # Save the trained model
    torch.save(network.state_dict(), Config.PPO_POLICY_NETWORK_MODEL_PATH)
    logger.info("Training complete and model saved.")


if __name__ == "__main__":
    main()
