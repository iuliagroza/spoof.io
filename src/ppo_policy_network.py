from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.log_config import setup_logger
from src.config import Config
from src.market_env import MarketEnvironment


# Set up logging to save environment logs for debugging purposes
logger = setup_logger(Config.LOG_PPO_POLICY_NETWORK_PATH)


class PPOPolicyNetwork(nn.Module):
    """
    A neural network for implementing Proximal Policy Optimization (PPO).
    """

    def __init__(self, num_features, num_actions):
        """
        Initializes the policy network with a simple feed-forward architecture.

        Args:
            num_features (int): Number of input features.
            num_actions (int): Number of possible actions.
        """
        super(PPOPolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        return self.layers(x)


def get_discounted_rewards(rewards, gamma):
    """
    Computes discounted rewards for reinforcement learning.

    Args:
        rewards (list): List of rewards.
        gamma (float): Discount factor.

    Returns:
        list: List of discounted rewards.
    """
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return discounted_rewards


def compute_advantages(rewards, values, gamma, lam):
    """
    Computes the generalized advantage estimation (GAE).

    Args:
        rewards (list): List of rewards.
        values (list): List of values estimated by the critic.
        gamma (float): Discount factor.
        lam (float): Lambda for GAE.

    Returns:
        list: List of advantage estimates.
    """
    advantages = []
    last_adv = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        last_adv = delta + gamma * lam * last_adv
        advantages.insert(0, last_adv)
    return advantages


def ppo_update(network, optimizer, states, actions, old_log_probs, advantages, returns, clip_param):
    """
    Performs a Proximal Policy Optimization (PPO) update on the policy network.

    Args:
        network (nn.Module): The policy network.
        optimizer (torch.optim.Optimizer): The optimizer.
        states (torch.Tensor): Collected state tensors.
        actions (torch.Tensor): Collected action tensors.
        old_log_probs (torch.Tensor): Log probabilities of the actions taken, under the policy at the time.
        advantages (torch.Tensor): Computed advantages.
        returns (torch.Tensor): Discounted returns.
        clip_param (float): The clipping parameter for PPO.

    Returns:
        float: The loss after the update.
    """
    logits = network(states)
    dist = Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()
    ratios = torch.exp(new_log_probs - old_log_probs.detach())  # Importance sampling ratio

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = 0.5 * (returns - dist.mean).pow(2).mean()
    loss = actor_loss + critic_loss - Config.PPO_CONFIG['ent_coef'] * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), Config.PPO_CONFIG['max_grad_norm'])
    optimizer.step()

    return loss.item()


# Test
if __name__ == "__main__":
    # Environment and Network setup
    env = MarketEnvironment()
    num_features = len(env.reset())
    num_actions = 2  # Binary actions: 0 (no spoofing), 1 (spoofing)
    network = PPOPolicyNetwork(num_features, num_actions)
    optimizer = optim.Adam(network.parameters(), lr=Config.PPO_CONFIG['learning_rate'])

    # Training loop
    state = env.reset()
    if state is not None:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    while not done and state is not None:
        log_probs = []
        values = []
        rewards = []
        states = []
        actions = []
        anomaly_scores = []
        spoofing_thresholds = []
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

        if not done:
            next_value = logits.mean().item()
        else:
            next_value = 0

        discounted_rewards = get_discounted_rewards(rewards + [next_value], Config.PPO_CONFIG['gamma'])
        advantages = compute_advantages(rewards, values + [next_value], Config.PPO_CONFIG['gamma'], Config.PPO_CONFIG['gae_lambda'])

        # Convert lists to tensors
        states = torch.cat(states)
        actions = torch.tensor(actions)
        log_probs = torch.tensor(log_probs)
        returns = torch.tensor(discounted_rewards[:-1])  # exclude the last next_value
        advantages = torch.tensor(advantages)

        # PPO update
        loss = ppo_update(network, optimizer, states, actions, log_probs, advantages, returns, Config.PPO_CONFIG['clip_range'])
        logger.info(f"Loss after update: {loss}")
        logger.info(f"Step: {Config.PPO_CONFIG['n_steps']-1}, Action: {action.item()}, Reward: {reward}, Anomaly Score: {anomaly_scores[-1]}, Spoofing Threshold: {spoofing_thresholds[-1]}, Cumulative Reward: {sum(rewards)}, Next Value: {next_value}")
        logger.info(f"Completed {Config.PPO_CONFIG['n_steps']} steps with final reward: {rewards[-1]}")

    # Save model after training
    torch.save(network.state_dict(), Config.PPO_POLICY_NETWORK_MODEL_PATH)
    logger.info("Model training complete and saved.")
