from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.log_config import setup_logger
from src.config import Config
from src.market_env import MarketEnvironment


# Set up logging to save environment logs for debugging purposes
logger = setup_logger(__name__, Config.LOG_PPO_POLICY_NETWORK_PATH)


class PPOPolicyNetwork(nn.Module):
    """
    A neural network for implementing Proximal Policy Optimization (PPO).
    This network uses a simple feed-forward design suitable for policy-based RL methods.
    """

    def __init__(self, num_features, num_actions):
        """
        Initializes the PPO policy network with a specified number of features and actions.

        Args:
            num_features (int): Number of input features.
            num_actions (int): Number of output actions.
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
        Forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor containing feature data.

        Returns:
            torch.Tensor: The output tensor containing the action logits.
        """
        return self.layers(x)


def get_discounted_rewards(rewards, gamma):
    """
    Calculate the discounted rewards backwards through time.

    Args:
        rewards (list of floats): Rewards obtained from the environment.
        gamma (float): Discount factor.

    Returns:
        list of floats: Discounted rewards.
    """
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return discounted_rewards


def compute_advantages(rewards, values, gamma, lam):
    """
    Compute the Generalized Advantage Estimation (GAE) for more stable policy updates.

    Args:
        rewards (list of floats): Rewards obtained from the environment.
        values (list of floats): Value function estimates at each timestep.
        gamma (float): Discount factor.
        lam (float): Lambda parameter for GAE.

    Returns:
        list of floats: Advantage estimates.
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
    Perform a single update step of the PPO algorithm.

    Args:
        network (PPOPolicyNetwork): The policy network model.
        optimizer (torch.optim.Optimizer): Optimizer for updating network weights.
        states (torch.Tensor): Tensor of states.
        actions (torch.Tensor): Tensor of actions taken.
        old_log_probs (torch.Tensor): Tensor of log probabilities of actions under the old policy.
        advantages (torch.Tensor): Tensor of advantage estimates.
        returns (torch.Tensor): Tensor of discounted returns.
        clip_param (float): Clipping parameter used in PPO loss to limit the policy update step.

    Returns:
        float: The loss value computed during the PPO update.
    """
    try:
        logits = network(states)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        ratios = torch.exp(new_log_probs - old_log_probs.detach())

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = 0.5 * (returns - dist.probs.mean()).pow(2).mean()  # ensure probs are never zero
        loss = actor_loss + critic_loss - Config.PPO_CONFIG['ent_coef'] * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), Config.PPO_CONFIG['max_grad_norm'])
        optimizer.step()

        return loss.item()
    except Exception as e:
        logger.error(f"Failed during PPO update: {e}")
        raise Exception(f"PPO update failed: {e}")


# Test
if __name__ == "__main__":
    try:
        # Enviornment and network setup
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
            logger.info(f"Loss after update: {loss}")
            logger.info(f"Step: {Config.PPO_CONFIG['n_steps']-1}, Action: {action.item()}, Reward: {reward}, Anomaly Score: {anomaly_scores[-1]}, Spoofing Threshold: {spoofing_thresholds[-1]}, Cumulative Reward: {sum(rewards)}, Next Value: {next_value}")
            logger.info(f"Completed {Config.PPO_CONFIG['n_steps']} steps with final reward: {rewards[-1]}")

        torch.save(network.state_dict(), Config.PPO_POLICY_NETWORK_MODEL_PATH)
        logger.info("Model training complete and saved.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
