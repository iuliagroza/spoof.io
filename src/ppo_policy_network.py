from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.utils.log_config import setup_logger
from src.config import Config
from src.market_env import MarketEnvironment


# Set up logging to save environment logs for debugging purposes
logger = setup_logger(Config.LOG_PPO_POLICY_NETWORK_PATH)


class PPOPolicyNetwork(nn.Module):
    """ 
    Policy Network for the Proximal Policy Optimization algorithm. 
    """

    def __init__(self, num_features, num_actions):
        """
        Initializes the network with three fully connected layers.

        Args:
            num_features (int): Number of input features.
            num_actions (int): Number of possible actions.
        """
        super(PPOPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        """ 
        Performs the forward pass in the network. 
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_logits = self.fc3(x)
        action_probs = torch.softmax(action_logits, dim=-1)
        return Categorical(probs=action_probs)

def compute_returns(next_value, rewards, masks, gamma=Config.PPO_CONFIG['gamma']):
    """
    Compute the returns for each time step, using the rewards and the discount factor.

    Args:
        next_value (float): The estimated value of the subsequent state.
        rewards (list): List of rewards obtained.
        masks (list): List of masks indicating whether the episode is continuing.
        gamma (float): Discount factor.
    
    Returns:
        list: The list of computed returns.
    """
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
    """
    Generator for mini-batches of training data.

    Args:
        mini_batch_size (int): Size of the mini-batch.
        states (torch.Tensor): State inputs.
        actions (torch.Tensor): Action outputs.
        log_probs (torch.Tensor): Logarithms of the probabilities of actions taken.
        returns (torch.Tensor): Returns computed from rewards.
        advantages (torch.Tensor): Advantage estimates.
    """
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantages[rand_ids, :]

def ppo_update(policy_net, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=Config.PPO_CONFIG['clip_range']):
    """
    Update the policy network using the PPO algorithm.

    Args:
        policy_net (PPOPolicyNetwork): The policy network to update.
        optimizer (torch.optim.Optimizer): Optimizer.
        ppo_epochs (int): Number of epochs to update over.
        mini_batch_size (int): Size of each mini-batch.
        states, actions, log_probs, returns, advantages: Training data tensors.
        clip_param (float): Clipping parameter for PPO.
    """
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist = policy_net(state)
            entropy = -torch.sum(dist.mean() * torch.log(dist))
            new_log_probs = dist.log_prob(action)
            ratio = (new_log_probs - old_log_probs).exp()

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            loss = -torch.min(surr1, surr2) + 0.5 * ((return_ - dist.mean()) ** 2) - 0.01 * entropy
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

# Test
if __name__ == "__main__":
    env = MarketEnvironment(Config.FULL_CHANNEL_ENHANCED_PATH, Config.TICKER_ENHANCED_PATH)
    num_features = len(env.reset())
    num_actions = 2

    policy_net = PPOPolicyNetwork(num_features, num_actions)
    optimizer = optim.Adam(policy_net.parameters(), lr=Config.PPO_CONFIG['learning_rate'])

    # Training loop
    state = env.reset()
    while not env.done:
        state = np.array([np.float32(x) for x in state], dtype=np.float32)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        dist = policy_net(state_tensor)
        action = dist.sample()
        next_state, reward, done = env.step(action.item())
        logger.info(f"Reward: {reward}")
        logger.debug(f"New State: {next_state if next_state is not None else 'End of Data'}")
        if not done:
            state = next_state

    torch.save(policy_net.state_dict(), Config.PPO_POLICY_NETWORK_MODEL_PATH)
    logger.info("Training complete, model saved.")
