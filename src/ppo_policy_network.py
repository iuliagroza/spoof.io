from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from market_env import MarketEnvironment
from config import Config

class PPOPolicyNetwork(nn.Module):
    def __init__(self, num_features, num_actions):
        super(PPOPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_logits = self.fc3(x)
        action_probs = torch.softmax(action_logits, dim=-1)
        if torch.any(torch.isnan(action_probs)) or torch.any(action_probs < 0) or torch.any(action_probs > 1):
            print("Invalid probabilities detected")
            action_probs = torch.clamp(action_probs, 0.0001, 0.9999)  # Clamp to avoid invalid values
        return Categorical(probs=action_probs)

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

def ppo_update(policy_net, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
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

def safe_float_convert(data):
    try:
        return np.float32(data)
    except:
        return 0.0

if __name__ == "__main__":
    env = MarketEnvironment(Config.PROCESSED_DATA_PATH + "full_channel_enhanced.csv",
                            Config.PROCESSED_DATA_PATH + "ticker_enhanced.csv")
    num_features = len(env.reset())
    num_actions = 3 

    policy_net = PPOPolicyNetwork(num_features, num_actions)
    optimizer = optim.Adam(policy_net.parameters(), lr=Config.PPO_CONFIG['learning_rate'])

    # Training loop
    state = env.reset()
    while not env.done:
        state = np.array([safe_float_convert(x) for x in state], dtype=np.float32)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        dist = policy_net(state_tensor)
        action = dist.sample() 
        next_state, reward, done = env.step(action.item())

        print(f"Reward: {reward}, New State: {next_state if next_state is not None else 'End of Data'}")
        if not done:
            state = next_state

    torch.save(policy_net.state_dict(), Config.MODEL_SAVE_PATH + 'final_policy_network.pth')
    print("Training complete, model saved.")
