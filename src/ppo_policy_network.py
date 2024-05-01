import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Simulation environment
class MarketEnvironment:
    def __init__(self, data, initial_index=10, history_t=10):
        self.data = data
        self.current_index = initial_index
        self.history_t = history_t
        self.done = False

    def reset(self):
        self.current_index = self.history_t
        self.done = False
        initial_state = self.data.iloc[self.current_index - self.history_t:self.current_index]
        return initial_state.values.flatten()

    def step(self, action):
        # Fetch current market data point
        current_data = self.data.iloc[self.current_index]

        # Calculate anomaly score based on current state
        anomaly_score = self.calculate_anomaly_score(current_data)

        # Example threshold, adjust based on analysis or during hyperparameter tuning
        spoofing_threshold = 0.8

        # Determine if the current state is considered spoofing
        is_spoofing = anomaly_score > spoofing_threshold
        reward = 0

        # Assign rewards based on action correctness
        if action == 1 and is_spoofing:
            reward = 1  # Correctly flagged spoofing
        elif action == 0 and not is_spoofing:
            reward = 1  # Correctly flagged normal behavior
        else:
            reward = -1  # Incorrect action

        # Update to the next state
        self.current_index += 1
        self.done = self.current_index >= len(self.data)
        if not self.done:
            next_state = self.data.iloc[self.current_index - self.history_t:self.current_index]
            return next_state.values.flatten(), reward, self.done
        else:
            return np.zeros(self.data.shape[1] * self.history_t), reward, self.done

    def calculate_anomaly_score(self, current_data):
        # Define how anomaly score is calculated using the current data point
        order_flow_imbalance = abs(current_data['order_flow_imbalance'])
        cancel_to_received_ratio = current_data['cancel_to_received_ratio']
        price_fluctuations = np.std([current_data[f'price_{i}_std'] for i in (5, 10, 15)])

        # Example weighted sum of factors
        anomaly_score = 0.3 * order_flow_imbalance + 0.4 * cancel_to_received_ratio + 0.3 * price_fluctuations
        return anomaly_score



# PPO Policy Network
class PPOPolicyNetwork(nn.Module):
    def __init__(self, num_features, num_actions):
        super(PPOPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs

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

# Train
def train_ppo(env, policy_net, optimizer):
    max_frames = 15000
    frame_idx = 0
    train_rewards = []

    state = env.reset()
    early_stop = False

    while frame_idx < max_frames and not early_stop:
        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []
        entropy = 0

        for _ in range(2048):
            state = torch.FloatTensor(state).unsqueeze(0)
            dist = policy_net(state)

            action = dist.sample()
            next_state, reward, done = env.step(action.numpy())

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            
            log_probs.append(log_prob)
            values.append(dist.mean())
            rewards.append(reward)
            masks.append(1-done)
            states.append(state)
            actions.append(action)

            state = next_state
            frame_idx += 1

            if frame_idx % 1000 == 0:
                test_reward = np.mean([test_env(policy_net) for _ in range(10)])
                train_rewards.append(test_reward)
                plot(frame_idx, train_rewards)
                if test_reward > threshold:  # Define your own threshold
                    early_stop = True

            if done:
                break

        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        next_value = policy_net(next_state).mean().detach()
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        states = torch.cat(states)
        actions = torch.cat(actions)
        advantage = returns - values

        ppo_update(policy_net, optimizer, 4, 64, states, actions, log_probs, returns, advantage)

# Main code
if __name__ == "__main__":
    data = pd.read_csv('your_data.csv')
    env = MarketEnvironment(data)

    # Initialize Policy Network and Optimizer
    num_features = env.state.shape[1]
    num_actions = 3  # Adjust based on your actions
    policy_net = PPOPolicyNetwork(num_features, num_actions)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    train_ppo(env, policy_net, optimizer)
