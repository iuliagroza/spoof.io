import torch
import numpy as np
from torch import optim
from market_env import MarketEnvironment
from ppo_policy_network import PPOPolicyNetwork, compute_returns, ppo_update
from config import Config

def safe_float_convert(data):
    try:
        return np.float32(data)
    except ValueError:
        return np.nan  # Replace non-convertible elements with NaN

def train_ppo(env, policy_net, optimizer):
    max_frames = 15000
    frame_idx = 0
    train_rewards = []

    state = env.reset()
    state = np.array([safe_float_convert(x) for x in state], dtype=np.float32)
    state = torch.tensor(state).unsqueeze(0)  # Ensure state is correctly formatted

    while frame_idx < max_frames:
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []
        entropy = 0

        for _ in range(Config.PPO_CONFIG['n_steps']):
            dist = policy_net(state)
            if torch.any(torch.isnan(dist.probs)):
                print("NaN detected in action probabilities, skipping update.")
                continue

            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            next_state, reward, done = env.step(action.item())
            if next_state is not None:
                next_state = np.array([safe_float_convert(x) for x in next_state], dtype=np.float32)
                next_state = torch.tensor(next_state).unsqueeze(0)

            log_probs.append(log_prob)
            values.append(dist.mean())
            rewards.append(reward)
            masks.append(1 - done)
            states.append(state)
            actions.append(action)

            state = next_state if not done else torch.tensor(env.reset()).unsqueeze(0)
            frame_idx += 1

            if done:
                break

        next_value = 0 if done else policy_net(next_state).mean().detach()
        returns = compute_returns(next_value, rewards, masks, Config.PPO_CONFIG['gamma'])

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        states = torch.cat(states)
        actions = torch.cat(actions)
        advantage = returns - values

        ppo_update(policy_net, optimizer, Config.PPO_CONFIG['n_epochs'], Config.PPO_CONFIG['batch_size'], states, actions, log_probs, returns, advantage, Config.PPO_CONFIG['clip_range'])

        average_reward = sum(rewards) / len(rewards) if rewards else 0
        train_rewards.append(average_reward)
        print(f"Frame: {frame_idx}, Average Reward: {average_reward}")

    torch.save(policy_net.state_dict(), Config.MODEL_SAVE_PATH + 'final_policy_network.pth')
    print("Training complete, model saved.")

if __name__ == "__main__":
    train_env = MarketEnvironment(Config.PROCESSED_DATA_PATH + 'full_channel_enhanced.csv',
                                  Config.PROCESSED_DATA_PATH + 'ticker_enhanced.csv')
    num_features = train_env.reset().shape[0]
    policy_net = PPOPolicyNetwork(num_features, 3)
    optimizer = optim.Adam(policy_net.parameters(), lr=Config.PPO_CONFIG['learning_rate'], eps=1e-5)  # Added eps to prevent division by zero in Adam optimizer

    train_ppo(train_env, policy_net, optimizer)
