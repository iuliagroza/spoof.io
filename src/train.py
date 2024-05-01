import torch
import torch.optim as optim
import numpy as np
from ppo_policy_network import PPOPolicyNetwork
from market_env import MarketEnvironment
from config import Config

def train_ppo(env, policy_net, optimizer):
    max_frames = 15000
    frame_idx = 0
    train_rewards = []

    state, _ = env.reset()  # Assuming the reset returns a state and potentially other info
    state = torch.FloatTensor(state).unsqueeze(0)
    early_stop = False

    while frame_idx < max_frames and not early_stop:
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []

        for _ in range(Config.PPO_CONFIG['n_steps']):
            dist = policy_net(state)
            action = dist.sample()
            
            log_prob = dist.log_prob(action)
            value = policy_net(state).mean()  # Assuming you have a method to calculate value

            next_state, reward, done = env.step(action.numpy())
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            masks.append(1 - done)
            states.append(state)
            actions.append(action)

            state = next_state
            frame_idx += 1

            if done:
                break

        if frame_idx % 1000 == 0 or done:
            with torch.no_grad():
                next_value = policy_net(next_state).mean()
            returns = compute_returns(next_value, rewards, masks, Config.PPO_CONFIG['gamma'])

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns)
            values = torch.cat(values)
            states = torch.cat(states)
            actions = torch.cat(actions)
            advantage = returns - values

            ppo_update(policy_net, optimizer, Config.PPO_CONFIG['n_epochs'], Config.PPO_CONFIG['batch_size'],
                       states, actions, log_probs, returns, advantage, Config.PPO_CONFIG['clip_range'])

        if frame_idx % 1000 == 0:
            avg_reward = np.mean(rewards)
            train_rewards.append(avg_reward)
            print(f'Frame {frame_idx}: Average Reward: {avg_reward}')
            if avg_reward > some_defined_threshold:  # Define your threshold
                early_stop = True

        if done:
            state, _ = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0)

    torch.save(policy_net.state_dict(), Config.MODEL_SAVE_PATH + 'ppo_model.pt')

def main():
    env = MarketEnvironment(Config.PROCESSED_DATA_PATH + 'full_channel_enhanced.csv', Config.PROCESSED_DATA_PATH + 'ticker_enhanced.csv')
    num_features = len(env.reset()[0])  # Assuming reset returns a full state representation
    policy_net = PPOPolicyNetwork(num_features, 3)  # Assuming 3 possible actions
    optimizer = optim.Adam(policy_net.parameters(), lr=Config.PPO_CONFIG['learning_rate'])

    train_ppo(env, policy_net, optimizer)

if __name__ == "__main__":
    main()
