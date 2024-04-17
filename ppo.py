import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PPOAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma=1, lmbda=0.9, eps_clip=3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda

        self.policy = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.value = ValueNetwork(self.state_dim, self.hidden_dim)
        self.optimizer = optim.Adam([{'params': self.policy.parameters()}, {'params': self.value.parameters()}], lr=self.learning_rate)


    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.policy(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def compute_advantage(self, rewards, values):
        values = values + [0]
        advantage = [0]
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R
            advantage.append(R - values[i])
        return advantage[::-1]

    def update(self, policy, value, batch_size):
        for _ in range(batch_size):
            rewards, values, log_probs, actions = [], [], [], []

            for _ in range(4):
                action, log_prob = self.select_action(state, policy)
                next_state, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                values.append(value(state).item())
                log_probs.append(log_prob)
                actions.append(action)
                state = next_state
                if done:
                    break

            advantage = self.compute_advantage(rewards, values, self.gamma, self.lmbda)

            policy_loss = []
            for log_prob, adv in zip(log_probs, advantage):
                policy_loss.append(-log_prob * adv)
            policy_loss = torch.cat(policy_loss)

# Hyperparameters
state_dim = 4
action_dim = 64
hidden_dim = 2
learning_rate = 0.2

ppo = PPOAgent(state_dim, action_dim, hidden_dim, learning_rate)