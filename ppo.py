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
    def __init__(self, input_size, hidden_size, output_size, lr, gamma, lmbda, eps_clip):
        self.state_dim = 4
        self.action_dim = 64
        self.hidden_dim = 2
        self.learning_rate = 0.2

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

    def compute_advantage(rewards, values, gamma, lamda):
        values = values + [0]
        advantage = [0]
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma * R
            advantage.append(R - values[i])
        return advantage[::-1]

    def update(self, policy, value, batch_size, gamma, lamda):
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

            advantage = self.compute_advantage(rewards, values, gamma, lamda)

            policy_loss = []
            for log_prob, adv in zip(log_probs, advantage):
                policy_loss.append(-log_prob * adv)
            policy_loss = torch.cat(policy_loss)