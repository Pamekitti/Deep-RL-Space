import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
from collections import namedtuple, deque
from torch.distributions import Categorical
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
import time


gym.logger.set_level(40)
env = gym.make("CartPole-v0")
env.seed(0)
print("observation space",env.observation_space.shape)
print("action space",env.action_space.n)


class Policy(nn.Module):
    def __init__(self, state_size=4, action_size=2, seed=0, fc1_units=8):
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


if __name__ == '__main__':
    policy = Policy.to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    n_episodes = 5000
    max_t = 200
    gamma = 0.995
    print_every = 100
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        env.render()
        saved_log_probs = []
        rewards = []
        state = env.reset()
        for t in range(max_t):
            action, lop_prob = policy.act(state)
            saved_log_probs.append(lop_prob)
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                break




