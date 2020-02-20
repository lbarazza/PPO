import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy
from collections import deque
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.episode_rewards = []
        self.T = 0

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.T = 0

    def add(self, state, action, reward, done):
        self.T += 1
        self.states.append(state)
        self.actions.append(action)
        self.episode_rewards.append(reward)
        if done:
            self.rewards.append(self.episode_rewards)
            self.episode_rewards = []

    def get(self):
        states = torch.tensor(self.states).float()
        actions = torch.tensor(self.actions).float()
        return states, actions, self.rewards, self.T
