import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy
from collections import deque
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Actor(nn.Module):
    def __init__(self, nS, nA):
        super(Actor, self).__init__()

        self.h = nn.Linear(nS, 100)
        self.i = nn.Linear(100, 100)
        self.out_mean = nn.Linear(100, nA)
        self.out_std  = nn.Linear(100, nA)

    def forward(self, x):
        x = F.relu(self.h(x))
        x = F.relu(self.i(x))
        mean = self.out_mean(x)
        std  = torch.abs(self.out_std(x))
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, nS):
        super(Critic, self).__init__()

        self.h = nn.Linear(nS, 100)
        self.i = nn.Linear(100, 100)
        self.out = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.h(x))
        x = F.relu(self.i(x))
        v = self.out(x)
        return v
