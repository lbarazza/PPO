import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    """
    Neural network for the actor.

    Args:
        nS (int): length of state vector
        nA (int): dimensionality of action
    """

    def __init__(self, nS, nA):
        super(Actor, self).__init__()

        self.h = nn.Linear(nS, 100)
        self.i = nn.Linear(100, 100)
        self.out_mean = nn.Linear(100, nA)
        self.out_std  = nn.Linear(100, nA)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.tensor): input of the network

        Returns:
            (torch.distributions.Normal): Normal distribution for the action
        """

        x = F.relu(self.h(x))
        x = F.relu(self.i(x))
        mean = self.out_mean(x)
        std  = torch.abs(self.out_std(x))
        return Normal(mean, std)


class Critic(nn.Module):
    """
    Neural network for the critic.

    Args:
        nS (int): length of state vector
    """

    def __init__(self, nS):
        super(Critic, self).__init__()

        self.h = nn.Linear(nS, 100)
        self.i = nn.Linear(100, 100)
        self.out = nn.Linear(100, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.tensor): input of the network

        Returns:
            v (torch.tensor): predicted value of the V function
        """

        x = F.relu(self.h(x))
        x = F.relu(self.i(x))
        v = self.out(x)
        return v


class ActorCritic(nn.Module):

    def __init__(self, nS, nA):
        super(ActorCritic, self).__init__()

        self.h = nn.Linear(nS, 100)
        self.i = nn.Linear(100, 100)
        self.out_mean = nn.Linear(100, nA)
        self.out_std  = nn.Linear(100, nA)
        self.out_v    = nn.Linear(100, 1)

    def forward(self, x):

        x = F.relu(self.h(x))
        x = F.relu(self.i(x))
        mean = self.out_mean(x)
        std  = torch.abs(self.out_std(x))
        v = self.out_v(x)
        return Normal(mean, std), v
