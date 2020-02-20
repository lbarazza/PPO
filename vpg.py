import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy
from collections import deque
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from models import *
from memory import Memory


class vpg:
    def __init__(self):
        self.memory = Memory()
        self.policy = Actor(3, 1)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.005)
        self.critic = Critic(3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.005)
        self.gam = 0.96
        self.lam = 0.93#0.95

    @torch.no_grad()
    def choose_action(self, state):

        return self.policy(state).sample()

    def step(self, state, action, reward, done):
        self.memory.add(state, action, reward, done)

    @torch.no_grad()
    def gae_rtg(self, R, v, T):
        j = 0
        gae = torch.empty(T, 1)
        rtg = torch.empty(T, 1)
        for r in R:
            for i in reversed(range(len(r))):
                t = j + i

                # calculate rewards to go
                rtg[t] = r[i] + self.gam * (rtg[t+1] if i != len(r)-1 else 0)

                # calculate GAE
                delta_t = - v[t] + r[i] + self.gam * (v[t+1] if i != len(r)-1 else 0)
                gae[t] = delta_t + self.gam * self.lam * (gae[t+1] if i != len(r)-1 else 0)

            j += len(r)
        return gae, rtg


    def update(self):
        states, actions, R, T = self.memory.get()
        v = self.critic(states)
        gae, rtg = self.gae_rtg(R, v, T)

        n_ep = len(R)
        dist = self.policy(states)
        log_probs = dist.log_prob(actions)

        vpg_loss = -1/n_ep * torch.sum(log_probs * gae)

        self.policy_optimizer.zero_grad()
        vpg_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        critic_loss = 1/n_ep * F.mse_loss(v, rtg)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.memory.clear()

        return vpg_loss, critic_loss, v, actions
