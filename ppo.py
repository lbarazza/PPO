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
import utils


class ppo:
    def __init__(self):
        self.memory = Memory()
        self.policy = Actor(3, 1)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)
        self.critic = Critic(3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.005)#0.001
        self.gam = 0.96
        self.lam = 0.93
        self.eps = 0.20

    @torch.no_grad()
    def choose_action(self, state):
        return self.policy(state).sample()

    def step(self, state, action, reward, done):
        self.memory.add(state, action, reward, done)

    # ( "_" subscript stands for "all of")
    def update(self, batch_size, n_updates, v_updates):
        states, actions, R_, T_ = self.memory.get()
        n_ep = len(R_)

        v_ = self.critic(states)
        A_, rtg_ = utils.gae_rtg((R_, v_, T_), self.gam, self.lam)
        policy_old = copy.deepcopy(self.policy)
        log_probs_old_ = policy_old(states).log_prob(actions)

        for (v, A, rtg, log_probs_old), i in utils.sample_batch((v_, A_, rtg_, log_probs_old_), batch_size, n_updates): # 'i' is the index of the sampled items
            log_probs = self.policy(states).log_prob(actions)[i]

            r = torch.exp(log_probs - log_probs_old)

            l_1 = r * A
            l_2 = torch.clamp(r, 1-self.eps, 1+self.eps) * A

            # TODO: implement entropy
            l_clip = -1/n_ep * torch.sum(torch.min(l_1, l_2))

            self.policy_optimizer.zero_grad()
            l_clip.backward(retain_graph=True)
            self.policy_optimizer.step()

        for (v, rtg), _ in utils.sample_batch((v_, rtg_), batch_size, v_updates):
            critic_loss = 1/n_ep * F.mse_loss(v, rtg)
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

        self.memory.clear()

        return l_clip, critic_loss, v, actions
