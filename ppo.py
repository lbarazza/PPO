import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from models import *
from memory import Memory
import copy
import utils


class PPOAgent:
    def __init__(self, nS, nA, lr_policy, lr_critic, gam, lam, eps, batch_size, policy_updates, v_updates):
        self.memory = Memory()
        self.policy = Actor(nS, nA)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.critic = Critic(nS)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)#0.001
        self.gam = gam
        self.lam = lam
        self.eps = eps
        self.batch_size = batch_size
        self.policy_updates = policy_updates
        self.v_updates = v_updates

    @torch.no_grad()
    def choose_action(self, state):
        return self.policy(state).sample()

    def step(self, state, action, reward, done):
        self.memory.add(state, action, reward, done)

    def update(self):
        states, actions, R, T = self.memory.get()
        n_ep = len(R)

        v = self.critic(states)
        A, rtg = utils.gae_rtg((R, v, T), self.gam, self.lam)
        policy_old = copy.deepcopy(self.policy)
        log_probs_old = policy_old(states).log_prob(actions)

        # "_" subscript indicates "sampled from"
        for (v_, A_, rtg_, log_probs_old_), i in utils.sample_batch((v, A, rtg, log_probs_old), self.batch_size, self.policy_updates):
            log_probs_ = self.policy(states).log_prob(actions)[i]

            r_ = torch.exp(log_probs_ - log_probs_old_)

            l_1 = r_ * A_
            l_2 = torch.clamp(r_, 1-self.eps, 1+self.eps) * A_

            # TODO: implement entropy
            # TODO: merge policy and critic
            l_clip = -1/n_ep * torch.sum(torch.min(l_1, l_2))

            self.policy_optimizer.zero_grad()
            l_clip.backward(retain_graph=True)
            self.policy_optimizer.step()

        for (v_, rtg_), _ in utils.sample_batch((v, rtg), self.batch_size, self.v_updates):
            critic_loss = 1/n_ep * F.mse_loss(v_, rtg_)
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

        self.memory.clear()

        return F.mse_loss(v, rtg)
