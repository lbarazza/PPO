import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy
from collections import deque
import numpy as np
from torch.utils.tensorboard import SummaryWriter

env = gym.make("Pendulum-v0")
seed = 123456#1234567
#random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)
np.random.seed(seed)


class actor(nn.Module):
    def __init__(self, nS, nA):
        super(actor, self).__init__()

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

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.episode_rewards = []

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def add(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.episode_rewards.append(reward)

        if done:
            self.rewards.append(self.episode_rewards)
            self.episode_rewards = []

    def get(self):
        states = torch.tensor(self.states).float()
        actions = torch.tensor(self.actions).float()
        # I have to flatten the rewards before convertingh to tensor because episodes mught not all have equal length in another environment
        n_episodes = len(self.rewards)
        return states, actions, self.rewards

class ppo:
    def __init__(self):
        self.memory = Memory()
        self.policy = actor(3, 1)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.005)
        self.critic = Critic(3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.005)
        self.gam = 0.96
        self.lam = 0.93#0.95


    def choose_action(self, state):
        with torch.no_grad():
            dist = self.policy(state)
            action = dist.sample()
        return action

    def step(self, state, action, reward, done):
        self.memory.add(state, action, reward, done)

    def update(self):
        states, actions, R = self.memory.get()
        dist = self.policy(states)
        log_probs = dist.log_prob(actions)
        v = self.critic(states)

        with torch.no_grad():
            R = torch.tensor(R).view(-1, 1)
            gae = torch.empty_like(R)
            rtg = torch.empty_like(R)
            for t in reversed(range(R.shape[0])):
                # calculate rewards to go
                rtg[t] = R[t] + self.gam * (rtg[t+1] if t != rtg.shape[0]-1 else 0)

                # calculate GAE
                delta_t = - v[t] + R[t] + self.gam * (v[t+1]   if t != v.shape[0]-1   else 0)
                gae[t] = delta_t + self.gam * self.lam * (gae[t+1] if t != gae.shape[0]-1 else 0)

        vpg_loss = -torch.sum(log_probs * gae)

        self.policy_optimizer.zero_grad()
        vpg_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        critic_loss = F.mse_loss(v, rtg)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.memory.clear()

        return vpg_loss, critic_loss, v, actions

tb = SummaryWriter()

agent = ppo()

n_episode = 0

returns = deque(maxlen=100)
while True:
    n_episode += 1

    episode_reward = 0
    state = env.reset()
    while True:
        env.render()

        # action needs to be a list since this accepts Box Actions
        # the reeward is of the same type as the action that we pass in

        action = [agent.choose_action(torch.tensor(state).unsqueeze(0).float()).item()]
        next_state, reward, done, info = env.step(action) # take a random action
        agent.step(state, action, reward, done)

        state = next_state
        episode_reward += reward
        if done:

            if n_episode%1==0:
                vpg_loss, critic_loss, v_predicted, actions = agent.update()

                tb.add_scalar("Rewards", episode_reward, n_episode)
                tb.add_scalar("Critic Loss", critic_loss, n_episode)
                tb.add_histogram("Actions", actions)
                tb.add_histogram("V-Predicted", v_predicted, n_episode)

            returns.append(episode_reward)
            print("Episode n. {:6d}    Return: {:9.2f}  Avg. Return: {:9.2f}".format(n_episode, episode_reward, np.mean(returns)))

            break

env.close()
tb.close()
