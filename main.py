import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from ppo import PPOAgent
from datetime import datetime

env = gym.make("Pendulum-v0")
seed = 123456
torch.manual_seed(seed)
env.seed(seed)
np.random.seed(seed)

lr_policy=0.001
lr_critic=0.005
gam=0.96
lam=0.93
eps=0.20
batch_size=100
policy_updates=80
v_updates=25
update_freq=1

comment = datetime.now().strftime('%b%d_%H-%M-%S') + f' lr_policy={lr_policy} lr_critic={lr_critic} gamma={gam} lambda={lam} epsilon={eps} batch_size={batch_size} policy_updates={policy_updates} v_updates={v_updates} update_freq={update_freq}'
tb = SummaryWriter("runs/" + comment)

agent = PPOAgent(
    nS = env.observation_space.shape[0],
    nA = env.action_space.shape[0],
    lr_policy=lr_policy,
    lr_critic=lr_critic,
    gam=gam,
    lam=lam,
    eps=eps,
    batch_size=batch_size,
    policy_updates=policy_updates,
    v_updates=v_updates,
    update_freq=update_freq
)

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
        critic_loss = agent.step(state, action, reward, done)

        state = next_state
        episode_reward += reward
        if done:
            tb.add_scalar("Rewards", episode_reward, n_episode)
            if not None: tb.add_scalar("Critic Loss", critic_loss, n_episode)
            returns.append(episode_reward)
            print("Episode n. {:6d}   Return: {:9.2f}   Avg. Return: {:9.2f}".format(n_episode, episode_reward, np.mean(returns)))
            break

env.close()
tb.close()
