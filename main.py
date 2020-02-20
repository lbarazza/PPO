import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy
from collections import deque
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from vpg import vpg

env = gym.make("Pendulum-v0")
seed = 1234567#1234567
#random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)
np.random.seed(seed)

tb = SummaryWriter()

agent = vpg()

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

                tb.add_scalar("Critic Loss", critic_loss, n_episode)
                tb.add_histogram("Actions", actions)
                tb.add_histogram("V-Predicted", v_predicted, n_episode)
            tb.add_scalar("Rewards", episode_reward, n_episode)

            returns.append(episode_reward)
            print("Episode n. {:6d}    Return: {:9.2f}  Avg. Return: {:9.2f}".format(n_episode, episode_reward, np.mean(returns)))

            break

env.close()
tb.close()
